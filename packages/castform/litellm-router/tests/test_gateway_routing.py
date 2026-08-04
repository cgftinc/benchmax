from __future__ import annotations

import json

import pytest
from castform_router.gateway_routing import (
    GatewayCandidate,
    QwenGatewayRouter,
    load_gateway_candidates,
)
from castform_router.types import HarnessRoutePrediction, HarnessRouteRequest


class FakeQwenScorer:
    router_model_version = "fake-qwen-v1"

    def __init__(self) -> None:
        self.requests: list[HarnessRouteRequest] = []

    def score(
        self,
        request: HarnessRouteRequest,
    ) -> tuple[HarnessRoutePrediction, ...]:
        self.requests.append(request)
        probabilities = {
            "small": 0.86,
            "medium": 0.91,
            "large": 0.97,
        }
        return tuple(
            HarnessRoutePrediction(
                route_id=route.route_id,
                success_probability=probabilities[route.model],
                expected_input_tokens=100,
                expected_cache_read_tokens=0,
                expected_output_tokens=20,
            )
            for route in request.candidate_routes
        )


CANDIDATES = (
    GatewayCandidate(
        gateway_model="small-route",
        model="small",
        provider="local",
        estimated_cost_usd=0.05,
    ),
    GatewayCandidate(
        gateway_model="medium-route",
        model="medium",
        provider="cloud",
        estimated_cost_usd=0.10,
    ),
    GatewayCandidate(
        gateway_model="large-route",
        model="large",
        provider="cloud",
        estimated_cost_usd=0.30,
    ),
)


def test_qwen_scores_models_for_the_already_selected_harness() -> None:
    scorer = FakeQwenScorer()
    router = QwenGatewayRouter(scorer=scorer, candidates=CANDIDATES)

    decision = router.route(
        task_text="Fix the parser.",
        harness="codex",
        request_id="request-1",
        session_id=None,
    )

    assert decision.selected_route.gateway_model == "small-route"
    assert decision.reason == "cheapest_above_quality_threshold"
    assert {route.harness for route in scorer.requests[0].candidate_routes} == {
        "codex"
    }


def test_session_pin_prevents_mid_session_model_switches() -> None:
    scorer = FakeQwenScorer()
    router = QwenGatewayRouter(scorer=scorer, candidates=CANDIDATES)

    first = router.route(
        task_text="First turn.",
        harness="claude-code",
        request_id="request-1",
        session_id="session-1",
    )
    second = router.route(
        task_text="A very different later turn.",
        harness="claude-code",
        request_id="request-2",
        session_id="session-1",
    )

    assert first.selected_route == second.selected_route
    assert second.cache_hit is True
    assert len(scorer.requests) == 1


def test_gateway_alias_can_override_qwen_for_diagnostics() -> None:
    router = QwenGatewayRouter(scorer=FakeQwenScorer(), candidates=CANDIDATES)

    decision = router.route(
        task_text="Force the large route.",
        harness="openai-compatible",
        request_id="request-1",
        session_id=None,
        route_override="large-route",
    )

    assert decision.selected_route.gateway_model == "large-route"
    assert decision.reason == "request_override"


def test_incompatible_models_are_not_shown_to_qwen() -> None:
    scorer = FakeQwenScorer()
    restricted = GatewayCandidate(
        gateway_model="claude-only",
        model="large",
        provider="cloud",
        estimated_cost_usd=0.3,
        compatible_harnesses=("claude-code",),
    )
    router = QwenGatewayRouter(
        scorer=scorer,
        candidates=(CANDIDATES[0], restricted),
    )

    router.route(
        task_text="Use Codex.",
        harness="codex",
        request_id="request-1",
        session_id=None,
    )

    assert [
        route.gateway_model for route in scorer.requests[0].candidate_routes
    ] == ["small-route"]


def test_routes_can_be_configured_from_json() -> None:
    candidates = load_gateway_candidates(
        json.dumps(
            [
                {
                    "gateway_model": "local-route",
                    "model": "llama",
                    "provider": "ollama",
                    "estimated_cost_usd": 0,
                    "compatible_harnesses": ["openai-compatible"],
                }
            ]
        )
    )

    assert candidates[0].model == "llama"
    assert candidates[0].compatible_harnesses == ("openai-compatible",)


def test_invalid_route_configuration_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="non-empty array"):
        load_gateway_candidates("[]")
