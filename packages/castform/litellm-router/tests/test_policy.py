from __future__ import annotations

import asyncio

import pytest
from castform_router.policy import HeuristicRoutePolicy
from castform_router.types import RouteRequest

CANDIDATES = ("cheap-model", "frontier-model")


def request(text: str, **metadata: object) -> RouteRequest:
    return RouteRequest(
        request_id="test-request",
        session_id=None,
        task_text=text,
        candidate_models=CANDIDATES,
        metadata=dict(metadata),
    )


def test_simple_task_uses_cheap_model() -> None:
    policy = HeuristicRoutePolicy("cheap-model", "frontier-model")

    decision = asyncio.run(policy.decide(request("Rename this local variable.")))

    assert decision.selected_model == "cheap-model"
    assert decision.reason == "default_cheap"
    assert {prediction.model for prediction in decision.predictions} == set(CANDIDATES)
    cheap = next(
        prediction
        for prediction in decision.predictions
        if prediction.model == "cheap-model"
    )
    assert cheap.success_probability == 0.79


def test_complex_task_uses_frontier_model() -> None:
    policy = HeuristicRoutePolicy("cheap-model", "frontier-model")

    decision = asyncio.run(policy.decide(request("Fix a distributed race condition.")))

    assert decision.selected_model == "frontier-model"
    assert decision.reason == "complexity_marker:distributed"
    frontier = next(
        prediction
        for prediction in decision.predictions
        if prediction.model == "frontier-model"
    )
    assert frontier.success_probability == 0.86


def test_valid_override_wins() -> None:
    policy = HeuristicRoutePolicy("cheap-model", "frontier-model")

    decision = asyncio.run(
        policy.decide(
            request(
                "This task would otherwise be cheap.",
                castform_route_override="frontier-model",
            )
        )
    )

    assert decision.selected_model == "frontier-model"
    assert decision.reason == "request_override"


def test_unknown_override_is_rejected() -> None:
    policy = HeuristicRoutePolicy("cheap-model", "frontier-model")

    with pytest.raises(ValueError, match="available candidate"):
        asyncio.run(
            policy.decide(
                request("Hello.", castform_route_override="unconfigured-model")
            )
        )
