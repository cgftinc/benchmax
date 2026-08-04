"""Job-level harness/model/provider routing above LiteLLM."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from castform_router.router_protocol import RouteScorer
from castform_router.types import (
    HarnessRoute,
    HarnessRouteDecision,
    HarnessRoutePrediction,
    HarnessRouteRequest,
)

ROUTES = (
    HarnessRoute(
        route_id="claude-code/claude-sonnet@anthropic",
        harness="claude-code",
        model="claude-sonnet",
        provider="anthropic",
        gateway_model="claude-route",
        estimated_cost_usd=0.30,
    ),
    HarnessRoute(
        route_id="claude-code/glm-5.1@zai",
        harness="claude-code",
        model="glm-5.1",
        provider="zai",
        gateway_model="glm-route",
        estimated_cost_usd=0.08,
    ),
    HarnessRoute(
        route_id="codex/openai-codex@openai",
        harness="codex",
        model="openai-codex",
        provider="openai",
        gateway_model="codex-route",
        estimated_cost_usd=0.18,
    ),
)

_COMPLEXITY_MARKERS = frozenset(
    {
        "architecture",
        "concurrency",
        "deadlock",
        "distributed",
        "migration",
        "multi-file",
        "performance",
        "race condition",
        "refactor",
        "security",
    }
)


class HeuristicRouteScorer:
    """Traceable stand-in implementing the learned-model scoring boundary."""

    router_model_version = "heuristic-placeholder-v1"

    @staticmethod
    def score(
        request: HarnessRouteRequest,
    ) -> tuple[HarnessRoutePrediction, ...]:
        normalized = " ".join(request.task_text.lower().split())
        is_complex = any(marker in normalized for marker in _COMPLEXITY_MARKERS)

        if is_complex:
            values = {
                "claude-code/claude-sonnet@anthropic": (
                    0.91,
                    45_000,
                    12_000,
                    11_000,
                    0.06,
                    ("complex_task", "strong_long_horizon_fit"),
                ),
                "claude-code/glm-5.1@zai": (
                    0.80,
                    34_000,
                    8_000,
                    9_000,
                    0.13,
                    ("complex_task", "cost_efficient", "quality_threshold_risk"),
                ),
                "codex/openai-codex@openai": (
                    0.88,
                    39_000,
                    10_000,
                    10_000,
                    0.08,
                    ("complex_task", "repository_agent_fit"),
                ),
            }
        else:
            values = {
                "claude-code/claude-sonnet@anthropic": (
                    0.91,
                    10_000,
                    2_000,
                    3_000,
                    0.05,
                    ("simple_task", "higher_capacity"),
                ),
                "claude-code/glm-5.1@zai": (
                    0.85,
                    6_000,
                    1_000,
                    2_000,
                    0.08,
                    ("simple_task", "cost_efficient"),
                ),
                "codex/openai-codex@openai": (
                    0.87,
                    8_000,
                    1_000,
                    3_000,
                    0.07,
                    ("simple_task", "repository_agent_fit"),
                ),
            }

        predictions: list[HarnessRoutePrediction] = []
        for route in request.candidate_routes:
            (
                probability,
                input_tokens,
                cache_tokens,
                output_tokens,
                uncertainty,
                reason_codes,
            ) = values.get(
                route.route_id,
                (0.5, 30_000, 2_000, 8_000, 0.25, ("unknown_route",)),
            )
            predictions.append(
                HarnessRoutePrediction(
                    route_id=route.route_id,
                    success_probability=probability,
                    expected_input_tokens=input_tokens,
                    expected_cache_read_tokens=cache_tokens,
                    expected_output_tokens=output_tokens,
                    uncertainty=uncertainty,
                    reason_codes=reason_codes,
                )
            )
        return tuple(predictions)


class JobRouter:
    """Score complete routes, then apply live cost and quality policy."""

    policy_version = "cheapest-above-threshold-v1"

    def __init__(
        self,
        *,
        scorer: RouteScorer,
        quality_threshold: float = 0.84,
        ttl_seconds: int = 3600,
    ) -> None:
        self._scorer = scorer
        self.quality_threshold = quality_threshold
        self._ttl_seconds = ttl_seconds
        self._pinned: dict[str, tuple[float, HarnessRouteDecision]] = {}
        self._lock = threading.Lock()

    def route(self, request: HarnessRouteRequest) -> HarnessRouteDecision:
        if request.session_id is None:
            return self._decide(request)

        with self._lock:
            now = time.monotonic()
            pinned = self._pinned.get(request.session_id)
            if pinned is not None and pinned[0] > now:
                return replace(pinned[1], cache_hit=True)
            if pinned is not None:
                self._pinned.pop(request.session_id, None)

            decision = self._decide(request)
            self._pinned[request.session_id] = (
                now + self._ttl_seconds,
                decision,
            )
            return decision

    def _decide(self, request: HarnessRouteRequest) -> HarnessRouteDecision:
        predictions = self._scorer.score(request)
        routes_by_id = {
            route.route_id: route for route in request.candidate_routes
        }

        if request.route_override is not None:
            selected_route = routes_by_id.get(request.route_override)
            if selected_route is None:
                raise ValueError("route_override must name an eligible route")
            reason = "request_override"
        else:
            eligible = [
                prediction
                for prediction in predictions
                if prediction.success_probability >= self.quality_threshold
            ]
            if eligible:
                chosen_prediction = min(
                    eligible,
                    key=lambda prediction: routes_by_id[
                        prediction.route_id
                    ].estimated_cost_usd,
                )
                reason = "cheapest_above_quality_threshold"
            else:
                chosen_prediction = max(
                    predictions,
                    key=lambda prediction: prediction.success_probability,
                )
                reason = "highest_quality_fallback"
            selected_route = routes_by_id[chosen_prediction.route_id]

        return HarnessRouteDecision(
            selected_route=selected_route,
            predictions=predictions,
            reason=reason,
            router_model_version=self._scorer.router_model_version,
            policy_version=self.policy_version,
            quality_threshold=self.quality_threshold,
        )



class HeuristicJobRouter(JobRouter):
    """Backward-compatible local router using the deterministic scorer."""

    def __init__(
        self,
        *,
        quality_threshold: float = 0.84,
        ttl_seconds: int = 3600,
    ) -> None:
        super().__init__(
            scorer=HeuristicRouteScorer(),
            quality_threshold=quality_threshold,
            ttl_seconds=ttl_seconds,
        )
