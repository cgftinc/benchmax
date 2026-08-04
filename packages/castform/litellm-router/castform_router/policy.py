"""Replaceable routing policy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from castform_router.types import RouteDecision, RoutePrediction, RouteRequest


class RoutePolicy(Protocol):
    """Interface that the trained router will implement."""

    async def decide(self, request: RouteRequest) -> RouteDecision:
        """Choose exactly one model from ``request.candidate_models``."""


@dataclass(frozen=True, slots=True)
class HeuristicRoutePolicy:
    """Transparent placeholder policy for exercising the gateway.

    This is intentionally simple and is not an evaluation baseline. Replace it
    with a policy that emits calibrated per-candidate success and token
    predictions, then applies the configured cost-quality decision rule.
    """

    cheap_model: str
    frontier_model: str
    policy_version: str = "heuristic-v0"

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

    async def decide(self, request: RouteRequest) -> RouteDecision:
        candidates = set(request.candidate_models)
        override = request.metadata.get("castform_route_override")
        normalized = " ".join(request.task_text.lower().split())
        marker = next(
            (item for item in sorted(self._COMPLEXITY_MARKERS) if item in normalized),
            None,
        )
        predictions = self._predictions(request, marker=marker)
        if override is not None:
            if not isinstance(override, str) or override not in candidates:
                raise ValueError(
                    "metadata.castform_route_override must name an available candidate"
                )
            return RouteDecision(
                selected_model=override,
                reason="request_override",
                policy_version=self.policy_version,
                predictions=predictions,
            )

        if marker is not None and self.frontier_model in candidates:
            return RouteDecision(
                selected_model=self.frontier_model,
                reason=f"complexity_marker:{marker}",
                policy_version=self.policy_version,
                predictions=predictions,
            )
        if self.cheap_model in candidates:
            return RouteDecision(
                selected_model=self.cheap_model,
                reason="default_cheap",
                policy_version=self.policy_version,
                predictions=predictions,
            )
        if self.frontier_model in candidates:
            return RouteDecision(
                selected_model=self.frontier_model,
                reason="cheap_unavailable",
                policy_version=self.policy_version,
                predictions=predictions,
            )
        raise ValueError("castform-auto has no available candidate models")

    def _predictions(
        self,
        request: RouteRequest,
        *,
        marker: str | None,
    ) -> tuple[RoutePrediction, ...]:
        """Emit illustrative scores in the future trained-router shape."""

        predictions: list[RoutePrediction] = []
        for model in request.candidate_models:
            if marker is not None and model == self.frontier_model:
                values = (0.86, 62_000, 0.10, ("complex_task", "frontier_fit"))
            elif marker is not None:
                values = (0.48, 44_000, 0.18, ("complex_task", "capacity_risk"))
            elif model == self.cheap_model:
                values = (0.79, 9_000, 0.08, ("simple_task", "cost_efficient"))
            else:
                values = (0.88, 15_000, 0.07, ("simple_task", "higher_capacity"))
            probability, tokens, uncertainty, reason_codes = values
            predictions.append(
                RoutePrediction(
                    model=model,
                    success_probability=probability,
                    expected_total_tokens=tokens,
                    uncertainty=uncertainty,
                    reason_codes=reason_codes,
                )
            )
        return tuple(predictions)
