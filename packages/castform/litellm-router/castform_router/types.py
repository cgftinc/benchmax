"""Stable request and decision shapes at the router boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class RoutePrediction:
    """One candidate score emitted by the replaceable router model."""

    model: str
    success_probability: float
    expected_total_tokens: int
    uncertainty: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteRequest:
    """Pre-solve information available when selecting a model."""

    request_id: str | None
    session_id: str | None
    task_text: str
    candidate_models: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteDecision:
    """One policy decision, suitable for logging and session pinning."""

    selected_model: str
    reason: str
    policy_version: str
    predictions: tuple[RoutePrediction, ...] = ()
    cache_hit: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessRoute:
    """One tested harness/model/provider combination."""

    route_id: str
    harness: str
    model: str
    provider: str
    gateway_model: str
    estimated_cost_usd: float


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessRouteRequest:
    """Pre-solve job context available to the trained router."""

    request_id: str
    session_id: str | None
    task_text: str
    task_domain: str
    user_context: dict[str, Any]
    workspace_context: dict[str, Any]
    candidate_routes: tuple[HarnessRoute, ...]
    route_override: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessRoutePrediction:
    """Model-owned prediction for one complete execution route."""

    route_id: str
    success_probability: float
    expected_input_tokens: int
    expected_cache_read_tokens: int
    expected_output_tokens: int
    uncertainty: float | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def expected_total_tokens(self) -> int:
        """Return a display-only total; pricing uses the individual token classes."""

        return (
            self.expected_input_tokens
            + self.expected_cache_read_tokens
            + self.expected_output_tokens
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessRouteDecision:
    """Policy-owned choice produced from model predictions and live data."""

    selected_route: HarnessRoute
    predictions: tuple[HarnessRoutePrediction, ...]
    reason: str
    router_model_version: str
    policy_version: str
    quality_threshold: float
    cache_hit: bool = False
