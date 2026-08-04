"""Qwen-backed model routing for an already-running coding harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from castform_router.job_router import JobRouter
from castform_router.router_protocol import RouteScorer
from castform_router.types import HarnessRoute, HarnessRouteDecision, HarnessRouteRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class GatewayCandidate:
    """One LiteLLM model alias the in-harness router may select."""

    gateway_model: str
    model: str
    provider: str
    estimated_cost_usd: float
    compatible_harnesses: tuple[str, ...] = (
        "codex",
        "claude-code",
        "openai-compatible",
    )

    def route_for(self, harness: str) -> HarnessRoute:
        return HarnessRoute(
            route_id=f"{harness}/{self.model}@{self.provider}",
            harness=harness,
            model=self.model,
            provider=self.provider,
            gateway_model=self.gateway_model,
            estimated_cost_usd=self.estimated_cost_usd,
        )


DEFAULT_GATEWAY_CANDIDATES = (
    GatewayCandidate(
        gateway_model="glm-route",
        model="glm-5.1",
        provider="zai",
        estimated_cost_usd=0.08,
    ),
    GatewayCandidate(
        gateway_model="codex-route",
        model="openai-codex",
        provider="openai",
        estimated_cost_usd=0.18,
    ),
    GatewayCandidate(
        gateway_model="claude-route",
        model="claude-sonnet",
        provider="anthropic",
        estimated_cost_usd=0.30,
    ),
)


def load_gateway_candidates(raw_json: str | None) -> tuple[GatewayCandidate, ...]:
    """Load optional deployment metadata from ``CASTFORM_AUTO_ROUTES_JSON``."""

    if raw_json is None or not raw_json.strip():
        return DEFAULT_GATEWAY_CANDIDATES
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise ValueError("CASTFORM_AUTO_ROUTES_JSON must be valid JSON") from error
    if not isinstance(value, list) or not value:
        raise ValueError("CASTFORM_AUTO_ROUTES_JSON must be a non-empty array")

    candidates: list[GatewayCandidate] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"route {index} must be an object")
        required = {"gateway_model", "model", "provider", "estimated_cost_usd"}
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"route {index} is missing: " + ", ".join(sorted(missing))
            )
        harnesses = item.get(
            "compatible_harnesses",
            ["codex", "claude-code", "openai-compatible"],
        )
        if not isinstance(harnesses, list) or not harnesses or not all(
            isinstance(harness, str) and harness for harness in harnesses
        ):
            raise ValueError(
                f"route {index} compatible_harnesses must be non-empty strings"
            )
        try:
            cost = float(item["estimated_cost_usd"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"route {index} estimated_cost_usd must be numeric"
            ) from error
        if cost < 0:
            raise ValueError(
                f"route {index} estimated_cost_usd must be non-negative"
            )
        candidates.append(
            GatewayCandidate(
                gateway_model=_required_string(item, "gateway_model", index),
                model=_required_string(item, "model", index),
                provider=_required_string(item, "provider", index),
                estimated_cost_usd=cost,
                compatible_harnesses=tuple(harnesses),
            )
        )

    aliases = [candidate.gateway_model for candidate in candidates]
    if len(set(aliases)) != len(aliases):
        raise ValueError("gateway_model values must be unique")
    return tuple(candidates)


def _required_string(item: dict[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"route {index} {key} must be a non-empty string")
    return value.strip()


class QwenGatewayRouter:
    """Score model deployments with Qwen, then apply the deterministic policy."""

    def __init__(
        self,
        *,
        scorer: RouteScorer,
        candidates: tuple[GatewayCandidate, ...] = DEFAULT_GATEWAY_CANDIDATES,
        quality_threshold: float = 0.84,
        ttl_seconds: int = 3600,
    ) -> None:
        if not candidates:
            raise ValueError("at least one gateway candidate is required")
        self._candidates = candidates
        self._router = JobRouter(
            scorer=scorer,
            quality_threshold=quality_threshold,
            ttl_seconds=ttl_seconds,
        )

    @property
    def candidates(self) -> tuple[GatewayCandidate, ...]:
        return self._candidates

    def route(
        self,
        *,
        task_text: str,
        harness: str,
        request_id: str,
        session_id: str | None,
        user_context: dict[str, Any] | None = None,
        workspace_context: dict[str, Any] | None = None,
        route_override: str | None = None,
    ) -> HarnessRouteDecision:
        routes = tuple(
            candidate.route_for(harness)
            for candidate in self._candidates
            if harness in candidate.compatible_harnesses
        )
        if not routes:
            raise ValueError(f"no configured model route supports harness {harness!r}")

        normalized_override = route_override
        if route_override is not None:
            aliases = {
                route.gateway_model: route.route_id
                for route in routes
            }
            normalized_override = aliases.get(route_override, route_override)

        scoped_session_id = (
            f"{harness}:{session_id}" if session_id is not None else None
        )
        return self._router.route(
            HarnessRouteRequest(
                request_id=request_id,
                session_id=scoped_session_id,
                task_text=task_text,
                task_domain="software_engineering",
                user_context=user_context or {"client": harness},
                workspace_context=workspace_context or {},
                candidate_routes=routes,
                route_override=normalized_override,
            )
        )
