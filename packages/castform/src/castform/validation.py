"""Script-facing environment validation.

Local validation executes the real BenchMax group contract with two sibling
rollouts. Hosted validation is intentionally unavailable until rollout-service
supports the same group-native interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmax.auth import ModelAuth, bind_model_auth
from benchmax.envs import Environment, Example, RolloutOutcome, RolloutRequest

from castform import config
from castform.model_auth import CastformModelAuth

__all__ = [
    "RemoteValidationUnavailable",
    "ValidationReport",
    "validate_environment",
]


class RemoteValidationUnavailable(RuntimeError):
    """Hosted validation is waiting on group-native rollout-service support."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcomes from mandatory local and optional hosted validation."""

    local: dict[str, RolloutOutcome]
    remote: dict[str, RolloutOutcome] | None = None

    @property
    def ok(self) -> bool:
        return _outcomes_finished(self.local) and (
            self.remote is None or _outcomes_finished(self.remote)
        )

    def __bool__(self) -> bool:
        return self.ok


async def validate_environment(
    env: Environment[Any, Any],
    *,
    example: Example[Any],
    model: str,
    base_url: str | None = None,
    model_auth: ModelAuth | None = None,
    include_remote: bool = False,
) -> ValidationReport:
    """Validate ``env`` by running one real local group of two siblings.

    Castform authentication is deliberately implicit for this SDK operation:
    when ``model_auth`` is omitted, the active Castform session is resolved at
    each model request. Callers targeting another model provider must pass an
    explicit BenchMax ``ModelAuth`` implementation.

    ``include_remote`` is additive: local validation always runs first. It
    currently raises after the local run because rollout-service has not yet
    migrated to ``Environment.run_group``.
    """

    resolved_base_url = base_url or config.llm_url()
    resolved_auth = model_auth or CastformModelAuth()
    requests = [
        RolloutRequest(
            rollout_id=f"validate-{index}",
            example=example,
            model=model,
            base_url=resolved_base_url,
            model_auth=resolved_auth,
        )
        for index in range(2)
    ]
    # Environments refer to their judge credential explicitly as
    # InjectedAuth("judge"). Validation owns the Castform-specific resolution and
    # binds it for the duration of the run, preserving call-time token refresh.
    with bind_model_auth({"judge": resolved_auth}):
        local = dict(await env.run_group(requests))
    if set(local) != {"validate-0", "validate-1"}:
        raise ValueError(
            f"local validation returned unexpected rollout IDs: {sorted(local)}"
        )

    if include_remote:
        raise RemoteValidationUnavailable(
            "Remote validation is unavailable until rollout-service supports "
            "the group-native BenchMax runtime. Local validation completed."
        )

    return ValidationReport(local=local)


def _outcomes_finished(outcomes: dict[str, RolloutOutcome]) -> bool:
    """Return whether validation produced only successful terminal outcomes.

    Rewards are deliberately not part of this check: a correctly executed
    rollout may earn zero. BenchMax records execution failures in the
    termination reason while preserving the environment's reward shape.
    """

    return bool(outcomes) and all(
        outcome.termination_reason == "finished" for outcome in outcomes.values()
    )
