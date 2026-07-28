"""Script-facing local and hosted environment validation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmax.auth import ModelAuth, bind_model_auth
from benchmax.envs import Environment, RolloutOutcome, RolloutRequest

from castform import config
from castform.model_auth import CastformModelAuth
from castform.platform.client import RolloutClient
from castform.platform.training_run import UploadedTrainingRun

__all__ = ["ValidationReport", "validate_environment"]

_VALIDATION_NON_ERROR_TERMINATIONS = Environment.scorable_termination_reasons


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcomes from mandatory local and optional hosted validation."""

    local: dict[str, RolloutOutcome]
    remote: dict[str, RolloutOutcome] | None = None
    remote_errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return _outcomes_executed(self.local) and (
            self.remote is None
            or (not self.remote_errors and _outcomes_executed(self.remote))
        )

    def __bool__(self) -> bool:
        return self.ok


async def validate_environment(
    env: Environment[Any, Any],
    *,
    model: str,
    split: str = "eval",
    base_dir: Path = Path("."),
    model_auth: ModelAuth | None = None,
    auth_bindings: Mapping[str, ModelAuth] | None = None,
    remote_assets: UploadedTrainingRun | None = None,
    max_context_tokens: int = 2048,
    platform_url: str | None = None,
) -> ValidationReport:
    """Validate the first item of the environment's Dataset with two siblings.

    Dataset construction is always owned by ``env.create_dataset``. Local
    validation calls it against ``base_dir`` and selects item zero. Hosted
    validation is enabled by passing the same uploaded bundle paths and
    optional ``dataset_path`` used by trainer launch. Both paths ask the
    environment to construct at most one example.
    """

    resolved_base_url = config.llm_url()
    resolved_model_auth = model_auth or CastformModelAuth()
    if auth_bindings is None:
        runtime_auth = CastformModelAuth()
        resolved_auth_bindings = {
            "judge": runtime_auth,
            "embedding": runtime_auth,
            "tool_llm": runtime_auth,
        }
    else:
        resolved_auth_bindings = dict(auth_bindings)

    # The auth binding covers dataset creation too: managed environments may
    # need an embedding or tool model while materializing their Dataset.
    with bind_model_auth(resolved_auth_bindings):
        dataset = await env.create_dataset(split, base_dir, max_examples=1)
        if not dataset:
            raise ValueError(f"environment returned an empty {split!r} Dataset")
        example = dataset[0]
        requests = [
            RolloutRequest(
                rollout_id=f"validate-{index}",
                example=example,
                model=model,
                base_url=resolved_base_url,
                model_auth=resolved_model_auth,
            )
            for index in range(2)
        ]
        local = dict(await env.run_group(requests))

    if set(local) != {"validate-0", "validate-1"}:
        raise ValueError(
            f"local validation returned unexpected rollout IDs: {sorted(local)}"
        )

    remote: dict[str, RolloutOutcome] | None = None
    if remote_assets is not None:
        client = RolloutClient(server_url=platform_url)
        events = await asyncio.to_thread(
            client.run_group,
            samples=2,
            env_cls_path=remote_assets.env_cls_path,
            env_metadata_path=remote_assets.env_metadata_path,
            dataset_path=remote_assets.dataset_path,
            split=split,
            model=model,
            max_context_tokens=max_context_tokens,
            verbose=False,
        )
        if len(events) != 2:
            raise ValueError(
                f"hosted validation returned {len(events)} rollouts; expected 2"
            )
        zero_rewards = {str(key): 0.0 for key in env.reward_keys}
        remote = {}
        remote_errors: dict[str, str] = {}
        for index, event in enumerate(events):
            rollout_id = str(event.get("rollout_id") or f"remote-{index}")
            if event.get("success") is not True:
                remote_errors[rollout_id] = str(
                    event.get("error") or "rollout produced no usable model trace"
                )
            remote[rollout_id] = RolloutOutcome(
                rewards=dict(event.get("rewards") or zero_rewards),
                termination_reason=str(event.get("termination_reason") or "unknown"),
            )
    else:
        remote_errors = {}

    return ValidationReport(
        local=local,
        remote=remote,
        remote_errors=remote_errors,
    )


def _outcomes_executed(outcomes: dict[str, RolloutOutcome]) -> bool:
    """Accept completed work and intentional budgets; reject execution defects."""

    return bool(outcomes) and all(
        _termination_is_non_error(outcome.termination_reason)
        for outcome in outcomes.values()
    )


def _termination_is_non_error(reason: str) -> bool:
    return reason.strip().lower() in _VALIDATION_NON_ERROR_TERMINATIONS
