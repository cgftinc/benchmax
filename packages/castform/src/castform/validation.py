"""Script-facing local and hosted environment validation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmax.auth import ModelAuth, bind_model_auth
from benchmax.bundle import Bundle
from benchmax.envs import Environment, RolloutOutcome, RolloutRequest

from castform import config
from castform.model_auth import CastformModelAuth
from castform.platform.client import RolloutClient

__all__ = ["ValidationReport", "validate_environment"]


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
    model: str,
    split: str = "eval",
    base_dir: Path = Path("."),
    model_auth: ModelAuth | None = None,
    auth_bindings: Mapping[str, ModelAuth] | None = None,
    include_remote: bool = False,
    bundle: Bundle | None = None,
    remote_dataset_files: Mapping[str, bytes] | None = None,
    remote_dataset_prefix: str | None = None,
    max_context_tokens: int | None = None,
    max_completion_tokens: int = 1024,
    platform_url: str | None = None,
) -> ValidationReport:
    """Validate the first item of the environment's Dataset with two siblings.

    Dataset construction is always owned by ``env.create_dataset``. Local
    validation calls it against ``base_dir`` and selects item zero. Hosted
    validation sends only opaque artifact files (or an uploaded artifact
    prefix); the group-native worker calls the same method and applies
    ``max_examples=1``. No serialized-example or JSONL compatibility path
    exists.

    ``bundle`` is required only for hosted validation. It is the exact
    BenchMax artifact the caller selected; Castform does not rebuild it.
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
        dataset = await env.create_dataset(split, base_dir)
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
    if include_remote:
        if bundle is None:
            raise ValueError("bundle is required when include_remote=True")
        client = RolloutClient(server_url=platform_url)
        events = await asyncio.to_thread(
            client.run_group,
            samples=2,
            env_cls_bytes=bundle.pickled,
            env_metadata_bytes=bundle.metadata.to_json_bytes(),
            dataset_files=remote_dataset_files,
            dataset_prefix=remote_dataset_prefix,
            split=split,
            model=model,
            max_context_tokens=max_context_tokens,
            max_completion_tokens=max_completion_tokens,
            verbose=False,
        )
        if len(events) != 2:
            raise ValueError(
                f"hosted validation returned {len(events)} rollouts; expected 2"
            )
        zero_rewards = {str(key): 0.0 for key in env.reward_keys}
        remote = {}
        for index, event in enumerate(events):
            rollout_id = str(event.get("rollout_id") or f"remote-{index}")
            remote[rollout_id] = RolloutOutcome(
                rewards=dict(event.get("rewards") or zero_rewards),
                termination_reason=str(event.get("termination_reason") or "unknown"),
            )

    return ValidationReport(local=local, remote=remote)


def _outcomes_finished(outcomes: dict[str, RolloutOutcome]) -> bool:
    return bool(outcomes) and all(
        outcome.termination_reason == "finished" for outcome in outcomes.values()
    )
