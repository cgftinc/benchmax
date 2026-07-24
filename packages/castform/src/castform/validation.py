"""Script-facing environment validation.

Local validation executes the real BenchMax group contract with two sibling
rollouts in this process. Hosted validation (``include_remote=True``) runs the
same shape on Castform's rollout infrastructure: the environment is bundled,
shipped through the platform's ``/v1/rollout/batch/stream`` proxy, executed in
a sandbox by the group-native rollout worker, and the settled results stream
back over SSE.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from benchmax.auth import ModelAuth, bind_model_auth
from benchmax.bundle import dump_bundle
from benchmax.envs import Environment, Example, RolloutOutcome, RolloutRequest

from castform import config
from castform.model_auth import CastformModelAuth
from castform.platform.credentials import platform_bearer

__all__ = [
    "RemoteValidationUnavailable",
    "ValidationReport",
    "validate_environment",
]

_REMOTE_GROUP_SIZE = 2
_REMOTE_CONNECT_TIMEOUT_SECONDS = 15.0


class RemoteValidationUnavailable(RuntimeError):
    """Hosted validation could not run (transport or server-side rejection)."""


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
    auth_bindings: Mapping[str, ModelAuth] | None = None,
    include_remote: bool = False,
    remote_constructor_args: dict[str, Any] | None = None,
    remote_timeout_seconds: float = 900.0,
) -> ValidationReport:
    """Validate ``env`` by running one real local group of two siblings.

    ``model_auth`` authorizes rollout requests. When omitted, the active
    Castform session is resolved at each request. ``auth_bindings`` separately
    supplies the named providers used by ``InjectedAuth`` inside the
    environment. When omitted, Castform binds ``"judge"`` to the same kind of
    call-time session provider. Callers targeting another model provider can
    override either side without silently changing the other.

    ``include_remote`` is additive: local validation always runs first, then
    the environment is re-run remotely on Castform's rollout infrastructure —
    bundled with ``remote_constructor_args`` (pass exactly the kwargs the env
    instance was built with; omit for no-arg environments) and executed as one
    group of two siblings in a sandbox, with model traffic captured at the
    Castform model proxy. Remote outcomes are keyed ``remote-0`` /
    ``remote-1`` by sibling index.
    """

    resolved_base_url = base_url or config.llm_url()
    resolved_model_auth = model_auth or CastformModelAuth()
    resolved_auth_bindings = (
        {"judge": CastformModelAuth()} if auth_bindings is None else dict(auth_bindings)
    )
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
    # Environments refer to their judge credential explicitly as
    # InjectedAuth("judge"). Validation owns the Castform-specific resolution and
    # binds it for the duration of the run, preserving call-time token refresh.
    with bind_model_auth(resolved_auth_bindings):
        local = dict(await env.run_group(requests))
    if set(local) != {"validate-0", "validate-1"}:
        raise ValueError(
            f"local validation returned unexpected rollout IDs: {sorted(local)}"
        )

    remote: dict[str, RolloutOutcome] | None = None
    if include_remote:
        remote = await _remote_validation(
            env=env,
            example=example,
            model=model,
            constructor_args=remote_constructor_args,
            timeout_seconds=remote_timeout_seconds,
        )

    return ValidationReport(local=local, remote=remote)


async def _remote_validation(
    *,
    env: Environment[Any, Any],
    example: Example[Any],
    model: str,
    constructor_args: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, RolloutOutcome]:
    """One group of two siblings through the hosted batch-stream route."""

    try:
        payload_json = json.dumps(example.payload)
    except TypeError as exc:
        raise RemoteValidationUnavailable(
            "Remote validation ships the example inline, so its payload must be "
            f"JSON-serializable: {exc}"
        ) from exc
    del payload_json

    bundle = dump_bundle(type(env), constructor_args=constructor_args or {})
    body = {
        "env": {
            "env_cls_bytes": base64.b64encode(bundle.pickled).decode(),
            "env_metadata_bytes": base64.b64encode(bundle.metadata.to_json_bytes()).decode(),
        },
        "model": {"name": model},
        "inline_examples": [example.payload],
        "split": "eval",
        "group_size": _REMOTE_GROUP_SIZE,
        "max_in_flight": _REMOTE_GROUP_SIZE,
        "group_timeout_seconds": max(60, int(timeout_seconds) - 60),
    }

    events = await _stream_batch(body, timeout_seconds)

    reward_keys = tuple(env.reward_keys)
    outcomes: dict[str, RolloutOutcome] = {}
    for event in events:
        if event.get("event") != "rollout_completed":
            continue
        key = f"remote-{event.get('sample_index', len(outcomes))}"
        if event.get("success"):
            outcomes[key] = RolloutOutcome(
                rewards=dict(event.get("rewards") or {}),
                termination_reason=event.get("termination_reason") or "unknown",
            )
        else:
            # Settled or infrastructure failure: preserve the reported reason
            # under the environment's declared reward shape (all zero), the
            # same convention BenchMax uses for operational failures.
            outcomes[key] = RolloutOutcome(
                rewards={key_: 0.0 for key_ in reward_keys},
                termination_reason=event.get("termination_reason") or "unknown",
            )

    if len(outcomes) != _REMOTE_GROUP_SIZE:
        terminal = [e for e in events if e.get("event") in ("error", "worker_error")]
        raise RemoteValidationUnavailable(
            f"Remote validation returned {len(outcomes)}/{_REMOTE_GROUP_SIZE} "
            f"rollouts; terminal events: {terminal or events[-2:]}"
        )
    return outcomes


async def _stream_batch(body: dict[str, Any], timeout_seconds: float) -> list[dict[str, Any]]:
    url = f"{config.platform_url()}/v1/rollout/batch/stream"
    events: list[dict[str, Any]] = []
    timeout = httpx.Timeout(timeout_seconds, connect=_REMOTE_CONNECT_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={"Authorization": f"Bearer {platform_bearer()}"},
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")[:400]
                raise RemoteValidationUnavailable(
                    f"hosted batch stream rejected the run (HTTP "
                    f"{response.status_code}): {detail}"
                )
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: ") :]))
    return events


def _outcomes_finished(outcomes: dict[str, RolloutOutcome]) -> bool:
    """Return whether validation produced only successful terminal outcomes.

    Rewards are deliberately not part of this check: a correctly executed
    rollout may earn zero. BenchMax records execution failures in the
    termination reason while preserving the environment's reward shape.
    """

    return bool(outcomes) and all(
        outcome.termination_reason == "finished" for outcome in outcomes.values()
    )
