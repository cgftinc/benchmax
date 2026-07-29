"""Script-facing local and hosted environment validation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmax.auth import ModelAuth, StaticBearerAuth, bind_model_auth
from benchmax.envs import DatasetSplit, Environment, RolloutOutcome, RolloutRequest

from castform import config
from castform.model_auth import CastformModelAuth
from castform.platform.client import RolloutClient
from castform.platform.environment_assets import UploadedEnvironmentAssets
from castform.platform.model_session import ModelSession, ModelSessionClient

__all__ = ["ValidationReport", "validate_environment"]

logger = logging.getLogger(__name__)

_VALIDATION_NON_ERROR_TERMINATIONS = Environment.scorable_termination_reasons
_LOCAL_SESSION_TTL_SECONDS = 600


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcomes from mandatory local and optional hosted validation."""

    local: dict[str, RolloutOutcome]
    remote: dict[str, RolloutOutcome] | None = None
    local_errors: dict[str, str] = field(default_factory=dict)
    remote_errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            not self.local_errors
            and _outcomes_executed(self.local)
            and (
                self.remote is None or (not self.remote_errors and _outcomes_executed(self.remote))
            )
        )

    def __bool__(self) -> bool:
        return self.ok


async def validate_environment(
    env: Environment[Any, Any],
    *,
    model: str,
    split: DatasetSplit = "eval",
    base_dir: Path = Path("."),
    model_auth: ModelAuth | None = None,
    auth_bindings: Mapping[str, ModelAuth] | None = None,
    remote_assets: UploadedEnvironmentAssets | None = None,
    max_context_len: int = 2048,
    local_timeout_seconds: float | None = 120,
    platform_url: str | None = None,
) -> ValidationReport:
    """Validate the first item of the environment's Dataset with two siblings.

    Dataset construction is always owned by ``env.create_dataset``. Local
    validation calls it against ``base_dir`` and selects item zero. Hosted
    validation is enabled by passing the same uploaded bundle paths and
    optional ``dataset_path`` used by trainer launch. Both paths ask the
    environment to construct at most one example.
    """

    _validate_limits(
        max_context_tokens=max_context_len,
        local_timeout_seconds=local_timeout_seconds,
    )
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

    # One bounded retry per phase: a transient upstream failure (a flaky
    # model-call 5xx) should not force a full re-run of the whole pipeline.
    # A genuine environment defect still fails on the second attempt.
    for attempt in range(2):
        local, local_errors, rollout_ids = await _run_local_validation(
            env=env,
            model=model,
            split=split,
            base_dir=base_dir,
            model_auth=resolved_model_auth,
            auth_bindings=resolved_auth_bindings,
            proxy_base_url=resolved_base_url,
            max_context_tokens=max_context_len,
            timeout_seconds=local_timeout_seconds,
        )
        if not local_errors or attempt:
            break
        logger.warning(
            "castform.validation.local_retry failures=%s — retrying the local phase once",
            local_errors,
        )

    if set(local) != set(rollout_ids):
        raise ValueError(f"local validation returned unexpected rollout IDs: {sorted(local)}")

    remote: dict[str, RolloutOutcome] | None = None
    remote_errors: dict[str, str] = {}
    if remote_assets is not None:
        client = RolloutClient(server_url=platform_url)
        for attempt in range(2):
            events = await asyncio.to_thread(
                client.run_group,
                samples=2,
                env_cls_path=remote_assets.env_cls_path,
                env_metadata_path=remote_assets.env_metadata_path,
                dataset_path=remote_assets.dataset_path,
                split=split,
                model=model,
                max_context_tokens=max_context_len,
                verbose=False,
            )
            if len(events) != 2:
                raise ValueError(f"hosted validation returned {len(events)} rollouts; expected 2")
            remote = {}
            remote_errors = {}
            for index, event in enumerate(events):
                rollout_id = str(event.get("rollout_id") or f"remote-{index}")
                if event.get("success") is not True:
                    remote_errors[rollout_id] = str(
                        event.get("error") or "rollout produced no usable model trace"
                    )
                remote[rollout_id] = RolloutOutcome(
                    rewards=dict(event.get("rewards") or {}),
                    termination_reason=str(event.get("termination_reason") or "unknown"),
                )
            if not remote_errors or attempt:
                break
            logger.warning(
                "castform.validation.remote_retry failures=%s — retrying the hosted phase once",
                remote_errors,
            )

    return ValidationReport(
        local=local,
        remote=remote,
        local_errors=local_errors,
        remote_errors=remote_errors,
    )


async def _run_local_validation(
    *,
    env: Environment[Any, Any],
    model: str,
    split: DatasetSplit,
    base_dir: Path,
    model_auth: ModelAuth,
    auth_bindings: Mapping[str, ModelAuth],
    proxy_base_url: str,
    max_context_tokens: int,
    timeout_seconds: float | None,
) -> tuple[dict[str, RolloutOutcome], dict[str, str], tuple[str, str]]:
    """Run one local group through ephemeral tracked llm-proxy sessions."""

    rollout_ids = tuple(f"validate-{uuid.uuid4()}" for _ in range(2))
    session_client = ModelSessionClient(
        base_url=proxy_base_url,
        model_auth=model_auth,
    )
    sessions: list[ModelSession] = []
    collected_ids: set[str] = set()
    stage = "dataset construction"
    timeout = asyncio.timeout(timeout_seconds)
    try:
        async with timeout:
            # The auth binding covers dataset creation too: managed
            # environments may need an embedding or tool model while
            # materializing their Dataset.
            with bind_model_auth(auth_bindings):
                dataset = await env.create_dataset(
                    split,
                    base_dir,
                    max_examples=1,
                )
                if not dataset:
                    raise ValueError(f"environment returned an empty {split!r} Dataset")
                example = dataset[0]

                stage = "model-session creation"
                for rollout_id in rollout_ids:
                    sessions.append(
                        await session_client.create(
                            session_id=rollout_id,
                            model=model,
                            max_context_tokens=max_context_tokens,
                            ttl_seconds=_LOCAL_SESSION_TTL_SECONDS,
                        )
                    )

                requests = [
                    RolloutRequest(
                        rollout_id=session.session_id,
                        example=example,
                        model=model,
                        base_url=session.base_url,
                        model_auth=StaticBearerAuth(session.session_key),
                        split=split,
                    )
                    for session in sessions
                ]
                stage = "environment execution"
                local = dict(await env.run_group(requests))

                stage = "model-trace collection"
                local_errors: dict[str, str] = {}
                for session in sessions:
                    try:
                        capture = await session_client.collect(session)
                    except Exception as error:
                        local_errors[session.session_id] = f"model trace collection failed: {error}"
                    else:
                        collected_ids.add(session.session_id)
                        if not capture.get("num_calls"):
                            local_errors[session.session_id] = (
                                "rollout produced no usable model trace"
                            )
                return local, local_errors, rollout_ids
    except TimeoutError as error:
        if not timeout.expired():
            raise
        assert timeout_seconds is not None
        raise TimeoutError(
            f"local validation timed out after {timeout_seconds:g}s during {stage}"
        ) from error
    finally:
        outstanding = [session for session in sessions if session.session_id not in collected_ids]
        if outstanding:
            results = await asyncio.gather(
                *(session_client.discard(session) for session in outstanding),
                return_exceptions=True,
            )
            for session, result in zip(outstanding, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning(
                        "local validation could not discard model session %s",
                        session.session_id,
                        exc_info=(type(result), result, result.__traceback__),
                    )
        await session_client.aclose()


def _validate_limits(
    *,
    max_context_tokens: int,
    local_timeout_seconds: float | None,
) -> None:
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens <= 0
    ):
        raise ValueError("max_context_tokens must be a positive integer")
    if local_timeout_seconds is not None and (
        isinstance(local_timeout_seconds, bool)
        or not isinstance(local_timeout_seconds, (int, float))
        or local_timeout_seconds <= 0
    ):
        raise ValueError("local_timeout_seconds must be positive or None")


def _outcomes_executed(outcomes: dict[str, RolloutOutcome]) -> bool:
    """Accept completed work and intentional budgets; reject execution defects."""

    return bool(outcomes) and all(
        _termination_is_non_error(outcome.termination_reason) for outcome in outcomes.values()
    )


def _termination_is_non_error(reason: str) -> bool:
    return reason.strip().lower() in _VALIDATION_NON_ERROR_TERMINATIONS
