from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from benchmax.auth import ModelRequestContext
from benchmax.envs.dataset import Dataset
from benchmax.envs.environment import Environment
from benchmax.envs.harbor.dep_check import require_harbor
from benchmax.envs.harbor.bundled_agent import BundledHarborAgent
from benchmax.envs.harbor.credentials import (
    SandboxCredentials,
    sandbox_credentials_scope,
)
from benchmax.envs.harbor.dataset import HarborDataset
from benchmax.envs.harbor.types import HarborTrialTemplate
from benchmax.envs.shared_types import (
    DatasetSplit,
    RolloutAttempt,
    RolloutOutcome,
    RolloutRequest,
)

if TYPE_CHECKING:
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import AgentConfig, TaskConfig, VerifierConfig
    from harbor.models.trial.result import TrialResult

logger = logging.getLogger(__name__)

__all__ = ["HarborEnv"]

_SAFE_TRIAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEFAULT_MODAL_APP_NAME = "harbor-benchmax"
_DEFAULT_MODAL_SANDBOX_TIMEOUT_SECS = 10800
_DEFAULT_MODAL_SANDBOX_IDLE_TIMEOUT_SECS = 1800
_REWARDKIT_MAX_MISSES = 8
_REWARDKIT_NAME_LIMIT = 80
_REWARDKIT_DESCRIPTION_LIMIT = 180
_REWARDKIT_REASONING_LIMIT = 360
_TERMINATION_REASON_BY_EXCEPTION = {
    "AgentTimeoutError": "harness_timeout",
    "ContextLengthExceededError": "context_exceeded",
    "ContextWindowExceededError": "context_exceeded",
    "NonZeroAgentExitCodeError": "harness_error",
    "OutputLengthExceededError": "output_exceeded",
    "SandboxBuildFailedError": "sandbox_error",
    "VerifierTimeoutError": "verifier_timeout",
}


class HarborEnv(Environment["TaskConfig", RolloutAttempt]):
    """Configuration-driven Harbor adapter.

    ``eval_dataset`` is the curated evaluation source and takes precedence.
    ``eval_ratio`` is used only when ``eval_dataset`` is absent.
    """

    @property
    def requires_public_model_endpoint(self) -> bool:
        """Use a public model URL when Harbor runs in a remote sandbox."""

        return self._requires_public_model_endpoint

    @property
    def reward_keys(self) -> Sequence[str]:
        """Return the verifier reward shape declared at construction time."""

        return self._reward_keys

    def __init__(
        self,
        *,
        dataset: DatasetConfig,
        trial: HarborTrialTemplate,
        reward_keys: Sequence[str],
        sandbox_credentials: SandboxCredentials | None = None,
        eval_dataset: DatasetConfig | None = None,
        eval_ratio: float = 0.1,
        max_concurrent_trials: int | None = None,
        requires_public_model_endpoint: bool = True,
    ) -> None:
        require_harbor()
        _validate_configuration(
            dataset=dataset,
            eval_dataset=eval_dataset,
            trial=trial,
            sandbox_credentials=sandbox_credentials,
            eval_ratio=eval_ratio,
            max_concurrent_trials=max_concurrent_trials,
        )
        self._dataset = dataset.model_copy(deep=True)
        self._eval_dataset = (
            eval_dataset.model_copy(deep=True) if eval_dataset is not None else None
        )
        self._eval_ratio = float(eval_ratio)
        self._trial = _with_environment_defaults(trial)
        self._reward_keys = _normalize_reward_keys(reward_keys)
        self._sandbox_credentials = sandbox_credentials
        self._requires_public_model_endpoint = requires_public_model_endpoint
        self._trial_slots = (
            asyncio.Semaphore(max_concurrent_trials)
            if max_concurrent_trials is not None
            else None
        )
        self._dataset_cache: dict[Path, HarborDataset] = {}
        self._dataset_cache_lock = asyncio.Lock()

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
    ) -> Dataset[TaskConfig]:
        """Return the explicit eval source or a ratio-split primary snapshot."""

        if split not in ("train", "eval"):
            raise ValueError(f"unknown dataset split: {split!r}")

        if self._eval_dataset is not None:
            config = self._dataset if split == "train" else self._eval_dataset
            return await self._resolve_dataset(config, Path(base_dir) / split)

        complete = await self._resolve_dataset(
            self._dataset,
            Path(base_dir) / "main",
        )
        train, eval_ = complete.train_eval_split(self._eval_ratio)
        if split == "eval" and not eval_:
            if self._eval_ratio == 0:
                raise ValueError("HarborEnv automatic eval is disabled by eval_ratio=0")
            raise ValueError(
                "HarborEnv automatic eval requires at least two dataset examples"
            )
        return train if split == "train" else eval_

    async def _resolve_dataset(
        self,
        config: DatasetConfig,
        snapshot_dir: Path,
    ) -> HarborDataset:
        """Resolve each configured source once per local snapshot directory."""

        cache_key = snapshot_dir.expanduser().resolve()
        async with self._dataset_cache_lock:
            cached = self._dataset_cache.get(cache_key)
            if cached is None:
                cached = await HarborDataset.create(
                    config,
                    base_dir=cache_key,
                    disable_verification=_verifier_disabled(self._trial.verifier),
                )
                self._dataset_cache[cache_key] = cached
            return cached

    async def run_group(
        self,
        requests: Sequence[RolloutRequest[TaskConfig]],
    ) -> Mapping[str, RolloutOutcome]:
        """Run the shared group algorithm under one provider credential scope."""

        async with sandbox_credentials_scope(self._sandbox_credentials):
            return await super().run_group(requests)

    async def run_rollout(
        self,
        request: RolloutRequest[TaskConfig],
    ) -> RolloutAttempt:
        """Run one trial, waiting for provider capacity when configured."""

        if self._trial_slots is None:
            return await self._execute_rollout(request)
        async with self._trial_slots:
            return await self._execute_rollout(request)

    async def _execute_rollout(
        self,
        request: RolloutRequest[TaskConfig],
    ) -> RolloutAttempt:
        """Create and execute one Harbor trial, including its verifier."""

        async with sandbox_credentials_scope(self._sandbox_credentials):
            from harbor.models.trial.config import TaskConfig, TrialConfig
            from harbor.trial.trial import Trial

            task = request.example.payload
            if not isinstance(task, TaskConfig):
                raise TypeError(
                    "HarborEnv rollout payload must be Harbor TaskConfig, got "
                    f"{type(task).__name__}"
                )
            if not _SAFE_TRIAL_NAME.fullmatch(request.rollout_id):
                raise ValueError(
                    "Harbor rollout_id must contain only letters, numbers, '.', "
                    "'_', and '-', and must start with a letter or number"
                )

            # Harbor calls the in-sandbox harness an "agent"; this block adapts
            # a private copy of the shared harness template to this attempt:
            # inject the per-rollout model endpoint and credentials, and stamp
            # the model route from the request (the constructor rejects preset
            # model_name values, so the request is the single source of truth).
            agent = _prepare_agent_config(self._trial.agent).model_copy(deep=True)
            agent_env = dict(agent.env)
            auth_headers = await request.model_auth.headers_for_request(
                ModelRequestContext(
                    base_url=request.base_url,
                    model=request.model,
                    rollout_id=request.rollout_id,
                )
            )
            authorization = auth_headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                raise ValueError("Harbor model auth must provide a Bearer token")
            agent_env.update(
                {
                    "OPENAI_API_KEY": authorization.removeprefix("Bearer "),
                    "OPENAI_BASE_URL": request.base_url,
                    # Older OpenAI SDKs and some harnesses read this variant.
                    "OPENAI_API_BASE": request.base_url,
                }
            )
            model_name = _openai_model_name(request.model)
            agent = agent.model_copy(
                deep=True,
                update={"model_name": model_name, "env": agent_env},
            )

            trial_config = TrialConfig(
                task=task.model_copy(deep=True),
                trial_name=request.rollout_id,
                trials_dir=self._trial.trials_dir,
                timeout_multiplier=self._trial.timeout_multiplier,
                agent_timeout_multiplier=self._trial.agent_timeout_multiplier,
                verifier_timeout_multiplier=self._trial.verifier_timeout_multiplier,
                agent_setup_timeout_multiplier=(
                    self._trial.agent_setup_timeout_multiplier
                ),
                environment_build_timeout_multiplier=(
                    self._trial.environment_build_timeout_multiplier
                ),
                agent=agent,
                environment=self._trial.environment.model_copy(deep=True),
                verifier=_prepare_verifier_config(self._trial.verifier),
                artifacts=list(self._trial.artifacts),
                extra_instruction_paths=list(self._trial.extra_instruction_paths),
            )

            logger.info(
                "harbor.rollout.start rollout_id=%s task=%s sandbox=%s "
                "harness=%s model=%s",
                request.rollout_id,
                request.example.id,
                _sandbox_name(self._trial),
                agent.name or agent.import_path,
                model_name,
            )
            try:
                trial = await Trial.create(trial_config)
                result = await trial.run()
            except Exception as error:
                logger.exception(
                    "harbor.rollout.failed rollout_id=%s task=%s",
                    request.rollout_id,
                    request.example.id,
                )
                rollout = _zero_reward_rollout(
                    request.rollout_id,
                    reward_keys=self._reward_keys,
                    termination_reason=_exception_termination_reason(
                        type(error).__name__
                    ),
                )
            else:
                rollout = _rollout_attempt(
                    request.rollout_id,
                    result,
                    trial_dir=Path(trial_config.trials_dir) / request.rollout_id,
                    reward_keys=self._reward_keys,
                )
            logger.info(
                "harbor.rollout.done rollout_id=%s termination_reason=%s rewards=%s",
                request.rollout_id,
                rollout.termination_reason,
                dict(rollout.rewards or {}),
            )
            return rollout

    async def aclose(self) -> None:
        """Harbor trials own and close their individual sandbox resources."""


def _rollout_attempt(
    rollout_id: str,
    result: TrialResult,
    *,
    trial_dir: Path,
    reward_keys: Sequence[str],
) -> RolloutAttempt:
    """Normalize every completed Harbor trial into a scored rollout attempt."""

    verifier_result = result.verifier_result
    rewards = verifier_result.rewards if verifier_result is not None else None
    if not rewards:
        if result.exception_info is None:
            detail = "the verifier returned no rewards"
        else:
            detail = (
                f"{result.exception_info.exception_type}: "
                f"{result.exception_info.exception_message}"
            )
        logger.error(
            "harbor.rollout.zero_reward rollout_id=%s error=%s",
            rollout_id,
            detail,
        )
        return _zero_reward_rollout(
            rollout_id,
            reward_keys=reward_keys,
            termination_reason=(
                "verifier_error"
                if result.exception_info is None
                else _result_termination_reason(result)
            ),
        )

    if result.exception_info is not None:
        logger.error(
            "harbor.rollout.zero_reward rollout_id=%s error=%s: %s",
            rollout_id,
            result.exception_info.exception_type,
            result.exception_info.exception_message,
        )
        return _zero_reward_rollout(
            rollout_id,
            reward_keys=reward_keys,
            termination_reason=_result_termination_reason(result),
        )

    normalized_rewards = {str(key): float(value) for key, value in rewards.items()}
    rewardkit_criteria = _rewardkit_criteria(trial_dir)
    if "partial_credit" in reward_keys and "partial_credit" not in normalized_rewards:
        if result.exception_info is None and "reward" in normalized_rewards:
            partial_credit = _rewardkit_partial_credit(rewardkit_criteria)
            normalized_rewards["partial_credit"] = (
                0.0 if partial_credit is None else partial_credit
            )
        else:
            normalized_rewards["partial_credit"] = 0.0
    _log_rewardkit_criteria(rollout_id, rewardkit_criteria)
    return RolloutAttempt(
        rollout_id=rollout_id,
        termination_reason=_result_termination_reason(result),
        rewards=normalized_rewards,
    )


def _zero_reward_rollout(
    rollout_id: str,
    *,
    reward_keys: Sequence[str],
    termination_reason: str,
) -> RolloutAttempt:
    """Keep a failed Harbor rollout in its group without inventing reward signal."""

    return RolloutAttempt(
        rollout_id=rollout_id,
        termination_reason=termination_reason,
        rewards={key: 0.0 for key in reward_keys},
    )


def _result_termination_reason(result: TrialResult) -> str:
    """Use known Harbor terminal states and fall back to existing metadata."""

    if result.exception_info is None:
        return "finished"
    return _exception_termination_reason(result.exception_info.exception_type)


def _exception_termination_reason(exception_type: str) -> str:
    """Classify raised and returned Harbor failures through one vocabulary."""

    return _TERMINATION_REASON_BY_EXCEPTION.get(exception_type, "harness_error")


def _normalize_reward_keys(reward_keys: Sequence[str]) -> tuple[str, ...]:
    """Validate Harbor's explicit verifier reward schema eagerly."""

    if isinstance(reward_keys, (str, bytes)) or not isinstance(reward_keys, Sequence):
        raise TypeError("reward_keys must be a sequence of strings")
    keys = tuple(reward_keys)
    if not keys:
        raise ValueError("reward_keys must declare at least one verifier reward")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("reward_keys must contain non-empty strings")
    if len(set(keys)) != len(keys):
        raise ValueError("reward_keys must not contain duplicates")
    return keys


def _rewardkit_criteria(trial_dir: Path) -> list[Mapping[str, object]] | None:
    """Load criterion details when a Harbor verifier used RewardKit."""

    details_path = trial_dir / "verifier" / "reward-details.json"
    if not details_path.is_file():
        return None
    try:
        payload = json.loads(details_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning(
            "harbor.rewardkit.invalid_details path=%s",
            details_path,
            exc_info=True,
        )
        return None

    reward = payload.get("reward", payload) if isinstance(payload, Mapping) else None
    criteria = reward.get("criteria") if isinstance(reward, Mapping) else None
    if not isinstance(criteria, list) or not criteria:
        return None
    if not all(isinstance(criterion, Mapping) for criterion in criteria):
        return None
    return criteria


def _rewardkit_partial_credit(
    criteria: Sequence[Mapping[str, object]] | None,
) -> float | None:
    """Derive weighted criterion credit from RewardKit details."""

    if not criteria:
        return None
    total_weight = 0.0
    earned_credit = 0.0
    for criterion in criteria:
        weight = criterion.get("weight", 1.0)
        value = criterion.get("value", 0.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return None
        total_weight += float(weight)
        earned_credit += float(weight) * float(value)

    if total_weight <= 0:
        return None
    return earned_credit / total_weight


def _log_rewardkit_criteria(
    rollout_id: str,
    criteria: Sequence[Mapping[str, object]] | None,
) -> None:
    """Log compact scores always and bounded diagnostics only for misses."""

    if not criteria:
        return

    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    for index, criterion in enumerate(criteria, start=1):
        weight = criterion.get("weight", 1.0)
        value = criterion.get("value", 0.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return
        name = _rewardkit_criterion_name(criterion.get("name"), index, names)
        numeric_weight = float(weight)
        numeric_value = float(value)
        normalized.append(
            {
                "name": name,
                "value": numeric_value,
                "weight": numeric_weight,
                "impact": numeric_weight * max(0.0, 1.0 - numeric_value),
                "description": _truncate_log_text(
                    criterion.get("description"), _REWARDKIT_DESCRIPTION_LIMIT
                ),
                "reasoning": _truncate_log_text(
                    criterion.get("reasoning"), _REWARDKIT_REASONING_LIMIT
                ),
            }
        )

    total_weight = sum(float(item["weight"]) for item in normalized)
    weighted_score = (
        sum(float(item["weight"]) * float(item["value"]) for item in normalized)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    values = {str(item["name"]): float(item["value"]) for item in normalized}
    passed = sum(float(item["value"]) >= 1.0 for item in normalized)
    partial = sum(0.0 < float(item["value"]) < 1.0 for item in normalized)
    failed = len(normalized) - passed - partial
    logger.info(
        "harbor.rewardkit.criteria rollout_id=%s total=%d passed=%d partial=%d "
        "failed=%d weighted_score=%.6f values=%s",
        rollout_id,
        len(normalized),
        passed,
        partial,
        failed,
        weighted_score,
        json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )

    misses = sorted(
        (item for item in normalized if float(item["value"]) < 1.0),
        key=lambda item: (-float(item["impact"]), str(item["name"])),
    )
    if not misses:
        return
    shown = misses[:_REWARDKIT_MAX_MISSES]
    logger.info(
        "harbor.rewardkit.misses rollout_id=%s shown=%d omitted=%d details=%s",
        rollout_id,
        len(shown),
        len(misses) - len(shown),
        json.dumps(shown, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )


def _rewardkit_criterion_name(
    raw_name: object,
    index: int,
    existing: set[str],
) -> str:
    name = _truncate_log_text(raw_name, _REWARDKIT_NAME_LIMIT)
    if not name:
        name = f"criterion-{index}"
    unique_name = name
    duplicate = 2
    while unique_name in existing:
        suffix = f"#{duplicate}"
        unique_name = f"{name[: _REWARDKIT_NAME_LIMIT - len(suffix)]}{suffix}"
        duplicate += 1
    existing.add(unique_name)
    return unique_name


def _truncate_log_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _openai_model_name(model: str) -> str:
    """Qualify a bare model id for Harbor's LiteLLM-style routing.

    Built-in Harbor harnesses resolve ``model_name`` through LiteLLM, which
    reads the ``provider/`` prefix to pick an API adapter; a bare HF id such
    as ``Qwen/Qwen3.5-4B`` parses as an unknown provider and fails. Every
    trial talks to the OpenAI-compatible endpoint this env injects as
    ``OPENAI_BASE_URL``, so ``openai/`` is always the correct route. The
    constructor rejects preset ``agent.model_name`` values, so the request
    is the only source of the served model.
    """

    return model if model.startswith("openai/") else f"openai/{model}"


def _sandbox_name(trial: HarborTrialTemplate) -> str:
    """Describe the configured built-in or custom Harbor sandbox."""

    environment = trial.environment
    if environment.import_path:
        return environment.import_path
    if environment.type is None:
        return "unknown"
    return str(getattr(environment.type, "value", environment.type))


def _with_environment_defaults(trial: HarborTrialTemplate) -> HarborTrialTemplate:
    """Apply BenchMax provider defaults without replacing user settings."""

    from harbor.models.environment_type import EnvironmentType

    environment = trial.environment
    if environment.type != EnvironmentType.MODAL:
        return trial

    kwargs = dict(environment.kwargs)
    kwargs.setdefault("app_name", _DEFAULT_MODAL_APP_NAME)
    kwargs.setdefault(
        "sandbox_timeout_secs",
        _DEFAULT_MODAL_SANDBOX_TIMEOUT_SECS,
    )
    kwargs.setdefault(
        "sandbox_idle_timeout_secs",
        _DEFAULT_MODAL_SANDBOX_IDLE_TIMEOUT_SECS,
    )
    return replace(
        trial,
        environment=environment.model_copy(deep=True, update={"kwargs": kwargs}),
    )


def _validate_configuration(
    *,
    dataset: object,
    eval_dataset: object | None,
    trial: object,
    sandbox_credentials: object | None,
    eval_ratio: object,
    max_concurrent_trials: object,
) -> None:
    """Validate the complete constructor boundary before retaining any config."""

    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        VerifierConfig,
    )

    if not isinstance(dataset, DatasetConfig):
        raise TypeError(
            f"dataset must be Harbor DatasetConfig, got {type(dataset).__name__}"
        )
    if eval_dataset is not None and not isinstance(eval_dataset, DatasetConfig):
        raise TypeError(
            "eval_dataset must be Harbor DatasetConfig when provided, got "
            f"{type(eval_dataset).__name__}"
        )
    if not isinstance(trial, HarborTrialTemplate):
        raise TypeError(
            f"trial must be HarborTrialTemplate, got {type(trial).__name__}"
        )
    if not isinstance(trial.agent, (AgentConfig, BundledHarborAgent)):
        raise TypeError("trial.agent must be Harbor AgentConfig or BundledHarborAgent")
    agent_config = (
        trial.agent.config
        if isinstance(trial.agent, BundledHarborAgent)
        else trial.agent
    )
    if getattr(agent_config, "model_name", None):
        raise ValueError(
            "trial.agent.model_name must be unset: every rollout serves exactly "
            "the requested model, which HarborEnv routes as openai/<request "
            "model>. A harness that needs its own alias must implement that "
            "override itself."
        )
    if not isinstance(trial.environment, EnvironmentConfig):
        raise TypeError("trial.environment must be Harbor EnvironmentConfig")
    if not isinstance(trial.verifier, VerifierConfig):
        raise TypeError("trial.verifier must be Harbor VerifierConfig")
    if _verifier_disabled(trial.verifier):
        raise ValueError("HarborEnv requires an enabled verifier to produce rewards")
    if (
        isinstance(eval_ratio, bool)
        or not isinstance(eval_ratio, (int, float))
        or not math.isfinite(eval_ratio)
        or not 0 <= eval_ratio < 1
    ):
        raise ValueError("eval_ratio must satisfy 0 <= eval_ratio < 1")
    if sandbox_credentials is not None and not isinstance(
        sandbox_credentials, SandboxCredentials
    ):
        raise TypeError(
            "sandbox_credentials must implement SandboxCredentials, got "
            f"{type(sandbox_credentials).__name__}"
        )
    if max_concurrent_trials is not None and (
        isinstance(max_concurrent_trials, bool)
        or not isinstance(max_concurrent_trials, int)
        or max_concurrent_trials <= 0
    ):
        raise ValueError("max_concurrent_trials must be a positive integer or None")

    _validate_trial_orchestration(trial)
    _validate_sandbox_credentials(trial, sandbox_credentials)


def _validate_sandbox_credentials(
    trial: HarborTrialTemplate,
    credentials: SandboxCredentials | None,
) -> None:
    """Require explicit cloud credentials and reject provider mismatches."""

    environment = trial.environment
    if environment.import_path is not None or environment.type is None:
        return
    sandbox = str(getattr(environment.type, "value", environment.type))
    if sandbox in {"modal", "daytona"} and credentials is None:
        raise ValueError(
            f"configured Harbor sandbox {sandbox!r} requires explicit "
            "sandbox_credentials"
        )
    if credentials is None or credentials.provider is None:
        return
    if credentials.provider != sandbox:
        raise ValueError(
            f"{credentials.provider!r} sandbox credentials do not match "
            f"configured Harbor sandbox {sandbox!r}"
        )


def _validate_trial_orchestration(trial: HarborTrialTemplate) -> None:
    """Reject Harbor job-queue settings that Benchmax cannot honor."""

    agent = _agent_template_config(trial.agent)
    if agent.n_concurrent is not None or agent.concurrency_group is not None:
        raise ValueError(
            "agent.n_concurrent and agent.concurrency_group belong to Harbor's job "
            "queue; Benchmax owns rollout-group concurrency, so leave them unset"
        )


def _agent_template_config(agent: object) -> AgentConfig:
    """Read the underlying Harbor config without preparing bundled source."""

    if isinstance(agent, BundledHarborAgent):
        return agent.config
    return cast("AgentConfig", agent)


def _prepare_agent_config(agent: object) -> AgentConfig:
    """Resolve bundled source when constructing a concrete Harbor trial."""

    if isinstance(agent, BundledHarborAgent):
        return agent._harbor_config()
    return cast("AgentConfig", agent)


def _verifier_disabled(verifier: object) -> bool:
    return bool(cast("VerifierConfig", verifier).disable)


def _prepare_verifier_config(verifier: object) -> VerifierConfig:
    """Copy the template verifier when constructing a concrete Harbor trial."""

    return cast("VerifierConfig", verifier).model_copy(deep=True)
