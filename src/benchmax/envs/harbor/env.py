from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from benchmax.envs.dataset import Dataset
from benchmax.envs.environment import Environment
from benchmax.envs.harbor._optional import require_harbor
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
    from harbor.models.trial.config import TaskConfig
    from harbor.models.trial.result import TrialResult

logger = logging.getLogger(__name__)

__all__ = ["HarborEnv", "HarborTrialError"]

_SAFE_TRIAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SCORED_AGENT_TERMINATIONS = frozenset(
    {
        "AgentTimeoutError",
        "ContextLengthExceededError",
        "NonZeroAgentExitCodeError",
        "OutputLengthExceededError",
    }
)


class HarborTrialError(RuntimeError):
    """A Harbor trial failed before producing a trustworthy reward."""


class HarborEnv(Environment["TaskConfig", RolloutAttempt]):
    """Configuration-driven Harbor adapter.

    ``eval_dataset`` is the curated evaluation source and takes precedence.
    ``eval_ratio`` is used only when ``eval_dataset`` is absent.
    """

    PIP_DEPENDENCIES = ["harbor>=0.18.0,<0.19"]

    def __init__(
        self,
        *,
        dataset: DatasetConfig,
        trial: HarborTrialTemplate,
        sandbox_credentials: SandboxCredentials | None = None,
        eval_dataset: DatasetConfig | None = None,
        eval_ratio: float = 0.1,
    ) -> None:
        require_harbor()
        _validate_configuration(
            dataset=dataset,
            eval_dataset=eval_dataset,
            trial=trial,
            sandbox_credentials=sandbox_credentials,
            eval_ratio=eval_ratio,
        )
        self._dataset = dataset.model_copy(deep=True)
        self._eval_dataset = (
            eval_dataset.model_copy(deep=True) if eval_dataset is not None else None
        )
        self._eval_ratio = float(eval_ratio)
        self._trial = trial
        self._sandbox_credentials = sandbox_credentials
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
                    disable_verification=self._trial.verifier.disable,
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

            agent = self._trial.agent.model_copy(deep=True)
            agent_env = dict(agent.env)
            agent_env.update(
                {
                    "OPENAI_API_KEY": request.api_key,
                    "OPENAI_BASE_URL": request.base_url,
                    "OPENAI_API_BASE": request.base_url,
                }
            )
            model_name = agent.model_name or _openai_model_name(request.model)
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
                verifier=self._trial.verifier.model_copy(deep=True),
                artifacts=list(self._trial.artifacts),
                extra_instruction_paths=list(self._trial.extra_instruction_paths),
            )

            logger.info(
                "harbor.rollout.start rollout_id=%s task=%s sandbox=%s "
                "agent=%s model=%s",
                request.rollout_id,
                request.example.id,
                _sandbox_name(self._trial),
                agent.name or agent.import_path,
                model_name,
            )
            trial = await Trial.create(trial_config)
            result = await trial.run()
            rollout = _rollout_attempt(request.rollout_id, result)
            logger.info(
                "harbor.rollout.done rollout_id=%s termination_reason=%s rewards=%s",
                request.rollout_id,
                rollout.termination_reason,
                dict(rollout.rewards or {}),
            )
            return rollout

    async def aclose(self) -> None:
        """Harbor trials own and close their individual sandbox resources."""


def _rollout_attempt(rollout_id: str, result: TrialResult) -> RolloutAttempt:
    """Accept scored task terminations, but reject infrastructure failures."""

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
        raise HarborTrialError(
            f"Harbor trial {rollout_id!r} produced no trustworthy reward ({detail})"
        )

    if (
        result.exception_info is not None
        and result.exception_info.exception_type not in _SCORED_AGENT_TERMINATIONS
    ):
        raise HarborTrialError(
            f"Harbor trial {rollout_id!r} failed with infrastructure exception "
            f"{result.exception_info.exception_type}: "
            f"{result.exception_info.exception_message}"
        )

    termination_reason = (
        "finished"
        if result.exception_info is None
        else _exception_termination_reason(result.exception_info.exception_type)
    )
    return RolloutAttempt(
        rollout_id=rollout_id,
        termination_reason=termination_reason,
        rewards={str(key): float(value) for key, value in rewards.items()},
    )


def _exception_termination_reason(exception_type: str) -> str:
    """Turn Harbor exception class names into stable observability labels."""

    without_suffix = exception_type.removesuffix("Error")
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", without_suffix).lower()
    return snake_case or "harbor_exception"


def _openai_model_name(model: str) -> str:
    """Give Harbor agents a provider-qualified default for the TITO endpoint."""

    return model if "/" in model else f"openai/{model}"


def _sandbox_name(trial: HarborTrialTemplate) -> str:
    """Describe the configured built-in or custom Harbor sandbox."""

    environment = trial.environment
    if environment.import_path:
        return environment.import_path
    if environment.type is None:
        return "unknown"
    return str(getattr(environment.type, "value", environment.type))


def _validate_configuration(
    *,
    dataset: object,
    eval_dataset: object | None,
    trial: object,
    sandbox_credentials: object | None,
    eval_ratio: object,
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
    if not isinstance(trial.agent, AgentConfig):
        raise TypeError("trial.agent must be Harbor AgentConfig")
    if not isinstance(trial.environment, EnvironmentConfig):
        raise TypeError("trial.environment must be Harbor EnvironmentConfig")
    if not isinstance(trial.verifier, VerifierConfig):
        raise TypeError("trial.verifier must be Harbor VerifierConfig")
    if trial.verifier.disable:
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

    if (
        trial.agent.n_concurrent is not None
        or trial.agent.concurrency_group is not None
    ):
        raise ValueError(
            "agent.n_concurrent and agent.concurrency_group belong to Harbor's job "
            "queue; Benchmax owns rollout-group concurrency, so leave them unset"
        )
