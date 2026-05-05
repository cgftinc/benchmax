import logging
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from benchmax.envs.tracking import TrackingConfig, log_env, tracking_context
from benchmax.envs.types import Completion, StandardizedExample, ToolDefinition
from benchmax.prompts.tools import render_tools_prompt

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation.

    Auth contract (G2):
        Subclasses that issue HTTP to platform services MUST call
        ``self.auth_headers(url)`` rather than reading ``os.environ`` directly.
        The default ``auth_headers`` is a no-op (safe for standalone/test use).
        Training infrastructure injects a live implementation via
        ``trainer.data.async_workers.orchestrator._wrap_with_castform_auth``, which
        monkeypatches the instance before the env is handed to a Ray actor.
    """

    system_prompt: str = ""
    _tracking_config: TrackingConfig | None = None

    # Env-declared hints for training infra. None = use system defaults.
    recommended_max_turns: Optional[int] = None
    recommended_max_tool_calls: Optional[int] = None

    def __init__(self, **kwargs):
        self._tracking_config: Optional[TrackingConfig] = None

    def __init_subclass__(cls, **kwargs):
        """Warn when subclasses override auth_headers — easy to misunderstand.

        The trainer monkeypatches ``auth_headers`` at the *instance* level,
        which always wins over class-level overrides. A subclass that defines
        its own ``auth_headers`` is dead code on the trainer (instance attr
        beats class attr in Python attribute lookup). Surface that explicitly
        so users don't silently configure auth that never takes effect.
        """
        super().__init_subclass__(**kwargs)
        if "auth_headers" in cls.__dict__ and cls.__dict__["auth_headers"] is not BaseEnv.auth_headers:
            _LOGGER.warning(
                "%s overrides auth_headers; the trainer will replace this at "
                "the instance level, so the override has no effect on training "
                "rollouts. Issue HTTP via self.auth_headers(url) instead of "
                "reading os.environ directly.",
                cls.__name__,
            )

    def enable_tracking(
        self,
        run_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Enable run tracking. Wraps compute_reward on this instance with a tracking context.

        .. warning::
            This installs *instance-level* method wrappers. cloudpickle pickles
            the **class**, not the instance, so an env that has tracking enabled
            and is then bundled via ``bundle_env`` will lose tracking on the
            remote side. Tracking on the trainer is set up separately by
            training infrastructure (which calls ``enable_tracking`` on the
            unpickled instance after construction).

            For local validation use this is fine. For remote training, do not
            rely on tracking surviving the bundle boundary.
        """
        self._tracking_config = TrackingConfig(
            run_id=run_id, api_key=api_key
        )
        cls_compute_reward = type(self).compute_reward

        @wraps(cls_compute_reward)
        async def _tracked(*args, **kwargs):
            with tracking_context(self._tracking_config):
                return await cls_compute_reward(self, *args, **kwargs)

        self.compute_reward = _tracked

        cls_compute_group_reward = type(self).compute_group_reward

        @wraps(cls_compute_group_reward)
        async def _tracked_group(*args, **kwargs):
            with tracking_context(self._tracking_config):
                return await cls_compute_group_reward(self, *args, **kwargs)

        self.compute_group_reward = _tracked_group

    def get_tracking_config(self) -> Optional[TrackingConfig]:
        return self._tracking_config

    def auth_headers(self, url: str) -> dict[str, str]:
        """No-op by default. Override in a host-app mixin/subclass to attach
        bearer tokens to outbound platform-service calls. Custom envs MUST call
        self.auth_headers(url) when issuing HTTP to internal services rather
        than reading os.environ directly.
        """
        return {}

    def log_env(self, rollout_id: str, message: str) -> None:
        log_env(rollout_id, message)

    # Override this method if your example does not match the default structure
    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> StandardizedExample:
        """
        Preprocess a single dataset example into a dict with keys:
        - "prompt": str
        - "ground_truth": Any
        - "init_rollout_args": Dict[str, Any]

        Treats ``example`` as read-only — the original is never mutated, so
        the same example can be passed through multiple envs (e.g. multi-env
        eval) or preprocessed twice without raising KeyError on the second call.
        """
        # Copy + read; the original example dict is left untouched.
        remaining = {k: v for k, v in example.items()
                     if k not in ("prompt", "ground_truth", "init_rollout_args")}
        return StandardizedExample(
            prompt=example.get("prompt", ""),
            ground_truth=example.get("ground_truth", ""),
            init_rollout_args=example["init_rollout_args"],
            **remaining,
        )

    @classmethod
    def load_dataset(
        cls, dataset_name: str, **kwargs
    ) -> Tuple[
        "DatasetDict | Dataset | IterableDatasetDict | IterableDataset", str | None
    ]:
        """
        Download and prepare a dataset for use with this environment.

        This method should handle retrieving the specified dataset (e.g., from HuggingFace, local files,
        or a custom source), preprocessing or converting it into a compatible structure, and storing it
        locally in a reusable format. The processed dataset should be suitable for downstream use with
        `dataset_preprocess`, which standardizes individual examples into the expected format.

        Args:
            dataset_name (str): Identifier of the dataset to be loaded.
            **kwargs: Additional dataset-specific arguments (e.g., split, filtering options, cache directory).

        Returns:
            Dataset: A dataset object (e.g., HuggingFace Dataset or similar) ready for processing.
            str: Optional string pointing to where the dataset is stored locally
        """
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs), None

    # Methods all environment subclasses must implement

    @abstractmethod
    async def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        pass

    @abstractmethod
    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        pass

    @abstractmethod
    async def compute_reward(
        self, rollout_id: str, completion: Completion, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """Compute rewards using registered functions

        Returns dict mapping reward function names to their computed scores.
        """
        pass

    async def compute_group_reward(
        self,
        rollout_ids: List[str],
        completions: List[str | List[Dict[str, str]]],
        ground_truths: List[Any],
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Compute rewards across a group of rollouts jointly.

        Override this when reward computation requires cross-rollout context (e.g.,
        relative scoring, group normalization, or deduplication). Can be used alongside
        ``compute_reward`` — the two are not mutually exclusive. The default implementation
        returns empty reward dicts, deferring entirely to per-rollout ``compute_reward`` calls.

        Args:
            rollout_ids: Identifiers for each rollout in the group.
            completions: Model outputs, one per rollout. Each entry is either a
                plain string or a list of message dicts.
            ground_truths: Reference answers, one per rollout.
            **kwargs: Additional environment-specific arguments.

        Returns:
            A list of reward dicts (one per rollout), each mapping reward function
            names to their computed scores. An empty dict signals that no group
            reward was computed for that rollout.
        """
        return [{} for _ in rollout_ids]

    async def get_system_prompt(self, add_tool_defs: bool = False) -> str:
        """Get system prompt. To add tool definitions, set add_tool_defs to True."""
        if add_tool_defs:
            return render_tools_prompt(
                await self.list_tools(), self.system_prompt or ""
            )
        else:
            return self.system_prompt

    # Optional rollout lifecycle management methods

    async def shutdown(self):
        pass

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize resources for a new rollout"""
        return None

    async def release_rollout(self, rollout_id: str) -> None:
        """Free up resources for a new rollout. Called by compute_reward internally but also available for cleanup."""
        return None

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """Copy a file to the workspace for a specific rollout. If dst_filename is None, use the original filename."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace file copy operations"
        )

    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: str | bytes, dst_filename: str
    ) -> None:
        """Create a file with given content in the workspace for a specific rollout"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace content copy operations"
        )

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """Copy a file from the workspace for a specific rollout"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support workspace file retrieval operations"
        )
