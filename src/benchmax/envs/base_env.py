from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from benchmax.envs.tracking import TrackingConfig, log_env, tracking_context
from benchmax.envs.types import StandardizedExample, ToolDefinition
from benchmax.prompts.tools import render_tools_prompt

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation"""

    system_prompt: str = ""
    _tracking_config: TrackingConfig | None = None

    def __init__(self, **kwargs):
        self._tracking_config: Optional[TrackingConfig] = None

    def enable_tracking(
        self,
        experiment_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Enable experiment tracking. Wraps compute_reward on this instance with a tracking context."""
        self._tracking_config = TrackingConfig(
            experiment_id=experiment_id, api_key=api_key
        )
        cls_compute_reward = type(self).compute_reward

        @wraps(cls_compute_reward)
        async def _tracked(*args, **kwargs):
            with tracking_context(self._tracking_config):
                return await cls_compute_reward(self, *args, **kwargs)

        self.compute_reward = _tracked

    def get_tracking_config(self) -> Optional[TrackingConfig]:
        return self._tracking_config

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
        """
        prompt = example.pop("prompt", "")
        ground_truth = example.pop("ground_truth", "")
        init_rollout_args = example.pop("init_rollout_args")
        return StandardizedExample(
            prompt=prompt,
            ground_truth=ground_truth,
            init_rollout_args=init_rollout_args,
            **example,
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
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """Compute rewards using registered functions

        Returns dict mapping reward function names to their computed scores.
        """
        pass

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
