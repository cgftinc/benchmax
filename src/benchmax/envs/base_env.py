from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple
from pathlib import Path

from benchmax.envs.types import ToolDefinition, StandardizedExample
from benchmax.prompts.tools import render_tools_prompt
from benchmax.envs.tracking import TrackingConfig, log_env, with_tracking

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation"""

    system_prompt: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        compute_reward = cls.__dict__.get("compute_reward")
        if compute_reward is None:
            return
        if getattr(compute_reward, "__benchmax_tracking_wrapped__", False):
            return

        wrapped = with_tracking(lambda self, *a, **kw: self.get_tracking_config())(
            compute_reward
        )
        setattr(wrapped, "__benchmax_tracking_wrapped__", True)
        setattr(cls, "compute_reward", wrapped)

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self._tracking_config = TrackingConfig(
            experiment_id=experiment_id, api_key=api_key
        )

    def get_tracking_config(self) -> TrackingConfig:
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
