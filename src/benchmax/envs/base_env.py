from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

from benchmax.envs.types import ToolDefinition, StandardizedExample
from benchmax.prompts.tools import render_tools_prompt


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation"""

    system_prompt: str = ""

    def __init__(
        self,
        train_dataset_path: str | Path,
        eval_dataset_path: Optional[str | Path] = None,
        **kwargs,
    ):
        self.train_dataset = self._load_and_process(train_dataset_path)
        self.eval_dataset = self._load_and_process(eval_dataset_path)

    @staticmethod
    def load_raw(path: str | Path) -> List[Any]:
        """Load examples from a local/cloud JSONL file or a HuggingFace hub dataset.

        - Local or cloud files: path must end with .jsonl or .json, or start with
          a cloud prefix (s3://, gs://, https://, etc.).
        - HuggingFace hub: pass the dataset name, optionally with a split suffix,
          e.g. "squad:train".
        """
        from datasets import Dataset
        from datasets import load_dataset as hf_load_dataset

        path_str = str(path)
        _CLOUD_PREFIXES = ("s3://", "gs://", "gcs://", "https://", "http://")

        if path_str.endswith((".jsonl", ".json")) or path_str.startswith(_CLOUD_PREFIXES):
            dataset = hf_load_dataset("json", data_files=path_str, split="train")
        elif ":" in path_str:
            dataset_name, split = path_str.rsplit(":", 1)
            dataset = hf_load_dataset(dataset_name, split=split)
        else:
            dataset = hf_load_dataset(path_str)

        if not isinstance(dataset, Dataset):
            raise ValueError(
                f"'{path_str}' returned a DatasetDict. Specify a split using the "
                f"format '<dataset>:<split>', e.g. '{path_str}:train'."
            )
        return list(dataset)

    def load_and_process(self, path: Optional[str | Path]) -> List[StandardizedExample]:
        if path is None:
            return []
        return [self.dataset_preprocess(ex) for ex in self.load_raw(path)]

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

    async def compute_group_reward(
        self,
        rollout_ids: List[str],
        completions: List[str],
        ground_truths: List[Any],
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Compute rewards over a group of completions, returning one reward dict per completion.

        Override this to implement group-level scoring (e.g. relative scoring, diversity rewards)
        where a completion's score depends on the other completions in the group.

        Returns a list of dicts (one per completion) mapping reward names to scores.
        These are merged with the results of compute_reward by the caller.
        """
        return [{} for _ in completions]

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
