import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from benchmax.envs.example_id import canonical_example_id, make_example
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.prompts.tools import render_tools_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation.

    Logging:
        Use ``logging.getLogger(__name__)`` from any env method — your
        logs (and tracebacks from ``logger.exception``) show up in the
        trainer's log view, attributed to the rollout that triggered them.

        In ``compute_group_reward``, logs apply to every rollout in the
        group by default. To attribute a log to one specific rollout
        only, wrap it in ``rollout_context(rid)`` from
        :mod:`benchmax.envs.logging`.
    """

    system_prompt: str = ""

    # Env-declared hints for training infra. None = use system defaults.
    recommended_max_turns: Optional[int] = None
    recommended_max_tool_calls: Optional[int] = None

    def __init__(self, **kwargs):
        pass

    @classmethod
    def dataset_preprocess(cls, row: Any, **kwargs) -> Example:
        """Turn a dataset row into an :class:`Example`.

        ``prompt_messages`` is built from the first column that's present:

        - ``prompt_messages``: already a chat list — used as-is.
        - ``messages``: chat list — used as ``prompt_messages``.
        - ``prompt``: single string — wrapped as one user message.

        ``task`` is the entire row as a dict, so any column
        (``answer``, ``ground_truth``, etc.) is available to
        ``compute_reward`` without per-env wiring. ``init_rollout_args``
        is pulled out separately if present.

        Override this for datasets with other column names or to project
        ``task`` down to a subset of fields.
        """
        if "prompt_messages" in row:
            prompt_messages = row["prompt_messages"]
        elif "messages" in row:
            prompt_messages = row["messages"]
        elif "prompt" in row:
            prompt_messages = [{"role": "user", "content": row["prompt"]}]
        else:
            raise ValueError(
                f"{cls.__name__}.dataset_preprocess: row has none of "
                "'prompt_messages', 'messages', 'prompt'. Override "
                "dataset_preprocess to build prompt_messages from your "
                "dataset's columns."
            )
        task = dict(row)
        return Example(
            id=canonical_example_id(prompt_messages, task),
            prompt_messages=prompt_messages,
            task=task,
            init_rollout_args=row.get("init_rollout_args"),
        )

    def playground_preprocess(self, prompt: str, **kwargs: Any) -> Example:
        """Wrap a one-shot playground prompt into an :class:`Example`.

        Instance method (vs classmethod :meth:`dataset_preprocess`) so
        subclasses that interpolate ``self.system_prompt`` in ``__init__``
        see the resolved value. Default prepends ``self.system_prompt`` via
        :func:`make_example` with ``task=None`` — the rollout worker skips
        reward computation for playground examples.
        """
        return make_example(
            prompt_messages=[{"role": "user", "content": prompt}],
            task=None,
            system_prompt=self.system_prompt,
        )

    @classmethod
    def load_dataset(
        cls, dataset_name: str, **kwargs
    ) -> Tuple[
        "DatasetDict | Dataset | IterableDatasetDict | IterableDataset", str | None
    ]:
        """Load + prepare a dataset for this env.

        Default thin-wraps ``datasets.load_dataset``. Override to fetch from
        custom sources or to materialize a local cache; return the dataset
        and an optional local path.
        """
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs), None

    # Methods all environment subclasses must implement.

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
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score a rollout.

        ``messages`` is the full transcript (seed + assistant + tool turns).
        ``task`` carries per-example reward-side data (e.g. ``ground_truth``,
        scoring config); ``None`` for envs that grade without per-row data.
        Returns ``{reward_name: score}``.
        """
        pass

    async def compute_group_reward(
        self,
        rollout_ids: List[str],
        messages_list: List[Messages],
        tasks: List[Optional[Dict[str, Any]]],
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        """Score a rollout group jointly.

        Override when reward needs cross-rollout context (relative scoring,
        group normalization, dedup). Default returns one empty dict per
        rollout, signalling per-rollout :meth:`compute_reward` runs in
        isolation. Returns are paired with ``rollout_ids`` by index.
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
