import logging
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from benchmax.envs.example_id import canonical_example_id, make_example
from benchmax.envs.types import Example, Messages, PolicyConfig, ToolDefinition, Trajectory
from benchmax.prompts.tools import render_tools_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class BaseEnv(ABC):
    """Base benchmax environment.

    The common trainer-facing contract is :meth:`run_rollouts`: given
    preprocessed examples and a policy endpoint/config, return rewarded
    token trajectories. Older tool-stepped envs can continue to override
    ``list_tools`` / ``run_tool`` / ``compute_reward``; ``ToolEnv`` captures
    that stricter contract explicitly.

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

        System prompt
            Set ``system_prompt`` as a **static class attribute** and read it
            via ``cls.system_prompt`` (the default below does). Preprocessing
            then never constructs an env instance, and the system prompt
            baked into training Examples matches what the playground uses.

            If your prompt is templated (e.g. a corpus description), render it
            at class-definition time and assign the result — don't defer it to
            ``__init__``. See
            :meth:`benchmax.envs.postgres_search.search_env.SearchEnv.render_system_prompt`
            for the reference pattern.
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

    @classmethod
    def playground_preprocess(
        cls,
        prompt: str | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
    ) -> Example:
        """Wrap a playground input into an :class:`Example`.

        Accepts either ``prompt`` (single user string — the typical one-shot
        chat case) or ``messages`` (a full chat list, used when replaying a
        multi-turn eval prompt). Exactly one must be provided.

        Classmethod (like :meth:`dataset_preprocess`), reading the static
        ``cls.system_prompt`` class attribute — so a playground input is
        preprocessed without constructing an env instance, and the system
        prompt matches what training uses. ``cls.system_prompt`` is prepended
        unless the caller already supplied a system message (a replayed eval
        prompt typically does). ``task=None`` — the rollout worker skips
        reward computation for playground examples.
        """
        if messages is None:
            if not prompt:
                raise ValueError(
                    "playground_preprocess requires either 'prompt' or 'messages'"
                )
            messages = [{"role": "user", "content": prompt}]
        has_system = any(m.get("role") == "system" for m in messages)
        return make_example(
            prompt_messages=messages,
            task=None,
            system_prompt=None if has_system else cls.system_prompt,
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

    async def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose stepped tools"
        )

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose stepped tools"
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not expose stepped rewards"
        )

    async def run_rollouts(
        self,
        examples: List[Example],
        *,
        num_generations: int,
        policy: PolicyConfig | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> List[Trajectory | Dict[str, Any]]:
        """Run rollout attempts and return trainer-ready trajectories.

        Full-loop envs such as Harbor should override this method. The method
        name refers to execution attempts; the return value is the trajectory
        artifact consumed by the learner.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement env-owned rollouts"
        )

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


class ToolEnv(BaseEnv):
    """Adapter contract for turn-stepped tool/reward envs.

    ``ToolEnv`` keeps the old env surface explicit, but exposes it through the
    new trainer-facing ``run_rollouts`` entrypoint. The policy supplies the
    model loop; this class owns tool execution, reward computation, and
    conversion to ``Trajectory``.
    """

    async def run_rollouts(
        self,
        examples: List[Example],
        *,
        num_generations: int,
        policy: PolicyConfig | Dict[str, Any] | None = None,
        split: str = "train",
        **kwargs: Any,
    ) -> List[Trajectory]:
        rollout_func = _policy_get(policy, "rollout_func") or _policy_get(
            policy, "runner"
        )
        if rollout_func is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.run_rollouts requires "
                "policy.rollout_func for ToolEnv rollouts"
            )

        prompts = [example["prompt_messages"] for example in examples]
        init_rollout_args = [
            example.get("init_rollout_args") or {} for example in examples
        ]
        completions = rollout_func(
            prompts,
            num_generations,
            self,
            init_rollout_args=init_rollout_args,
        )
        if inspect.isawaitable(completions):
            completions = await completions

        trajectories: List[Trajectory] = []
        for example_index, example in enumerate(examples):
            start = example_index * num_generations
            stop = start + num_generations
            rollout_ids: List[str] = []
            messages_list: List[Messages] = []
            tasks: List[Optional[Dict[str, Any]]] = []
            group_items: List[Dict[str, Any]] = []

            for flat_index in range(start, stop):
                item = _completion_item(completions, flat_index)
                rollout_id = item["rollout_ids"]
                messages = item["completions"]
                rollout_ids.append(rollout_id)
                messages_list.append(messages)
                tasks.append(example.get("task"))
                group_items.append(item)

            group_rewards = await self.compute_group_reward(
                rollout_ids,
                messages_list,
                tasks,
                **init_rollout_args[example_index],
            )

            for item, rollout_id, messages, task, extra_rewards in zip(
                group_items,
                rollout_ids,
                messages_list,
                tasks,
                group_rewards,
                strict=False,
            ):
                rewards = await self.compute_reward(
                    rollout_id,
                    messages,
                    task,
                    **init_rollout_args[example_index],
                )
                rewards.update(extra_rewards)
                trajectories.append(
                    Trajectory(
                        rollout_id=rollout_id,
                        example_id=example["id"],
                        prompt_messages=example["prompt_messages"],
                        messages=messages,
                        task=task,
                        prompt_ids=item.get("prompt_ids", []),
                        prompt_mask=item.get("prompt_mask"),
                        completion_ids=item.get("completion_ids", []),
                        completion_mask=item.get("completion_mask", []),
                        logprobs=item.get("logprobs", []),
                        rewards=rewards,
                        truncated=bool(item.get("truncated", False)),
                        workspace_path=_optional_str(
                            item.get("workspace_paths") or item.get("workspace_path")
                        ),
                        metadata=_trajectory_metadata(item),
                    )
                )

        return trajectories

    @abstractmethod
    async def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        raise NotImplementedError

    @abstractmethod
    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        raise NotImplementedError

    @abstractmethod
    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Score a stepped rollout."""
        raise NotImplementedError


def _policy_get(policy: PolicyConfig | Dict[str, Any] | None, key: str) -> Any:
    if policy is None:
        return None
    if isinstance(policy, dict):
        return policy.get(key)
    return getattr(policy, key, None)


def _completion_item(completions: Dict[str, Any], index: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    for key, values in completions.items():
        if isinstance(values, list) and len(values) > index:
            item[key] = values[index]
    missing = [key for key in ("rollout_ids", "completions") if key not in item]
    if missing:
        raise ValueError(
            f"rollout output missing required field(s): {', '.join(missing)}"
        )
    return item


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _trajectory_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    core = {
        "completion_ids",
        "completions",
        "completion_mask",
        "logprobs",
        "prompt_ids",
        "prompt_mask",
        "rollout_ids",
        "truncated",
        "workspace_path",
        "workspace_paths",
    }
    return {key: value for key, value in item.items() if key not in core}
