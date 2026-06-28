from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from benchmax.envs.example_id import canonical_example_id, make_example
from benchmax.envs.types import Example, Messages, ToolDefinition

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


@dataclass
class RolloutResult:
    messages: list[dict[str, Any]]
    reward: dict[str, float]
    task: dict[str, Any] | None = None
    truncated: bool = False
    tool_calls_total: int = 0
    tool_calls_failed: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """Minimal environment contract for trainer-driven RL rollouts.

    The env owns the harness. The trainer provides an OpenAI-compatible model
    endpoint for a single rollout session, then reads token-faithful traces from
    the gateway after this method returns.
    """

    system_prompt: str = ""
    recommended_max_turns: int | None = None
    recommended_max_tool_calls: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def dataset_preprocess(cls, row: Any, **kwargs: Any) -> Example:
        if "prompt_messages" in row:
            prompt_messages = row["prompt_messages"]
        elif "messages" in row:
            prompt_messages = row["messages"]
        elif "prompt" in row:
            prompt_messages = [{"role": "user", "content": row["prompt"]}]
        else:
            raise ValueError(
                f"{cls.__name__}.dataset_preprocess: row has none of "
                "'prompt_messages', 'messages', 'prompt'. Override dataset_preprocess."
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
        if messages is None:
            if not prompt:
                raise ValueError("playground_preprocess requires either 'prompt' or 'messages'")
            messages = [{"role": "user", "content": prompt}]
        has_system = any(m.get("role") == "system" for m in messages)
        return make_example(
            prompt_messages=messages,
            task=None,
            system_prompt=None if has_system else cls.system_prompt,
        )

    @classmethod
    def load_dataset(
        cls, dataset_name: str, **kwargs: Any
    ) -> tuple["DatasetDict | Dataset | IterableDatasetDict | IterableDataset", str | None]:
        from datasets import load_dataset

        return load_dataset(dataset_name, **kwargs), None

    @abstractmethod
    async def run_rollout(
        self,
        *,
        rollout_id: str,
        model: str,
        base_url: str,
        api_key: str,
        task: dict[str, Any],
        timeout: float,
    ) -> RolloutResult:
        """Run one harness rollout against the supplied OpenAI-compatible endpoint."""

    async def shutdown(self) -> None:
        pass
