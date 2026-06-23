from __future__ import annotations

from typing import Any

import pytest

from benchmax.envs.base_env import ToolEnv
from benchmax.envs.types import Messages, ToolDefinition


class _ToyToolEnv(ToolEnv):
    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        return None

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {"score": 1.0 if task and messages[-1]["content"] == task["answer"] else 0.0}


async def _toy_rollout_func(prompts, num_generations, benchmax_env, init_rollout_args=None):
    return {
        "rollout_ids": ["rid-1", "rid-2"],
        "completions": [
            [*prompts[0], {"role": "assistant", "content": "yes"}],
            [*prompts[0], {"role": "assistant", "content": "no"}],
        ],
        "prompt_ids": [[1, 2], [1, 2]],
        "prompt_mask": [[1, 1], [1, 1]],
        "completion_ids": [[3], [4]],
        "completion_mask": [[1], [1]],
        "logprobs": [[-0.1], [-0.2]],
        "truncated": [False, False],
        "tool_calls_total": [0, 0],
    }


@pytest.mark.asyncio
async def test_tool_env_run_rollouts_returns_rewarded_trajectories() -> None:
    example = {
        "id": "example-1",
        "prompt_messages": [{"role": "user", "content": "answer?"}],
        "task": {"answer": "yes"},
    }

    trajectories = await _ToyToolEnv().run_rollouts(
        [example],
        num_generations=2,
        policy={"rollout_func": _toy_rollout_func},
    )

    assert [t.rollout_id for t in trajectories] == ["rid-1", "rid-2"]
    assert [t.rewards for t in trajectories] == [{"score": 1.0}, {"score": 0.0}]
    assert trajectories[0].to_sample_dict() == {
        "rollout_id": "rid-1",
        "example_id": "example-1",
        "prompt_messages": [{"role": "user", "content": "answer?"}],
        "messages": [
            {"role": "user", "content": "answer?"},
            {"role": "assistant", "content": "yes"},
        ],
        "task": {"answer": "yes"},
        "prompt_ids": [1, 2],
        "prompt_mask": [1, 1],
        "completion_ids": [3],
        "completion_mask": [1],
        "logprobs": [-0.1],
        "rewards": {"score": 1.0},
        "truncated": False,
        "tool_calls_total": 0,
    }
