from __future__ import annotations

from typing import Any

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Example, Trajectory


class FullLoopMathEnv(BaseEnv):
    """Tiny Harbor-shaped env: rollout, transcript, tokens, and reward live here."""

    @classmethod
    def dataset_preprocess(cls, row: Any, **kwargs: Any) -> Example:
        lhs = int(row["lhs"])
        rhs = int(row["rhs"])
        prompt = f"What is {lhs} + {rhs}?"
        return {
            "id": f"math-{lhs}-{rhs}",
            "prompt_messages": [{"role": "user", "content": prompt}],
            "task": {"answer": str(lhs + rhs)},
        }

    async def run_rollouts(
        self,
        examples: list[Example],
        *,
        num_generations: int,
        policy=None,
        split: str = "train",
        **kwargs: Any,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for example in examples:
            answer = example["task"]["answer"] if example.get("task") else ""
            for generation_idx in range(num_generations):
                content = answer if generation_idx == 0 else "0"
                messages = [
                    *example["prompt_messages"],
                    {"role": "assistant", "content": content},
                ]
                completion_ids = _fake_token_ids(content)
                trajectories.append(
                    Trajectory(
                        rollout_id=f"{example['id']}-{generation_idx}",
                        example_id=example["id"],
                        prompt_messages=example["prompt_messages"],
                        messages=messages,
                        task=example.get("task"),
                        prompt_ids=_fake_token_ids(example["prompt_messages"][0]["content"]),
                        completion_ids=completion_ids,
                        completion_mask=[1] * len(completion_ids),
                        logprobs=[0.0] * len(completion_ids),
                        rewards={"score": 1.0 if content == answer else 0.0},
                    )
                )
        return trajectories


def _fake_token_ids(text: str) -> list[int]:
    return [ord(char) for char in text]


@pytest.mark.asyncio
async def test_full_loop_math_env_owns_rollout_and_reward() -> None:
    example = FullLoopMathEnv.dataset_preprocess({"lhs": 2, "rhs": 3})

    trajectories = await FullLoopMathEnv().run_rollouts(
        [example],
        num_generations=2,
    )

    assert [trajectory.rollout_id for trajectory in trajectories] == [
        "math-2-3-0",
        "math-2-3-1",
    ]
    assert [trajectory.messages[-1]["content"] for trajectory in trajectories] == [
        "5",
        "0",
    ]
    assert [trajectory.rewards for trajectory in trajectories] == [
        {"score": 1.0},
        {"score": 0.0},
    ]

    sample = trajectories[0].to_sample_dict()
    assert sample["example_id"] == "math-2-3"
    assert sample["completion_ids"] == [ord("5")]
    assert sample["completion_mask"] == [1]
    assert sample["logprobs"] == [0.0]
