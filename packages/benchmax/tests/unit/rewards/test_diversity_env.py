"""Tests for diversity reward wired into an actual environment.

Verifies pickle round-trip and ``compute_group_rewards`` with realistic
rollouts. Uses ngram clustering to avoid LLM calls.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cloudpickle
import pytest
from benchmax.auth import InjectedAuth
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    Example,
    JsonlDataset,
    JsonRow,
    Messages,
    RewardMap,
    Tool,
    canonical_example_id,
)
from benchmax.rewards import Judge, LLMDiversityConfig, NgramDiversityConfig
from benchmax.rewards.diversity import scale_by_diversity

# ---------------------------------------------------------------------------
# Toy env that uses diversity in compute_group_rewards
# ---------------------------------------------------------------------------

_DIVERSITY_CFG = NgramDiversityConfig(n=3, similarity_threshold=0.5)


class _DiversityEnv(BaseEnv):
    reward_keys = ("quality",)
    system_prompt = "You are a helpful assistant."
    max_turns = 1

    async def create_dataset(self, split, base_dir: Path) -> JsonlDataset[JsonRow]:
        def make_example(row: JsonRow) -> Example[JsonRow]:
            payload: JsonRow = {
                "prompt_messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(row.get("prompt", ""))},
                ],
                **{key: value for key, value in row.items() if key != "prompt"},
            }
            return Example(id=canonical_example_id(payload), payload=payload)

        return JsonlDataset(base_dir / f"{split}.jsonl", row_to_example=make_example)

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo back input text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return tool_args.get("text", "")

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> RewardMap:
        return {"quality": 1.0}

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, RewardMap]:
        raw_rewards = [await self.compute_reward(rollout) for rollout in rollouts]
        texts = [
            rollout.messages[-1]["content"] if rollout.messages else ""
            for rollout in rollouts
        ]
        context = rollouts[0].example_args.get("behavior", "") if rollouts else ""
        scaled, _ = await scale_by_diversity(
            raw_rewards, texts, _DIVERSITY_CFG, context=context
        )
        return {
            rollout.rollout_id: reward
            for rollout, reward in zip(rollouts, scaled, strict=True)
        }


def _make_messages(assistant_content: str) -> Messages:
    """Build a realistic message list: user seed + assistant response."""
    return [
        {"role": "user", "content": "What is your approach?"},
        {"role": "assistant", "content": assistant_content},
    ]


def _make_task(behavior: str = "test behavior") -> dict[str, Any]:
    return {"behavior": behavior, "ground_truth": "expected answer"}


def _make_rollouts(
    rollout_ids: Sequence[str],
    messages_list: Sequence[Messages],
    example_args_list: Sequence[Mapping[str, Any]],
) -> list[BaseRollout]:
    return [
        BaseRollout(
            rollout_id=rollout_id,
            termination_reason="finished",
            messages=messages,
            example_args=example_args,
        )
        for rollout_id, messages, example_args in zip(
            rollout_ids,
            messages_list,
            example_args_list,
            strict=True,
        )
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvIntegrated:
    """``compute_group_rewards`` with realistic sibling rollouts."""

    @pytest.mark.asyncio
    async def test_duplicate_strategies_scaled_down(self):
        env = _DiversityEnv()
        rollout_ids = ["r1", "r2", "r3"]
        messages_list = [
            _make_messages("I'll use the academic research framing approach"),
            _make_messages("I'll use the academic research framing approach"),
            _make_messages("Let me try a completely different creative strategy"),
        ]
        tasks = [_make_task(), _make_task(), _make_task()]

        rewards = await env.compute_group_rewards(
            _make_rollouts(rollout_ids, messages_list, tasks)
        )

        assert len(rewards) == 3
        # All are dicts with "quality"
        for reward in rewards.values():
            assert "quality" in reward
        # First two duplicate → halved; third unique → full
        assert rewards["r1"]["quality"] == pytest.approx(0.5)
        assert rewards["r2"]["quality"] == pytest.approx(0.5)
        assert rewards["r3"]["quality"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_all_unique_strategies_full_reward(self):
        env = _DiversityEnv()
        rollout_ids = ["r1", "r2", "r3"]
        messages_list = [
            _make_messages("Alpha strategy using academic framing"),
            _make_messages("Beta strategy via roleplay scenario"),
            _make_messages("Gamma strategy through direct request"),
        ]
        tasks = [_make_task(), _make_task(), _make_task()]

        rewards = await env.compute_group_rewards(
            _make_rollouts(rollout_ids, messages_list, tasks)
        )

        # All unique → all get full reward
        for reward in rewards.values():
            assert reward["quality"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_single_rollout_no_scaling(self):
        env = _DiversityEnv()
        rewards = await env.compute_group_rewards(
            _make_rollouts(
                ["r1"],
                [_make_messages("solo strategy")],
                [_make_task()],
            )
        )
        assert len(rewards) == 1
        assert rewards["r1"]["quality"] == pytest.approx(1.0)


class TestPickleRoundTripEnv:
    """Verify env class survives cloudpickle and can run after restore."""

    def test_env_class_pickles_and_restores(self):
        restored_cls = pickle.loads(cloudpickle.dumps(_DiversityEnv))
        env = restored_cls()
        assert env.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_pickled_env_compute_group_rewards_works(self):
        restored_cls = pickle.loads(cloudpickle.dumps(_DiversityEnv))
        env = restored_cls()
        rollout_ids = ["r1", "r2"]
        messages_list = [
            _make_messages("same approach here"),
            _make_messages("same approach here"),
        ]
        tasks = [_make_task(), _make_task()]

        rewards = await env.compute_group_rewards(
            _make_rollouts(rollout_ids, messages_list, tasks)
        )

        assert len(rewards) == 2
        # Duplicates → halved
        assert rewards["r1"]["quality"] == pytest.approx(0.5)
        assert rewards["r2"]["quality"] == pytest.approx(0.5)

    def test_diversity_config_pickles_with_all_fields(self):
        config = LLMDiversityConfig(
            judge=Judge(
                model="test-model",
                base_url="http://fake/v1",
                auth=InjectedAuth("judge"),
            ),
            max_tokens=768,
        )
        restored = pickle.loads(cloudpickle.dumps(config))
        assert restored.judge.model == "test-model"
        assert restored.max_tokens == 768
