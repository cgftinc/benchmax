"""Tests for diversity reward wired into an actual env.

Verifies pickle round-trip, compute_group_reward with realistic messages/tasks,
and validate_env compatibility. Uses ngram clustering to avoid LLM calls.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import cloudpickle
import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Messages, ToolDefinition
from benchmax.rewards.diversity import DiversityConfig, scale_by_diversity


# ---------------------------------------------------------------------------
# Toy env that uses diversity in compute_group_reward
# ---------------------------------------------------------------------------

_DIVERSITY_CFG = DiversityConfig(method="ngram", ngram_n=3, similarity_threshold=0.5)


class _DiversityEnv(BaseEnv):
    system_prompt = "You are a helpful assistant."

    async def list_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="echo",
                description="Echo back input text",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return tool_args.get("text", "")

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"quality": 1.0}

    async def compute_group_reward(
        self,
        rollout_ids: List[str],
        messages_list: List[Messages],
        tasks: List[Optional[Dict[str, Any]]],
        **kwargs: Any,
    ) -> List[Dict[str, float]]:
        raw_rewards = []
        for rid, msgs, task in zip(rollout_ids, messages_list, tasks):
            r = await self.compute_reward(rid, msgs, task)
            raw_rewards.append(r)
        texts = [msgs[-1]["content"] if msgs else "" for msgs in messages_list]
        context = (tasks[0] or {}).get("behavior", "") if tasks else ""
        scaled, _ = await scale_by_diversity(
            raw_rewards, texts, _DIVERSITY_CFG, context=context
        )
        return scaled


def _make_messages(assistant_content: str) -> Messages:
    """Build a realistic message list: user seed + assistant response."""
    return [
        {"role": "user", "content": "What is your approach?"},
        {"role": "assistant", "content": assistant_content},
    ]


def _make_task(behavior: str = "test behavior") -> Dict[str, Any]:
    return {"behavior": behavior, "ground_truth": "expected answer"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnvIntegrated:
    """compute_group_reward with realistic messages_list and tasks."""

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

        rewards = await env.compute_group_reward(rollout_ids, messages_list, tasks)

        assert len(rewards) == 3
        # All are dicts with "quality"
        for r in rewards:
            assert "quality" in r
        # First two duplicate → halved; third unique → full
        assert rewards[0]["quality"] == pytest.approx(0.5)
        assert rewards[1]["quality"] == pytest.approx(0.5)
        assert rewards[2]["quality"] == pytest.approx(1.0)

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

        rewards = await env.compute_group_reward(rollout_ids, messages_list, tasks)

        # All unique → all get full reward
        for r in rewards:
            assert r["quality"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_single_rollout_no_scaling(self):
        env = _DiversityEnv()
        rewards = await env.compute_group_reward(
            ["r1"],
            [_make_messages("solo strategy")],
            [_make_task()],
        )
        assert len(rewards) == 1
        assert rewards[0]["quality"] == pytest.approx(1.0)


class TestPickleRoundTripEnv:
    """Verify env class survives cloudpickle and can run after restore."""

    def test_env_class_pickles_and_restores(self):
        restored_cls = pickle.loads(cloudpickle.dumps(_DiversityEnv))
        env = restored_cls()
        assert env.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_pickled_env_compute_group_reward_works(self):
        restored_cls = pickle.loads(cloudpickle.dumps(_DiversityEnv))
        env = restored_cls()
        rollout_ids = ["r1", "r2"]
        messages_list = [
            _make_messages("same approach here"),
            _make_messages("same approach here"),
        ]
        tasks = [_make_task(), _make_task()]

        rewards = await env.compute_group_reward(rollout_ids, messages_list, tasks)

        assert len(rewards) == 2
        # Duplicates → halved
        assert rewards[0]["quality"] == pytest.approx(0.5)
        assert rewards[1]["quality"] == pytest.approx(0.5)

    def test_diversity_config_pickles_with_all_fields(self):
        config = DiversityConfig(
            method="llm",
            model="test-model",
            base_url="http://fake/v1",
            api_key="",  # empty — resolved at call time
            max_tokens=768,
            ngram_n=4,
            similarity_threshold=0.6,
            max_retries=3,
            fallback_on_error="uniform",
        )
        restored = pickle.loads(cloudpickle.dumps(config))
        assert restored.method == "llm"
        assert restored.model == "test-model"
        assert restored.max_retries == 3
        assert restored.fallback_on_error == "uniform"


class TestValidateEnv:
    """Verify the diversity env passes validate_env."""

    def test_validate_env_passes(self):
        import tests.unit.rewards.test_diversity_env as this_module

        from benchmax.platform.validation import validate_env

        dataset = [
            {"prompt": "What is your approach?", "ground_truth": "expected"},
            {"prompt": "Describe your strategy.", "ground_truth": "expected"},
        ]

        result = validate_env(
            env_class=_DiversityEnv,
            env_args={},
            train_dataset=dataset,
            local_modules=[this_module],
        )
        assert result is True
