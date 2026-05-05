"""Test that BaseEnv.dataset_preprocess does not mutate its input (C3)."""

from __future__ import annotations

from typing import Any, Dict, List

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Completion, ToolDefinition


class _NoopEnv(BaseEnv):
    async def list_tools(self) -> List[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return None

    async def compute_reward(
        self, rollout_id: str, completion: Completion, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        return {}


def test_dataset_preprocess_does_not_mutate_input():
    """C3 regression: original dict must remain intact after preprocess."""
    example = {
        "prompt": "hello",
        "ground_truth": "world",
        "init_rollout_args": {"seed": 1},
        "extra": "kept",
    }
    snapshot = dict(example)

    _NoopEnv.dataset_preprocess(example)

    assert example == snapshot, "dataset_preprocess mutated its input"


def test_dataset_preprocess_can_be_called_twice_on_same_example():
    """C3 regression: idempotency check — calling twice must not raise."""
    example = {
        "prompt": "p",
        "ground_truth": "g",
        "init_rollout_args": {},
    }

    first = _NoopEnv.dataset_preprocess(example)
    second = _NoopEnv.dataset_preprocess(example)

    assert first["prompt"] == second["prompt"] == "p"
    assert first["ground_truth"] == second["ground_truth"] == "g"


def test_dataset_preprocess_preserves_extra_fields():
    example = {
        "prompt": "p",
        "ground_truth": "g",
        "init_rollout_args": {},
        "metadata": {"source": "abc"},
        "label": "x",
    }

    result = _NoopEnv.dataset_preprocess(example)

    assert result["metadata"] == {"source": "abc"}
    assert result["label"] == "x"
