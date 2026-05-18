"""Tests for ``BaseEnv.dataset_preprocess``.

The default implementation expects rows already shaped as Example fields
(``seed_messages`` + optional ``task``/``init_rollout_args``) and computes
the canonical ``id``. Envs whose source datasets use other shapes must
override.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Messages, ToolDefinition


class _NoopEnv(BaseEnv):
    async def list_tools(self) -> List[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return None

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {}


def test_dataset_preprocess_does_not_mutate_input():
    example = {
        "seed_messages": [{"role": "user", "content": "hello"}],
        "task": {"ground_truth": "world"},
        "init_rollout_args": {"seed": 1},
    }
    snapshot = {k: v for k, v in example.items()}

    _NoopEnv.dataset_preprocess(example)

    assert example.keys() == snapshot.keys()
    for k in snapshot:
        assert example[k] == snapshot[k]


def test_dataset_preprocess_computes_id():
    example = {
        "seed_messages": [{"role": "user", "content": "q"}],
        "task": {"ground_truth": "a"},
    }
    ex = _NoopEnv.dataset_preprocess(example)
    assert isinstance(ex["id"], str) and len(ex["id"]) == 64
    assert ex["seed_messages"] == example["seed_messages"]
    assert ex["task"] == example["task"]


def test_dataset_preprocess_allows_missing_task():
    example = {"seed_messages": [{"role": "user", "content": "q"}]}
    ex = _NoopEnv.dataset_preprocess(example)
    assert ex["task"] is None


def test_dataset_preprocess_idempotent():
    example = {
        "seed_messages": [{"role": "user", "content": "q"}],
        "task": {"ground_truth": "a"},
    }
    a = _NoopEnv.dataset_preprocess(example)
    b = _NoopEnv.dataset_preprocess(example)
    assert a["id"] == b["id"]


def test_dataset_preprocess_id_groups_equal_examples():
    a = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "task": {"gt": "x"}}
    )
    b = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "task": {"gt": "x"}}
    )
    c = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "task": {"gt": "y"}}
    )
    assert a["id"] == b["id"]
    assert a["id"] != c["id"]


def test_dataset_preprocess_requires_seed_messages():
    with pytest.raises(ValueError, match="seed_messages"):
        _NoopEnv.dataset_preprocess({"task": {"gt": "x"}})
