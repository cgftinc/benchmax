"""Tests for ``BaseEnv.dataset_preprocess``.

The default builds ``seed_messages`` from whichever of ``seed_messages`` /
``messages`` / ``prompt`` is present in the row, and dumps the whole row
into ``task`` so ``compute_reward`` can read any column. Envs whose source
datasets use other column names override.
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
    row = {
        "seed_messages": [{"role": "user", "content": "hello"}],
        "ground_truth": "world",
        "init_rollout_args": {"seed": 1},
    }
    snapshot = {k: v for k, v in row.items()}

    _NoopEnv.dataset_preprocess(row)

    assert row.keys() == snapshot.keys()
    for k in snapshot:
        assert row[k] == snapshot[k]


def test_dataset_preprocess_seed_messages_column():
    row = {
        "seed_messages": [{"role": "user", "content": "q"}],
        "ground_truth": "a",
    }
    ex = _NoopEnv.dataset_preprocess(row)
    assert isinstance(ex["id"], str) and len(ex["id"]) == 64
    assert ex["seed_messages"] == row["seed_messages"]
    # task carries the whole row, so compute_reward can read any column.
    assert ex["task"] == row


def test_dataset_preprocess_messages_column_used_as_seed():
    row = {
        "messages": [{"role": "user", "content": "q"}],
        "ground_truth": "a",
    }
    ex = _NoopEnv.dataset_preprocess(row)
    assert ex["seed_messages"] == row["messages"]
    assert ex["task"] == row


def test_dataset_preprocess_prompt_column_wrapped_as_user_message():
    row = {"prompt": "hello", "ground_truth": "world"}
    ex = _NoopEnv.dataset_preprocess(row)
    assert ex["seed_messages"] == [{"role": "user", "content": "hello"}]
    assert ex["task"] == row


def test_dataset_preprocess_priority_seed_over_messages_over_prompt():
    """When multiple seed columns are present, ``seed_messages`` wins, then
    ``messages``, then ``prompt`` — order matters because some upstream
    datasets carry redundant copies of the prompt."""
    row = {
        "seed_messages": [{"role": "user", "content": "from_seed"}],
        "messages": [{"role": "user", "content": "from_messages"}],
        "prompt": "from_prompt",
    }
    ex = _NoopEnv.dataset_preprocess(row)
    assert ex["seed_messages"][0]["content"] == "from_seed"

    ex2 = _NoopEnv.dataset_preprocess(
        {"messages": [{"role": "user", "content": "from_messages"}], "prompt": "from_prompt"}
    )
    assert ex2["seed_messages"][0]["content"] == "from_messages"


def test_dataset_preprocess_pulls_init_rollout_args():
    row = {
        "seed_messages": [{"role": "user", "content": "q"}],
        "init_rollout_args": {"seed": 7},
    }
    ex = _NoopEnv.dataset_preprocess(row)
    assert ex["init_rollout_args"] == {"seed": 7}


def test_dataset_preprocess_idempotent():
    row = {
        "seed_messages": [{"role": "user", "content": "q"}],
        "ground_truth": "a",
    }
    a = _NoopEnv.dataset_preprocess(row)
    b = _NoopEnv.dataset_preprocess(row)
    assert a["id"] == b["id"]


def test_dataset_preprocess_id_groups_equal_examples():
    a = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "gt": "x"}
    )
    b = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "gt": "x"}
    )
    c = _NoopEnv.dataset_preprocess(
        {"seed_messages": [{"role": "user", "content": "q"}], "gt": "y"}
    )
    assert a["id"] == b["id"]
    assert a["id"] != c["id"]


def test_dataset_preprocess_requires_recognized_seed_column():
    with pytest.raises(ValueError, match="seed_messages"):
        _NoopEnv.dataset_preprocess({"ground_truth": "x"})
