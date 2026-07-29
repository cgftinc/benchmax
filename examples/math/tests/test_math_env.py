from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmax.envs import BaseRollout
from main import MathEnv


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_dataset_builds_stable_base_env_examples(tmp_path: Path) -> None:
    rows = [
        {"question": "6 × 7", "answer": "42"},
        {"question": "8 + 5", "answer": "13"},
    ]
    _write_rows(tmp_path / "train.jsonl", rows)
    _write_rows(tmp_path / "eval.jsonl", rows)
    env = MathEnv()

    dataset = await env.create_dataset("train", tmp_path, max_examples=1)

    assert len(dataset) == 1
    assert dataset[0].payload["prompt_messages"][-1]["content"] == "6 × 7"
    assert dataset[0].payload["answer"] == "42"
    assert (
        dataset[0].id
        == (await env.create_dataset("train", tmp_path, max_examples=1))[0].id
    )


@pytest.mark.asyncio
async def test_tools_cover_arithmetic_and_pad_successful_results(monkeypatch) -> None:
    env = MathEnv()
    monkeypatch.setattr(env, "_random", _AlwaysPad())
    names = [tool["function"]["name"] for tool in await env.list_tools()]

    assert names == ["add", "subtract", "multiply", "divide"]
    result = await env.run_tool("rollout", "multiply", a=6, b=7)
    assert result == f"42\n{'x' * 1_000}"
    assert await env.run_tool("rollout", "divide", a=1, b=0) == (
        "error: division by zero"
    )


class _AlwaysPad:
    def random(self) -> float:
        return 0


@pytest.mark.asyncio
async def test_reward_requires_a_tool_and_correct_numeric_answer() -> None:
    env = MathEnv()

    async def score(answer: str, *, used_tool: bool) -> float:
        messages = []
        if used_tool:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call", "type": "function"}],
                }
            )
        messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
        rewards = await env.compute_reward(
            BaseRollout(
                rollout_id="rollout",
                termination_reason="finished",
                messages=messages,
                example_args={"answer": "42"},
            )
        )
        return rewards["correctness"]

    assert await score("42.0", used_tool=True) == 1.0
    assert await score("41", used_tool=True) == 0.0
    assert await score("42", used_tool=False) == 0.0
