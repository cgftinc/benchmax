from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import cloudpickle
import pytest
from benchmax.auth import StaticBearerAuth
from benchmax.bundle import dump_bundle
from benchmax.envs import BaseRollout, Example, RolloutRequest
from benchmax.envs.base import env as base_env
from main import MathEnv
from openai.types.chat import ChatCompletion


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
    assert dataset[0].id == (await env.create_dataset("train", tmp_path, max_examples=1))[0].id


@pytest.mark.asyncio
async def test_tools_cover_arithmetic_and_pad_successful_results(monkeypatch) -> None:
    env = MathEnv()
    monkeypatch.setattr(env, "_random", _AlwaysPad())
    names = [tool["function"]["name"] for tool in await env.list_tools()]

    assert names == ["add", "subtract", "multiply", "divide"]
    result = await env.run_tool("rollout", "multiply", a=6, b=7)
    assert result == f"42\n{'x' * 1_000}"
    assert await env.run_tool("rollout", "divide", a=1, b=0) == ("error: division by zero")


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_limit", "reason", "reward", "expected_calls", "serialized"),
    [
        (3, "max_turns_exceeded", 0.0, 3, False),
        (None, "finished", 1.0, 4, False),
        (None, "finished", 1.0, 4, True),
    ],
)
async def test_three_dependent_tools_leave_a_final_answer_turn(
    monkeypatch, turn_limit, reason, reward, expected_calls, serialized
) -> None:
    if serialized:
        env_cls, constructor_args = cloudpickle.loads(dump_bundle(MathEnv).pickled)
        env = env_cls(**constructor_args)
    else:
        env = MathEnv()
    if turn_limit is not None:
        env.max_turns = turn_limit
    operations = [("subtract", 36, 12), ("divide", 24, 6), ("add", 4, 9)]
    calls = 0

    @asynccontextmanager
    async def model_client(request):
        yield None

    async def completion(**kwargs):
        nonlocal calls
        if calls < len(operations):
            name, a, b = operations[calls]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{calls}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps({"a": a, "b": b})},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            assert kwargs["messages"][-1]["content"] == "13"
            message = {"role": "assistant", "content": "<answer>13</answer>"}
            finish_reason = "stop"
        calls += 1
        return ChatCompletion.model_validate(
            {
                "id": f"completion-{calls}",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            }
        )

    monkeypatch.setattr(base_env, "_model_client", model_client)
    monkeypatch.setattr(base_env, "_create_chat_completion", completion)
    outcome = await env.run_rollout(
        RolloutRequest(
            rollout_id="three-tools",
            model="test",
            base_url="http://unused.invalid/v1",
            model_auth=StaticBearerAuth("test-key"),
            example=Example(
                id="three-tools",
                payload={
                    "prompt_messages": [{"role": "user", "content": "(36 - 12) ÷ 6 + 9"}],
                    "answer": "13",
                },
            ),
        )
    )
    assert outcome.termination_reason == reason
    assert outcome.rewards == {"correctness": reward}
    assert calls == expected_calls
