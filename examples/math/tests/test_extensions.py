from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmax.envs import BaseRollout, Example, RolloutFailure, canonical_example_id
from extensions.math_group_env import MathGroupEnv
from extensions.stress_test_env import (
    FAILURE_KEY,
    FAILURE_MODES,
    StressTestMathEnv,
)
from main import MathEnv


def _write_rows(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            f"{json.dumps({'question': f'{index} + 1', 'answer': str(index + 1)})}\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def _rollout(
    rollout_id: str,
    *,
    failure: str | None = None,
) -> BaseRollout:
    example_args = {"answer": "42"}
    if failure is not None:
        example_args[FAILURE_KEY] = failure
    return BaseRollout(
        rollout_id=rollout_id,
        termination_reason="finished",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call", "type": "function"}],
            },
            {"role": "assistant", "content": "<answer>42</answer>"},
        ],
        example_args=example_args,
    )


def _example(failure: str) -> Example:
    payload = {
        "prompt_messages": [{"role": "user", "content": "6 * 7"}],
        "answer": "42",
        FAILURE_KEY: failure,
    }
    return Example(id=canonical_example_id(payload), payload=payload)


@pytest.mark.asyncio
async def test_group_extension_scores_only_at_group_boundary() -> None:
    env = MathGroupEnv()
    rollouts = [_rollout("a"), _rollout("b")]

    assert await env.compute_reward(rollouts[0]) is None
    assert await env.compute_group_rewards(rollouts) == {
        "a": {"correctness": 1.0},
        "b": {"correctness": 1.0},
    }


@pytest.mark.asyncio
async def test_stress_dataset_keeps_first_example_healthy_then_cycles_failures(
    tmp_path: Path,
) -> None:
    _write_rows(tmp_path / "train.jsonl", len(FAILURE_MODES) + 1)
    env = StressTestMathEnv()

    dataset = await env.create_dataset("train", tmp_path)

    assert FAILURE_KEY not in dataset[0].payload
    assert tuple(example.payload[FAILURE_KEY] for example in tuple(dataset)[1:]) == FAILURE_MODES


@pytest.mark.asyncio
async def test_stress_context_and_reward_failures_are_labeled() -> None:
    env = StressTestMathEnv()

    with pytest.raises(RolloutFailure, match="rollout setup failed") as setup:
        async with env.rollout_context("setup", _example("init_rollout")):
            pass
    assert setup.value.termination_reason == "harness_error"

    with pytest.raises(RolloutFailure, match="rollout cleanup failed") as cleanup:
        async with env.rollout_context("cleanup", _example("release_rollout")):
            pass
    assert cleanup.value.termination_reason == "harness_error"

    with pytest.raises(RuntimeError, match="tool execution failed"):
        async with env.rollout_context("tool", _example("run_tool")):
            await env.run_tool("tool", "multiply", a=6, b=7)

    with pytest.raises(RolloutFailure, match="reward service failed") as reward:
        await env.compute_reward(_rollout("reward", failure="compute_reward"))
    assert reward.value.termination_reason == "judge_error"

    with pytest.raises(RolloutFailure, match="group reward service failed") as group:
        await env.compute_group_rewards([_rollout("group", failure="compute_group_reward")])
    assert group.value.termination_reason == "judge_error"


@pytest.mark.asyncio
async def test_stress_crash_happens_once_then_recovers(monkeypatch) -> None:
    env = StressTestMathEnv()
    example = _example("crash_once")
    request = SimpleNamespace(example=example, rollout_id="rollout")

    async def successful_rollout(self, received):
        assert received is request
        return _rollout(received.rollout_id)

    monkeypatch.setattr(MathEnv, "run_rollout", successful_rollout)

    with pytest.raises(RuntimeError, match="crash once"):
        await env.run_rollout(request)

    recovered = await env.run_rollout(request)
    assert recovered.termination_reason == "finished"
