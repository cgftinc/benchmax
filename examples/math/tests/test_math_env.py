from __future__ import annotations

import json
from pathlib import Path

from benchmax.auth import StaticBearerAuth
from benchmax.envs import RolloutRequest, canonical_example_id
from main import MathEnv
from model_server import LocalModelServer, completion_response


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


async def test_math_env_loads_the_selected_normalized_dataset(tmp_path: Path) -> None:
    train_row = {
        "prompt_messages": [{"role": "user", "content": "What is 6 * 7?"}],
        "answer": "42",
    }
    eval_row = {
        "prompt_messages": [{"role": "user", "content": "What is 8 + 5?"}],
        "answer": "13",
    }
    # Custom filenames still work; they resolve relative to base_dir.
    _write_rows(tmp_path / "datasets/train.jsonl", [train_row])
    _write_rows(tmp_path / "datasets/eval.jsonl", [eval_row])
    env = MathEnv(
        train_dataset_path="datasets/train.jsonl",
        eval_dataset_path="datasets/eval.jsonl",
    )

    train = await env.create_dataset("train", tmp_path)
    evaluation = await env.create_dataset("eval", tmp_path)

    assert train[0].payload == train_row
    assert train[0].id == canonical_example_id(train_row)
    assert evaluation[0].payload == eval_row
    assert evaluation[0].id == canonical_example_id(eval_row)


async def test_math_env_runs_tools_and_discriminates_answers(tmp_path: Path) -> None:
    row = {
        "prompt_messages": [
            {
                "role": "system",
                "content": "Use a tool and put the result in <answer> tags.",
            },
            {"role": "user", "content": "What is 6 * 7 + 3 - 3?"},
        ],
        "answer": "42",
    }
    _write_rows(tmp_path / "train.jsonl", [row])
    _write_rows(tmp_path / "eval.jsonl", [row])
    env = MathEnv(
        long_tool_probability=0.0,
        max_turns=5,
    )
    example = (await env.create_dataset("train", tmp_path))[0]

    def respond(session_id: str, call_index: int, body: dict[str, object]):
        tool_calls = (
            ("multiply", '{"a":6,"b":7}'),
            ("add", '{"a":42,"b":3}'),
            ("subtract", '{"a":45,"b":3}'),
        )
        if call_index < len(tool_calls):
            tool_name, arguments = tool_calls[call_index]
            return 200, completion_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": f"{tool_name}-{session_id}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                    }
                ],
            )
        answer = "42" if session_id == "correct" else "41"
        return 200, completion_response(content=f"<answer>{answer}</answer>")

    with LocalModelServer(respond, concurrent_calls=2) as server:
        outcomes = await env.run_group(
            [
                RolloutRequest(
                    rollout_id=rollout_id,
                    example=example,
                    model="test-model",
                    base_url=server.base_url(rollout_id),
                    model_auth=StaticBearerAuth(f"key-{rollout_id}"),
                )
                for rollout_id in ("correct", "incorrect")
            ]
        )

    assert outcomes["correct"].rewards == {"correctness": 1.0}
    assert outcomes["incorrect"].rewards == {"correctness": 0.0}
    final_calls = [request for request in server.requests if request.call_index == 3]
    assert all(
        request.body["messages"][-1]["content"] == "42.0" for request in final_calls
    )


def _fixture_rows(sentinel_stage: str) -> list[dict[str, object]]:
    return [
        {"task": "6*7", "answer": "42", "__fixture_fail_in": sentinel_stage},
        {"task": "6*7", "answer": "42"},
    ]


def _tool_then_answer(session_id: str, call_index: int, body: dict[str, object]):
    if call_index == 0:
        return 200, completion_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": f"multiply-{session_id}",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
                }
            ],
        )
    return 200, completion_response(content="<answer>42</answer>")


async def _run_sentinel_group(tmp_path: Path, stage: str):
    """Run one 2-sibling group on a sentinel example and one on a healthy one.

    Groups share a single example, so a row sentinel affects every member of
    its group; the healthy example's group must remain untouched.
    """

    _write_rows(tmp_path / "train.jsonl", _fixture_rows(stage))
    _write_rows(tmp_path / "eval.jsonl", _fixture_rows(stage))
    env = MathEnv(long_tool_probability=0.0)
    dataset = await env.create_dataset("train", tmp_path)
    sentinel_example, healthy_example = dataset[0], dataset[1]
    with LocalModelServer(_tool_then_answer, concurrent_calls=2) as server:
        sentinel_outcomes = await env.run_group(
            [
                RolloutRequest(
                    rollout_id=f"sentinel-{index}",
                    example=sentinel_example,
                    model="test-model",
                    base_url=server.base_url(f"sentinel-{index}"),
                    model_auth=StaticBearerAuth("key-sentinel"),
                )
                for index in range(2)
            ]
        )
        healthy_outcomes = await env.run_group(
            [
                RolloutRequest(
                    rollout_id=f"healthy-{index}",
                    example=healthy_example,
                    model="test-model",
                    base_url=server.base_url(f"healthy-{index}"),
                    model_auth=StaticBearerAuth("key-healthy"),
                )
                for index in range(2)
            ]
        )
    return sentinel_outcomes, healthy_outcomes


async def test_task_rows_get_the_system_prompt_and_reference_reward(
    tmp_path: Path,
) -> None:
    """Historical mathenv rows ({task, answer}) remain loadable and scorable."""

    rows = [{"task": "6*7", "answer": "42"}]
    _write_rows(tmp_path / "train.jsonl", rows)
    _write_rows(tmp_path / "eval.jsonl", rows)
    env = MathEnv(long_tool_probability=0.0)
    example = (await env.create_dataset("train", tmp_path))[0]
    assert example.payload["prompt_messages"][0]["role"] == "system"
    assert example.payload["prompt_messages"][1] == {"role": "user", "content": "6*7"}

    with LocalModelServer(_tool_then_answer, concurrent_calls=1) as server:
        outcomes = await env.run_group(
            [
                RolloutRequest(
                    rollout_id="solo",
                    example=example,
                    model="test-model",
                    base_url=server.base_url("solo"),
                    model_auth=StaticBearerAuth("key-solo"),
                )
            ]
        )
    assert outcomes["solo"].rewards == {"correctness": 1.0}


async def test_sentinel_failures_zero_their_group_and_spare_other_groups(
    tmp_path: Path,
) -> None:
    zero = {"correctness": 0.0}
    healthy_expected = {
        "healthy-0": {"correctness": 1.0},
        "healthy-1": {"correctness": 1.0},
    }
    for stage, reason in (
        ("init_rollout", "harness_error"),
        ("release_rollout", "harness_error"),
        ("run_tool", "tool_error"),
        ("compute_reward", "judge_error"),
        ("compute_group_reward", "judge_error"),
    ):
        sentinel_outcomes, healthy_outcomes = await _run_sentinel_group(tmp_path, stage)
        for outcome in sentinel_outcomes.values():
            assert outcome.termination_reason == reason, stage
            assert outcome.rewards == zero, stage
        for rollout_id, rewards in healthy_expected.items():
            assert healthy_outcomes[rollout_id].termination_reason == "finished", stage
            assert healthy_outcomes[rollout_id].rewards == rewards, stage


async def test_preprocessing_sentinel_logs_and_keeps_the_row(
    tmp_path: Path, caplog
) -> None:
    rows = [{"task": "6*7", "answer": "42", "__fixture_fail_in": "preprocessing"}]
    _write_rows(tmp_path / "train.jsonl", rows)
    _write_rows(tmp_path / "eval.jsonl", rows)
    env = MathEnv()
    with caplog.at_level("ERROR"):
        dataset = await env.create_dataset("train", tmp_path)
    assert len(dataset) == 1
    assert "fixture sentinel fired at dataset preprocessing" in caplog.text


async def test_emit_log_sentinel_and_long_tool_padding(tmp_path: Path, caplog) -> None:
    rows = [
        {"task": "6*7", "answer": "42", "__fixture_emit_log": "fixture-warning"},
    ]
    _write_rows(tmp_path / "train.jsonl", rows)
    _write_rows(tmp_path / "eval.jsonl", rows)
    env = MathEnv(
        long_tool_probability=1.0,
        long_tool_chars=64,
    )
    example = (await env.create_dataset("train", tmp_path))[0]

    with LocalModelServer(_tool_then_answer, concurrent_calls=1) as server:
        with caplog.at_level("WARNING"):
            outcomes = await env.run_group(
                [
                    RolloutRequest(
                        rollout_id="solo",
                        example=example,
                        model="test-model",
                        base_url=server.base_url("solo"),
                        model_auth=StaticBearerAuth("key-solo"),
                    )
                ]
            )
    assert "[fixture_emit_log] fixture-warning" in caplog.text
    assert "padding tool output" in caplog.text
    assert outcomes["solo"].termination_reason == "finished"
    padded_tool_turn = [
        request
        for request in server.requests
        if request.call_index == 1 and "x" * 64 in str(request.body["messages"][-1])
    ]
    assert padded_tool_turn
