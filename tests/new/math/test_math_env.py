from __future__ import annotations

import json
from pathlib import Path

from benchmax.envs import RolloutRequest, canonical_example_id
from benchmax.envs.math import MathEnv
from tests.new.fakes.model_server import LocalModelServer, completion_response


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
        train_dataset_path="train.jsonl",
        eval_dataset_path="eval.jsonl",
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
                    api_key=f"key-{rollout_id}",
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
