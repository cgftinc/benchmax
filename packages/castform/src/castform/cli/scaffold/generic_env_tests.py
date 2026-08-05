"""Unit tests for the dataset and reward. Grow these alongside main.py: cover
malformed rows plus empty, wrong, partial and correct answers.

Plain `asyncio.run` keeps the suite dependency-free; run with `uv run pytest tests`.
"""

import asyncio
import json

import pytest
from benchmax.envs.base import BaseRollout
from main import CustomEnv


def _rollout(answer: str, ground_truth: str) -> BaseRollout:
    return BaseRollout(
        rollout_id="rollout-1",
        termination_reason="stop",
        messages=[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": answer},
        ],
        example_args={"ground_truth": ground_truth},
    )


def _reward(answer: str, ground_truth: str) -> dict[str, float]:
    return asyncio.run(CustomEnv().compute_reward(_rollout(answer, ground_truth)))


def test_empty_answer_is_incorrect() -> None:
    assert _reward("", "Paris") == {"correctness": 0.0}


def test_exact_answer_is_correct() -> None:
    assert _reward("Paris", "Paris") == {"correctness": 1.0}


def test_case_and_outer_whitespace_are_normalized() -> None:
    assert _reward("  PARIS\n", "Paris") == {"correctness": 1.0}


def test_wrong_answer_is_incorrect() -> None:
    assert _reward("Berlin", "Paris") == {"correctness": 0.0}


def test_verbose_answer_containing_gold_is_not_an_exact_answer() -> None:
    assert _reward("The capital of France is Paris.", "Paris") == {"correctness": 0.0}


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"prompt": "", "ground_truth": "Paris"},
        {"prompt": "Capital of France?"},
        {"prompt": "Capital of France?", "ground_truth": ""},
    ],
)
def test_malformed_rows_fail_loudly(row) -> None:
    with pytest.raises((TypeError, ValueError)):
        CustomEnv()._example_from_row(row)


def test_semantic_payload_has_stable_identity() -> None:
    env = CustomEnv()
    row = {"prompt": "Capital of France?", "ground_truth": "Paris"}

    assert env._example_from_row(row).id == env._example_from_row(dict(row)).id
    assert (
        env._example_from_row(row).id != env._example_from_row({**row, "ground_truth": "Lyon"}).id
    )


def test_split_order_and_identity_are_preserved(tmp_path) -> None:
    train_rows = [
        {"prompt": "first", "ground_truth": "one"},
        {"prompt": "second", "ground_truth": "two"},
    ]
    eval_rows = [{"prompt": "third", "ground_truth": "three"}]
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows),
            encoding="utf-8",
        )

    env = CustomEnv()
    train = asyncio.run(env.create_dataset("train", tmp_path))
    eval_ = asyncio.run(env.create_dataset("eval", tmp_path))

    assert [example.payload["ground_truth"] for example in train] == ["one", "two"]
    assert {example.id for example in train}.isdisjoint(example.id for example in eval_)
