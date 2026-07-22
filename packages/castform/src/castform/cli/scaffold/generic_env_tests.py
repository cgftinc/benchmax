"""Unit tests for the reward. Grow these alongside main.py: cover empty,
wrong, partial and correct answers so reward changes stay honest.

Plain `asyncio.run` keeps the suite dependency-free; run with `uv run pytest tests`.
"""

import asyncio

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


def test_empty_answer_scores_zero_everywhere() -> None:
    assert _reward("", "Paris") == {
        "overlap": 0.0,
        "contains_gold": 0.0,
        "answered": 0.0,
    }


def test_exact_answer_scores_full_overlap_and_contains_gold() -> None:
    reward = _reward("Paris", "Paris")

    assert reward["overlap"] == 1.0
    assert reward["contains_gold"] == 0.5
    assert reward["answered"] == 0.1


def test_wrong_answer_keeps_only_the_answered_shaping() -> None:
    reward = _reward("Berlin", "Paris")

    assert reward["contains_gold"] == 0.0
    assert reward["overlap"] < 0.5
    assert reward["answered"] == 0.1


def test_verbose_answer_containing_gold_gets_partial_overlap() -> None:
    reward = _reward("The capital of France is Paris.", "Paris")

    assert reward["contains_gold"] == 0.5
    assert 0.0 < reward["overlap"] < 1.0
