import json

import pytest

from benchmax.rewards import (
    Rubric,
    rank_group_rubrics,
    rubric_reward_key,
    score_group_rubrics,
    score_rubrics,
)


def test_reward_key_is_stable_and_rejects_empty_slug():
    assert rubric_reward_key(Rubric("Clear & correct", "d")) == "rubric_clear_correct"
    with pytest.raises(ValueError, match="letter or digit"):
        rubric_reward_key(Rubric("!!!", "d"))


@pytest.mark.asyncio
async def test_score_rubrics_normalizes_polarity(judge_factory):
    stub = judge_factory(['{"score": 1}', '{"score": 1}'])
    rewards = await score_rubrics(
        "r1",
        "answer",
        ground_truth="truth",
        rubrics=[
            Rubric("quality", "d"),
            Rubric("hallucination", "d", polarity="negative"),
        ],
        question="q",
        judge=stub.judge,
    )
    assert rewards == {"rubric_quality": 1.0, "rubric_hallucination": 0.0}


@pytest.mark.asyncio
async def test_empty_completion_always_has_full_zero_shape(judge_factory):
    stub = judge_factory([])
    rubrics = [
        Rubric("quality", "d"),
        Rubric("hallucination", "d", polarity="negative"),
    ]
    assert await score_rubrics(
        "r1", "", ground_truth="truth", rubrics=rubrics, question="q", judge=stub.judge
    ) == {"rubric_quality": 0.0, "rubric_hallucination": 0.0}
    assert stub.calls == []


@pytest.mark.asyncio
async def test_score_group_validates_alignment_and_key_collisions(judge_factory):
    stub = judge_factory([])
    with pytest.raises(ValueError, match="same length"):
        await score_group_rubrics(
            ["r1"], ["a", "b"], ground_truth="", question="q", judge=stub.judge
        )
    with pytest.raises(ValueError, match="unique reward keys"):
        await score_group_rubrics(
            ["r1"],
            ["a"],
            ground_truth="",
            question="q",
            judge=stub.judge,
            rubrics=[Rubric("A-B", "d"), Rubric("A B", "d")],
        )
    with pytest.raises(ValueError, match="at least one rubric"):
        await score_group_rubrics(
            ["r1"], ["a"], ground_truth="", question="q", judge=stub.judge
        )


@pytest.mark.asyncio
async def test_score_group_static_shape_includes_empty_sibling(judge_factory):
    stub = judge_factory(['{"score": 1}'])
    result = await score_group_rubrics(
        ["r1", "r2"],
        ["answer", ""],
        ground_truth="truth",
        question="q",
        judge=stub.judge,
        rubrics=[Rubric("quality", "d")],
    )
    assert result == [{"rubric_quality": 1.0}, {"rubric_quality": 0.0}]


@pytest.mark.asyncio
async def test_score_group_adaptive_averages_normalized_rewards(judge_factory):
    stub = judge_factory(
        [
            json.dumps(
                {
                    "positive_rubrics": [{"title": "Quality", "description": "d"}],
                    "negative_rubrics": [],
                }
            ),
            '{"score": 0}',
            '{"score": 1}',
            '{"score": 0}',
            '{"score": 1}',
        ]
    )
    result = await score_group_rubrics(
        ["r1", "r2"],
        ["bad", "good"],
        ground_truth="truth",
        question="q",
        judge=stub.judge,
        use_adaptive=True,
    )
    assert result == [{"rubric_adaptive": 0.0}, {"rubric_adaptive": 1.0}]


@pytest.mark.asyncio
async def test_rank_group_rubrics_returns_reward_shape(judge_factory):
    stub = judge_factory(['{"ranking": [[1], [2], [0]]}'])
    result = await rank_group_rubrics(
        ["r1", "r2"],
        ["bad", "good"],
        ground_truth="reference",
        question="q",
        judge=stub.judge,
        rubrics=[Rubric("quality", "d")],
    )
    assert set(result[0]) == {"rubric_quality"}
    assert result[1]["rubric_quality"] > result[0]["rubric_quality"]
