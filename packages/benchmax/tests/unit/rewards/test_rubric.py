import json

import pytest
from benchmax.rewards import (
    JudgeError,
    RankingAnchor,
    Rubric,
    RubricEvaluation,
    RubricRanking,
    evaluate_rubric_ranking,
    evaluate_single_rubric,
)


def test_rubric_normalizes_positive_and_negative_ranges():
    score_map = {-1: "bad", 1: "okay", 3: "good"}
    positive = Rubric("quality", "description", score_map=score_map)
    negative = Rubric("flaw", "description", polarity="negative", score_map=score_map)
    assert positive.reward_for(1) == pytest.approx(0.5)
    assert negative.reward_for(1) == pytest.approx(0.5)
    assert positive.reward_for(-1) == 0.0
    assert negative.reward_for(-1) == 1.0
    with pytest.raises(ValueError, match="one of"):
        positive.reward_for(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "", "description": "d"},
        {"title": "t", "description": ""},
        {"title": "t", "description": "d", "score_map": {0: "only"}},
        {"title": "t", "description": "d", "score_map": {0: "", 1: "yes"}},
    ],
)
def test_rubric_rejects_invalid_definitions(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Rubric(**kwargs)


@pytest.mark.asyncio
async def test_evaluate_single_rubric_returns_typed_result(judge_factory):
    stub = judge_factory(['{"score": 1, "reasoning": "correct"}'])
    result = await evaluate_single_rubric(
        Rubric("accuracy", "is correct"),
        question="question",
        response="answer",
        ground_truth="reference",
        judge=stub.judge,
    )
    assert result == RubricEvaluation(1.0, "correct", '{"score": 1, "reasoning": "correct"}')
    prompt = stub.calls[0]["messages"][0]["content"]
    assert "reference" in prompt


@pytest.mark.asyncio
async def test_evaluate_single_rubric_requires_an_allowed_score(judge_factory):
    stub = judge_factory(['{"score": 0.5}'])
    with pytest.raises(JudgeError, match="not one of"):
        await evaluate_single_rubric(
            Rubric("accuracy", "is correct"),
            question="q",
            response="r",
            judge=stub.judge,
        )


@pytest.mark.asyncio
async def test_ranking_preserves_ties_and_empty_response_zero(judge_factory):
    stub = judge_factory([json.dumps({"ranking": [[1, 2], [0]], "reasoning": "tie"})])
    result = await evaluate_rubric_ranking(
        Rubric("quality", "is good"),
        question="q",
        responses=["worst", "", "best"],
        ground_truth="reference",
        judge=stub.judge,
    )
    assert isinstance(result, RubricRanking)
    assert result.ranking == ((1, 2), (0,))
    assert result.scores[1] == 0.0
    assert result.scores[2] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_ranking_without_reference_maps_best_to_one(judge_factory):
    stub = judge_factory(['{"ranking": [[1], [0]], "reasoning": "clear"}'])
    result = await evaluate_rubric_ranking(
        Rubric("quality", "is good"),
        question="q",
        responses=["worse", "better"],
        judge=stub.judge,
    )
    assert result.scores == (0.0, 1.0)


@pytest.mark.asyncio
async def test_ranking_ground_truth_anchors_the_absolute_scale(judge_factory):
    stub = judge_factory(['{"ranking": [[1], [2], [0]]}'])
    result = await evaluate_rubric_ranking(
        Rubric("quality", "is good"),
        question="q",
        responses=["below", "above"],
        ground_truth="reference",
        judge=stub.judge,
    )
    assert result.scores == (0.0, 1.0)


@pytest.mark.asyncio
async def test_ranking_multiple_anchors_interpolates_quality_bands(judge_factory):
    # Judged order: great anchor, stronger response, acceptable anchor, weaker response.
    stub = judge_factory(['{"ranking": [[3], [1], [2], [0]]}'])
    result = await evaluate_rubric_ranking(
        Rubric("quality", "is good"),
        question="q",
        responses=["weak", "strong"],
        anchors=[
            RankingAnchor("acceptable", 0.4, "acceptable"),
            RankingAnchor("great", 0.7, "great"),
        ],
        judge=stub.judge,
    )
    assert result.scores == pytest.approx((0.0, 0.55))


@pytest.mark.asyncio
async def test_single_nonempty_ranking_uses_judge_local_indices(judge_factory):
    stub = judge_factory([])
    result = await evaluate_rubric_ranking(
        Rubric("quality", "is good"),
        question="q",
        responses=["", "answer"],
        judge=stub.judge,
    )
    assert result.scores == (0.0, 1.0)
    assert result.ranking == ((0,),)


@pytest.mark.asyncio
async def test_ranking_rejects_bare_integer_tiers(judge_factory):
    stub = judge_factory(['{"ranking": [1, 0]}'])
    with pytest.raises(JudgeError, match="tiers"):
        await evaluate_rubric_ranking(
            Rubric("quality", "is good"),
            question="q",
            responses=["a", "b"],
            judge=stub.judge,
        )


def test_ranking_anchor_validates_its_own_state():
    with pytest.raises(ValueError, match="response"):
        RankingAnchor("", 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RankingAnchor("anchor", 2)


@pytest.mark.asyncio
async def test_ranking_rejects_ground_truth_and_anchors(judge_factory):
    stub = judge_factory([])
    with pytest.raises(ValueError, match="not both"):
        await evaluate_rubric_ranking(
            Rubric("quality", "is good"),
            question="q",
            responses=["a"],
            ground_truth="truth",
            anchors=[RankingAnchor("anchor", 0.5)],
            judge=stub.judge,
        )


@pytest.mark.asyncio
async def test_ranking_rejects_anchors_in_reverse_score_order(judge_factory):
    stub = judge_factory([])
    with pytest.raises(ValueError, match="worst to best"):
        await evaluate_rubric_ranking(
            Rubric("quality", "is good"),
            question="q",
            responses=["a", "b"],
            anchors=[RankingAnchor("great", 0.8), RankingAnchor("okay", 0.4)],
            judge=stub.judge,
        )
