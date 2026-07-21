import json

import pytest

from benchmax.rewards import (
    JudgeError,
    Rubric,
    RubricCache,
    generate_adaptive_rubrics,
    generate_and_cache_adaptive_rubrics,
)


@pytest.mark.asyncio
async def test_generate_adaptive_rubrics_returns_typed_polarities(judge_factory):
    stub = judge_factory(
        [
            json.dumps(
                {
                    "positive_rubrics": [
                        {"title": "Specific", "description": "uses detail"}
                    ],
                    "negative_rubrics": [
                        {"title": "Contradiction", "description": "contradicts itself"}
                    ],
                }
            )
        ]
    )
    result = await generate_adaptive_rubrics(
        question="q",
        ground_truth="truth",
        responses=["one", "two"],
        existing_rubrics=[Rubric("Existing", "already covered")],
        judge=stub.judge,
    )
    assert result.positive[0].polarity == "positive"
    assert result.negative[0].polarity == "negative"
    assert (
        "Existing Rubrics:\nPositive rubrics:\n- Existing: already covered"
        in stub.calls[0]["messages"][0]["content"]
    )


@pytest.mark.asyncio
async def test_generate_adaptive_rubrics_rejects_loose_schema(judge_factory):
    stub = judge_factory(['{"positive_rubrics": {}}'])
    with pytest.raises(JudgeError, match="must be a list"):
        await generate_adaptive_rubrics(
            question="q",
            ground_truth="truth",
            responses=["one", "two"],
            judge=stub.judge,
        )


@pytest.mark.asyncio
async def test_generation_cache_retains_only_discriminative_rubrics(judge_factory):
    responses = [
        json.dumps(
            {
                "positive_rubrics": [
                    {"title": "Specific", "description": "uses detail"},
                    {"title": "Flat", "description": "same everywhere"},
                ],
                "negative_rubrics": [],
            }
        ),
        '{"score": 0}',
        '{"score": 1}',
        '{"score": 1}',
        '{"score": 1}',
    ]
    stub = judge_factory(responses)
    cache = RubricCache()
    result = await generate_and_cache_adaptive_rubrics(
        question="q",
        ground_truth="truth",
        responses=["one", "two"],
        judge=stub.judge,
        cache=cache,
    )
    assert [rubric.title for rubric in result.positive] == ["Specific"]


@pytest.mark.asyncio
async def test_generation_skips_judge_with_fewer_than_two_answers(judge_factory):
    stub = judge_factory([])
    result = await generate_and_cache_adaptive_rubrics(
        question="q",
        ground_truth="truth",
        responses=["", "one"],
        judge=stub.judge,
        cache=RubricCache(),
    )
    assert result.all == ()
    assert stub.calls == []
