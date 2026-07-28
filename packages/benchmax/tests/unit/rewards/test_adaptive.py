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


def _rubric(title: str, polarity: str = "positive") -> Rubric:
    return Rubric(title, "description", polarity=polarity)


def test_cache_is_caller_owned_and_prompt_isolation_is_stable():
    first, second = RubricCache(), RubricCache()
    first.consider("q", _rubric("varies"), [0, 1])
    assert [rubric.title for rubric in first.get("q").positive] == ["varies"]
    assert second.get("q").all == ()


def test_cache_rejects_non_discriminative_rubrics():
    cache = RubricCache()
    assert not cache.consider("q", _rubric("flat"), [1, 1, 1])
    assert cache.get("q").all == ()
    with pytest.raises(ValueError, match="finite"):
        cache.consider("q", _rubric("invalid"), [0, float("nan")])


def test_cache_keeps_highest_variance_per_polarity():
    cache = RubricCache(max_per_polarity=2)
    cache.consider("q", _rubric("low"), [0.4, 0.5])
    cache.consider("q", _rubric("high"), [0, 1])
    cache.consider("q", _rubric("medium"), [0.2, 0.8])
    cache.consider("q", _rubric("negative", "negative"), [0, 1])
    assert [rubric.title for rubric in cache.get("q").positive] == ["high", "medium"]
    assert [rubric.title for rubric in cache.get("q").negative] == ["negative"]


def test_get_does_not_expose_mutable_cache_internals():
    cache = RubricCache()
    cache.consider("q", _rubric("good"), [0, 1])
    selected = cache.get("q")
    cache.consider("q", _rubric("better"), [0, 0.5, 1])
    assert [rubric.title for rubric in selected.positive] == ["good"]
