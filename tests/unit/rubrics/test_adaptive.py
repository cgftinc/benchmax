import pytest

from benchmax.rubrics.adaptive import (
    _generate_and_cache_rubrics,
    generate_instance_wise_adaptive_rubrics,
)
from benchmax.rubrics.cache import _empty_cache_entry, load_rubric_cache


@pytest.mark.asyncio
async def test_generate_returns_parsed_json(stub_openai):
    stub_openai(['{"positive_rubrics": [{"title": "A", "description": "d"}], "negative_rubrics": []}'])
    out = await generate_instance_wise_adaptive_rubrics(
        question="q",
        ground_truth="gt",
        response_list=["r1", "r2"],
        model_name="m",
        base_url="u",
    )
    assert out == {"positive_rubrics": [{"title": "A", "description": "d"}], "negative_rubrics": []}


@pytest.mark.asyncio
async def test_generate_returns_none_on_empty(stub_openai):
    stub_openai([""])
    assert (
        await generate_instance_wise_adaptive_rubrics(
            question="q", ground_truth="gt", response_list=["r"], model_name="m", base_url="u"
        )
        is None
    )


@pytest.mark.asyncio
async def test_generate_returns_none_on_exception(stub_openai):
    def raiser(_):
        raise RuntimeError("boom")

    stub_openai(raiser)
    assert (
        await generate_instance_wise_adaptive_rubrics(
            question="q", ground_truth="gt", response_list=["r"], model_name="m", base_url="u"
        )
        is None
    )


@pytest.mark.asyncio
async def test_generate_appends_existing_rubrics_to_prompt(stub_openai):
    factory = stub_openai(['{"positive_rubrics": [], "negative_rubrics": []}'])
    await generate_instance_wise_adaptive_rubrics(
        question="q",
        ground_truth="gt",
        response_list=["r"],
        model_name="m",
        base_url="u",
        existing_rubrics="PRIOR_RUBRICS_BLOCK",
    )
    assert "PRIOR_RUBRICS_BLOCK" in factory.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_generate_and_cache_filters_and_persists(stub_openai, tmp_cache_file):
    # 1st call: generation returns one positive + one negative rubric.
    # Subsequent calls: each rubric is evaluated against each of the 2 responses.
    # We arrange the variance so the positive rubric survives (mixed scores) and
    # the negative rubric is dropped (all zeros -> zero variance).
    stub_openai(
        [
            '{"positive_rubrics": [{"title": "Pos", "description": "p"}],'
            ' "negative_rubrics": [{"title": "Neg", "description": "n"}]}',
            '{"score": 1}',  # Pos vs resp1
            '{"score": 0}',  # Pos vs resp2  -> variance -> keep
            '{"score": 0}',  # Neg vs resp1
            '{"score": 0}',  # Neg vs resp2  -> no variance -> skip
        ]
    )
    cache = await _generate_and_cache_rubrics(
        completion_texts=["a", "b"],
        user_prompt="q",
        ground_truth="gt",
        model_name="m",
        llm_judge_url="u",
        timeout=None,
        question_hash="qhash",
        existing_rubrics=None,
        cache=_empty_cache_entry(),
    )
    assert [r["title"] for r in cache["positive_rubrics"]] == ["Pos"]
    assert cache["negative_rubrics"] == []
    # Cache is persisted under the question_hash
    assert load_rubric_cache()["qhash"]["positive_rubrics"][0]["title"] == "Pos"
