import pytest

from benchmax.rubrics._utils import _static_rubric_key
from benchmax.rubrics.reward_fns import (
    _build_rubric_eval_tasks,
    group_rubric_based_reward_function,
    group_rubric_ranked_reward_function,
    single_rubric_based_reward_function,
)
from benchmax.rubrics.rubric import Rubric


def _r(title="Quality", type_="positive"):
    return Rubric(title=title, description="d", type=type_)


# ---------------------------------------------------------------------------
# _build_rubric_eval_tasks
# ---------------------------------------------------------------------------


def test_build_tasks_skips_judge_for_empty_text(stub_openai):
    stub_openai([])  # keep AsyncOpenAI patched so the live SDK never inits
    tasks, meta = _build_rubric_eval_tasks(
        ["", "ok"],
        [_r("A"), _r("B", "negative")],
        question="q",
        model_name="m",
        base_url="u",
        timeout=None,
    )
    try:
        assert len(tasks) == len(meta) == 4
        # meta carries (completion_index, rubric_type, rubric)
        assert [m[0] for m in meta] == [0, 0, 1, 1]
        assert [m[1] for m in meta] == ["positive", "negative", "positive", "negative"]
    finally:
        for t in tasks:
            t.close()


# ---------------------------------------------------------------------------
# group_rubric_based_reward_function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    [{"llm_judge_url": ""}, {"prompt": ""}, {"model": ""}],
)
@pytest.mark.asyncio
async def test_group_requires_kwargs(missing):
    kwargs = {"llm_judge_url": "u", "prompt": "q", "model": "m"} | missing
    with pytest.raises(ValueError):
        await group_rubric_based_reward_function(
            rollout_ids=["r1"], completions=["c"], ground_truths=[""], **kwargs
        )


@pytest.mark.asyncio
async def test_group_static_positive_passes_through_score(stub_openai):
    stub_openai(['{"score": 1, "reasoning": "ok"}'])
    out = await group_rubric_based_reward_function(
        rollout_ids=["r1"],
        completions=["resp"],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        static_rubrics=[_r("Clarity", "positive")],
    )
    assert out == [{_static_rubric_key("Clarity"): 1.0}]


@pytest.mark.asyncio
async def test_group_static_negative_flips_score(stub_openai):
    # Judge says the flaw IS present (score=1); negative reward should be 0.
    stub_openai(['{"score": 1, "reasoning": "flawed"}'])
    out = await group_rubric_based_reward_function(
        rollout_ids=["r1"],
        completions=["resp"],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        static_rubrics=[_r("Hallucination", "negative")],
    )
    assert out == [{_static_rubric_key("Hallucination"): 0.0}]


@pytest.mark.asyncio
async def test_group_empty_completion_scores_zero_without_judge(stub_openai):
    factory = stub_openai([])  # no canned responses -> would raise if called
    out = await group_rubric_based_reward_function(
        rollout_ids=["r1"],
        completions=[""],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        static_rubrics=[_r("Clarity", "positive")],
    )
    assert out == [{_static_rubric_key("Clarity"): 0}]
    assert factory.calls == []


@pytest.mark.asyncio
async def test_group_adaptive_rubrics_normalized_aggregate(stub_openai, tmp_cache_file):
    # 1 generation call producing 1 pos + 1 neg; 2 eval calls during generation
    # (variance check) + 2 eval calls for adaptive scoring; no static rubrics.
    # The judge says: Pos=1 (good), Neg=1 (flaw present)
    # adaptive_raw = +1 (pos) + (-1)*1 (neg) = 0
    # normalized = (0 + n_neg=1) / (n_pos=1 + n_neg=1) = 0.5
    stub_openai(
        [
            # generation: returns one pos + one neg
            '{"positive_rubrics": [{"title": "P", "description": "pd"}],'
            ' "negative_rubrics": [{"title": "N", "description": "nd"}]}',
            # variance probes during _generate_and_cache_rubrics (one per response)
            '{"score": 1}',  # P vs resp -> only 1 response, so 1 call
            '{"score": 1}',  # N vs resp
            # adaptive evaluation (one per (response, rubric))
            '{"score": 1}',  # P scoring
            '{"score": 1}',  # N scoring
        ]
    )
    out = await group_rubric_based_reward_function(
        rollout_ids=["r1"],
        completions=["resp"],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        use_adaptive_rubrics=True,
    )
    # With only 1 response, variance check skips the rubric (all-identical),
    # so the cache ends up empty, n_pos=n_neg=0, and the score_range fallback
    # (`or 1`) drives the normalization. raw=0 -> (0 + 0) / 1 = 0.
    assert out[0]["rubric_adaptive"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_group_adaptive_with_variance_aggregates(stub_openai, tmp_cache_file):
    # Two responses so variance can be non-zero and rubrics survive caching.
    # Generation -> 1 pos + 1 neg; variance: pos=[1,0] keep, neg=[1,0] keep.
    # Adaptive eval per (response, rubric): 2 resp x 2 rubrics = 4 calls.
    # We make Pos=1, Neg=0 for both responses so:
    #   adaptive_raw[i] = +1*1 + -1*0 = 1; n_pos=1 n_neg=1; range=2
    #   normalized = (1 + 1) / 2 = 1.0
    stub_openai(
        [
            '{"positive_rubrics": [{"title": "P", "description": "p"}],'
            ' "negative_rubrics": [{"title": "N", "description": "n"}]}',
            '{"score": 1}',  # variance: P vs r1
            '{"score": 0}',  # variance: P vs r2
            '{"score": 1}',  # variance: N vs r1
            '{"score": 0}',  # variance: N vs r2
            '{"score": 1}',  # adaptive eval: P vs r1
            '{"score": 1}',  # adaptive eval: P vs r2
            '{"score": 0}',  # adaptive eval: N vs r1
            '{"score": 0}',  # adaptive eval: N vs r2
        ]
    )
    out = await group_rubric_based_reward_function(
        rollout_ids=["r1", "r2"],
        completions=["resp1", "resp2"],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        use_adaptive_rubrics=True,
    )
    assert out[0]["rubric_adaptive"] == pytest.approx(1.0)
    assert out[1]["rubric_adaptive"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# group_rubric_ranked_reward_function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    [{"llm_judge_url": ""}, {"prompt": ""}, {"model": ""}],
)
@pytest.mark.asyncio
async def test_group_ranked_requires_kwargs(missing):
    kwargs = {"llm_judge_url": "u", "prompt": "q", "model": "m"} | missing
    with pytest.raises(ValueError):
        await group_rubric_ranked_reward_function(
            rollout_ids=["r1"], completions=["c"], ground_truths=[""], **kwargs
        )


@pytest.mark.asyncio
async def test_group_ranked_emits_one_key_per_rubric(stub_openai):
    stub_openai(['{"ranking": [[0], [1]]}'])  # no GT included
    out = await group_rubric_ranked_reward_function(
        rollout_ids=["r1", "r2"],
        completions=["A", "B"],
        ground_truths=[""],
        llm_judge_url="u",
        prompt="q",
        model="m",
        static_rubrics=[_r("Style")],
    )
    key = _static_rubric_key("Style")
    assert out[0][key] == pytest.approx(1.0)
    assert out[1][key] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_group_ranked_respects_include_ground_truth_flag(stub_openai):
    factory = stub_openai(['{"ranking": [[0], [1]]}'])
    await group_rubric_ranked_reward_function(
        rollout_ids=["r1", "r2"],
        completions=["A", "B"],
        ground_truths=["GT"],
        llm_judge_url="u",
        prompt="q",
        model="m",
        static_rubrics=[_r("Style")],
        include_ground_truth=False,
    )
    prompt = factory.calls[0]["messages"][0]["content"]
    assert "GT" not in prompt


# ---------------------------------------------------------------------------
# single_rubric_based_reward_function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_reward_scores_each_rubric(stub_openai):
    stub_openai(['{"score": 1}', '{"score": 1}'])
    out = await single_rubric_based_reward_function(
        rollout_id="r1",
        completion="resp",
        ground_truth="gt",
        rubrics=[_r("Pos", "positive"), _r("Neg", "negative")],
        llm_judge_url="u",
        prompt="q",
        model="m",
    )
    # positive raw=1 -> 1.0; negative raw=1 (flaw present) -> 0.0
    assert out == {_static_rubric_key("Pos"): 1.0, _static_rubric_key("Neg"): 0.0}


@pytest.mark.asyncio
async def test_single_reward_empty_completion_zero(stub_openai):
    factory = stub_openai([])
    out = await single_rubric_based_reward_function(
        rollout_id="r1",
        completion="",
        ground_truth="gt",
        rubrics=[_r("Pos", "positive")],
        llm_judge_url="u",
        prompt="q",
        model="m",
    )
    assert out == {_static_rubric_key("Pos"): 0}
    assert factory.calls == []
