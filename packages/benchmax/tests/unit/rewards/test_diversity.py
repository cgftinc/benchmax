import json
import pickle

import pytest

from benchmax.rewards import (
    JudgeError,
    LLMDiversityConfig,
    NgramDiversityConfig,
    cluster_texts,
    scale_by_diversity,
)
from benchmax.rewards.diversity import _jaccard, _ngram_set


def test_ngram_config_validates_inputs():
    with pytest.raises(ValueError, match="positive"):
        NgramDiversityConfig(n=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        NgramDiversityConfig(similarity_threshold=2)


def test_ngram_helpers():
    assert _ngram_set("Abcd", 2) == {"ab", "bc", "cd"}
    assert _ngram_set("", 3) == set()
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert _jaccard(set(), set()) == 1.0


@pytest.mark.asyncio
async def test_ngram_clustering_is_true_single_linkage():
    # ab~abc and abc~bc, while ab!~bc. All three belong to one connected
    # component under single-linkage.
    result = await cluster_texts(
        ["ab", "abc", "bc"],
        NgramDiversityConfig(n=1, similarity_threshold=0.6),
    )
    assert result.n_clusters == 1
    assert result.divisors == (3.0, 3.0, 3.0)


@pytest.mark.asyncio
async def test_empty_and_single_inputs_do_not_need_a_backend_call():
    config = NgramDiversityConfig()
    assert (await cluster_texts([], config)).cluster_ids == ()
    assert (await cluster_texts(["one"], config)).divisors == (1.0,)


@pytest.mark.asyncio
async def test_scale_by_diversity_divides_every_component():
    scaled, clusters = await scale_by_diversity(
        [{"a": 1.0, "b": 0.5}, {"a": 0.8, "b": 0.4}, {"a": 1.0}],
        ["same", "same", "different"],
        NgramDiversityConfig(similarity_threshold=0.9),
    )
    assert scaled[0] == {"a": 0.5, "b": 0.25}
    assert scaled[1] == {"a": 0.4, "b": 0.2}
    assert scaled[2] == {"a": 1.0}
    assert clusters.cluster_ids[0] == clusters.cluster_ids[1]


@pytest.mark.asyncio
async def test_scale_by_diversity_validates_alignment():
    with pytest.raises(ValueError, match="same length"):
        await scale_by_diversity([{"a": 1.0}], ["one", "two"], NgramDiversityConfig())


@pytest.mark.asyncio
async def test_llm_clustering_uses_shared_judge_and_typed_result(judge_factory):
    stub = judge_factory(
        [
            json.dumps(
                {
                    "assignments": [
                        {"index": 0, "cluster_id": "same", "label": "shared"},
                        {"index": 1, "cluster_id": "same", "label": "shared"},
                    ]
                }
            )
        ]
    )
    result = await cluster_texts(
        ["one", "two"],
        LLMDiversityConfig(judge=stub.judge),
        context="context",
    )
    assert result.cluster_ids == ("same", "same")
    assert result.labels == ("shared", "shared")
    assert result.divisors == (2.0, 2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assignments", "message"),
    [
        ([{"index": 0, "cluster_id": "a"}], "omitted"),
        (
            [
                {"index": 0, "cluster_id": "a"},
                {"index": 0, "cluster_id": "b"},
            ],
            "repeated",
        ),
        (
            [
                {"index": 0, "cluster_id": "a"},
                {"index": 2, "cluster_id": "b"},
            ],
            "out of range",
        ),
    ],
)
async def test_llm_clustering_rejects_partial_or_invalid_assignments(
    judge_factory, assignments, message
):
    stub = judge_factory([json.dumps({"assignments": assignments})])
    with pytest.raises(JudgeError, match=message):
        await cluster_texts(
            ["one", "two"],
            LLMDiversityConfig(judge=stub.judge),
        )


def test_configs_pickle_for_environment_bundles(judge_factory):
    stub = judge_factory([])
    assert pickle.loads(pickle.dumps(NgramDiversityConfig())).n == 3
    restored = pickle.loads(pickle.dumps(LLMDiversityConfig(judge=stub.judge)))
    assert restored.judge == stub.judge
