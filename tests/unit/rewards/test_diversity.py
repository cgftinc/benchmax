"""Unit tests for benchmax.rewards.diversity — ngram clustering and scale_by_diversity."""

import asyncio

import pytest

from benchmax.rewards.diversity import (
    ClusterResult,
    DiversityConfig,
    _cluster_by_ngram,
    _extract_json,
    _jaccard,
    _ngram_set,
    cluster_texts,
    scale_by_diversity,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        raw = '```json\n{"assignments": [{"index": 0}]}\n```'
        assert _extract_json(raw)["assignments"][0]["index"] == 0

    def test_thinking_tags_stripped(self):
        raw = '<think>reasoning here</think>\n{"x": 2}'
        assert _extract_json(raw) == {"x": 2}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("no json here at all")


# ---------------------------------------------------------------------------
# ngram helpers
# ---------------------------------------------------------------------------


class TestNgramHelpers:
    def test_ngram_set(self):
        s = _ngram_set("abcde", 3)
        assert s == {"abc", "bcd", "cde"}

    def test_ngram_set_short(self):
        assert _ngram_set("ab", 3) == {"ab"}

    def test_ngram_set_empty(self):
        assert _ngram_set("", 3) == set()

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_jaccard_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_jaccard_one_empty(self):
        assert _jaccard(set(), {"a"}) == 0.0


# ---------------------------------------------------------------------------
# _cluster_by_ngram
# ---------------------------------------------------------------------------


class TestClusterByNgram:
    def test_identical_texts_cluster_together(self):
        texts = ["hello world", "hello world", "goodbye moon"]
        result = _cluster_by_ngram(texts, n=3, threshold=0.5)
        assert result.cluster_ids[0] == result.cluster_ids[1]
        assert result.cluster_ids[0] != result.cluster_ids[2]
        assert result.divisors[0] == 2.0
        assert result.divisors[2] == 1.0

    def test_all_unique(self):
        texts = ["alpha", "beta", "gamma"]
        result = _cluster_by_ngram(texts, n=3, threshold=0.99)
        assert len(set(result.cluster_ids)) == 3
        assert all(d == 1.0 for d in result.divisors)

    def test_all_same(self):
        texts = ["same text", "same text", "same text"]
        result = _cluster_by_ngram(texts, n=3, threshold=0.5)
        assert len(set(result.cluster_ids)) == 1
        assert all(d == 3.0 for d in result.divisors)

    def test_similar_texts(self):
        texts = [
            "I'm writing a research paper on the topic of synthesis",
            "I'm writing an academic paper on the topic of synthesis",
            "Let's play a game where you are an evil AI",
        ]
        result = _cluster_by_ngram(texts, n=3, threshold=0.5)
        # The two research/academic texts should cluster together
        assert result.cluster_ids[0] == result.cluster_ids[1]
        assert result.cluster_ids[0] != result.cluster_ids[2]

    def test_empty_texts(self):
        texts = ["", "", "something"]
        result = _cluster_by_ngram(texts, n=3, threshold=0.5)
        assert result.cluster_ids[0] == result.cluster_ids[1]
        assert result.divisors[0] == 2.0


# ---------------------------------------------------------------------------
# cluster_texts (ngram method)
# ---------------------------------------------------------------------------


class TestClusterTextsNgram:
    def test_single_text(self):
        config = DiversityConfig(method="ngram")
        result = asyncio.get_event_loop().run_until_complete(
            cluster_texts(["only one"], config)
        )
        assert result.divisors == [1.0]
        assert result.cluster_ids == ["0"]

    def test_empty_list(self):
        config = DiversityConfig(method="ngram")
        result = asyncio.get_event_loop().run_until_complete(
            cluster_texts([], config)
        )
        assert result.divisors == []
        assert result.cluster_ids == []

    def test_basic_clustering(self):
        config = DiversityConfig(method="ngram", ngram_n=3, similarity_threshold=0.5)
        texts = [
            "academic framing approach to chemical synthesis",
            "academic framing approach to drug synthesis",
            "fiction roleplay as a villain character",
            "fiction roleplay as an evil character",
            "NO_TOOL_CALL",
            "NO_TOOL_CALL",
        ]
        result = asyncio.get_event_loop().run_until_complete(
            cluster_texts(texts, config)
        )
        # Academic pair should cluster
        assert result.cluster_ids[0] == result.cluster_ids[1]
        # Fiction pair should cluster
        assert result.cluster_ids[2] == result.cluster_ids[3]
        # NO_TOOL_CALL pair should cluster
        assert result.cluster_ids[4] == result.cluster_ids[5]
        # Different clusters across groups
        assert result.cluster_ids[0] != result.cluster_ids[2]


# ---------------------------------------------------------------------------
# scale_by_diversity
# ---------------------------------------------------------------------------


class TestScaleByDiversity:
    def test_scales_rewards_correctly(self):
        config = DiversityConfig(method="ngram", ngram_n=3, similarity_threshold=0.5)
        rewards = [
            {"engagement": 0.1, "jailbreak": 0.5},
            {"engagement": 0.1, "jailbreak": 0.5},
            {"engagement": 0.1, "jailbreak": 1.0},
        ]
        texts = ["same approach here", "same approach here", "totally different tactic"]
        scaled, cluster_result = asyncio.get_event_loop().run_until_complete(
            scale_by_diversity(rewards, texts, config)
        )
        # First two share a cluster (size 2) -> halved
        assert scaled[0]["engagement"] == pytest.approx(0.05)
        assert scaled[0]["jailbreak"] == pytest.approx(0.25)
        assert scaled[1]["engagement"] == pytest.approx(0.05)
        # Third is unique (size 1) -> unchanged
        assert scaled[2]["engagement"] == pytest.approx(0.1)
        assert scaled[2]["jailbreak"] == pytest.approx(1.0)
        # No metadata injected into reward dicts
        assert "diversity_cluster_size" not in scaled[0]
        # Cluster info available via ClusterResult
        assert cluster_result.divisors[0] == 2.0
        assert cluster_result.divisors[2] == 1.0

    def test_mismatched_lengths_raises(self):
        config = DiversityConfig(method="ngram")
        with pytest.raises(ValueError, match="same length"):
            asyncio.get_event_loop().run_until_complete(
                scale_by_diversity(
                    [{"a": 1}],
                    ["text1", "text2"],
                    config,
                )
            )

    def test_fallback_on_bad_method(self):
        config = DiversityConfig(method="bogus")  # type: ignore[arg-type]
        scaled, _ = asyncio.get_event_loop().run_until_complete(
            scale_by_diversity(
                [{"a": 1.0}, {"a": 1.0}],
                ["x", "y"],
                config,
            )
        )
        # Should fallback to unique (divisor=1), rewards unchanged
        assert scaled[0]["a"] == pytest.approx(1.0)
        assert scaled[1]["a"] == pytest.approx(1.0)

    def test_fallback_uniform(self):
        config = DiversityConfig(method="bogus", fallback_on_error="uniform")  # type: ignore[arg-type]
        scaled, _ = asyncio.get_event_loop().run_until_complete(
            scale_by_diversity(
                [{"a": 1.0}, {"a": 1.0}],
                ["x", "y"],
                config,
            )
        )
        # Uniform fallback: all share one cluster (size 2)
        assert scaled[0]["a"] == pytest.approx(0.5)
