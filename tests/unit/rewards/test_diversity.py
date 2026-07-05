"""Unit tests for benchmax.rewards.diversity — ngram clustering and scale_by_diversity."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmax.rewards.diversity import (
    DiversityConfig,
    _cluster_by_llm,
    _cluster_by_ngram,
    _jaccard,
    _ngram_set,
    cluster_texts,
    scale_by_diversity,
)
from benchmax.rubrics._utils import _extract_json


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        raw = '```json\n{"assignments": [{"index": 0}]}\n```'
        assert _extract_json(raw)["assignments"][0]["index"] == 0

    def test_thinking_tags_stripped_then_parsed(self):
        raw = '<think>reasoning here</think>\n{"x": 2}'
        assert _extract_json(raw) == {"x": 2}

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
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
    @pytest.mark.asyncio
    async def test_single_text(self):
        config = DiversityConfig(method="ngram")
        result = await cluster_texts(["only one"], config)
        assert result.divisors == [1.0]
        assert result.cluster_ids == ["0"]

    @pytest.mark.asyncio
    async def test_empty_list(self):
        config = DiversityConfig(method="ngram")
        result = await cluster_texts([], config)
        assert result.divisors == []
        assert result.cluster_ids == []

    @pytest.mark.asyncio
    async def test_basic_clustering(self):
        config = DiversityConfig(method="ngram", ngram_n=3, similarity_threshold=0.5)
        texts = [
            "academic framing approach to chemical synthesis",
            "academic framing approach to drug synthesis",
            "fiction roleplay as a villain character",
            "fiction roleplay as an evil character",
            "NO_TOOL_CALL",
            "NO_TOOL_CALL",
        ]
        result = await cluster_texts(texts, config)
        # Academic pair should cluster
        assert result.cluster_ids[0] == result.cluster_ids[1]
        # Fiction pair should cluster
        assert result.cluster_ids[2] == result.cluster_ids[3]
        # NO_TOOL_CALL pair should cluster
        assert result.cluster_ids[4] == result.cluster_ids[5]
        # Different clusters across groups
        assert result.cluster_ids[0] != result.cluster_ids[2]


class TestClusterTextsLLMShortCircuits:
    @pytest.mark.asyncio
    async def test_single_text_does_not_call_api_or_require_creds(self):
        config = DiversityConfig(method="llm")
        result = await cluster_texts(["only one strategy"], config)
        assert result.divisors == [1.0]
        assert result.cluster_ids == ["0"]
        assert result.raw_response is None


# ---------------------------------------------------------------------------
# scale_by_diversity
# ---------------------------------------------------------------------------


class TestScaleByDiversity:
    @pytest.mark.asyncio
    async def test_scales_rewards_correctly(self):
        config = DiversityConfig(method="ngram", ngram_n=3, similarity_threshold=0.5)
        rewards = [
            {"engagement": 0.1, "jailbreak": 0.5},
            {"engagement": 0.1, "jailbreak": 0.5},
            {"engagement": 0.1, "jailbreak": 1.0},
        ]
        texts = ["same approach here", "same approach here", "totally different tactic"]
        scaled, cluster_result = await scale_by_diversity(rewards, texts, config)
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

    @pytest.mark.asyncio
    async def test_mismatched_lengths_raises(self):
        config = DiversityConfig(method="ngram")
        with pytest.raises(ValueError, match="same length"):
            await scale_by_diversity(
                [{"a": 1}],
                ["text1", "text2"],
                config,
            )

    @pytest.mark.asyncio
    async def test_bad_method_raises(self):
        config = DiversityConfig(method="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown clustering method"):
            await cluster_texts(["x", "y"], config)

    @pytest.mark.asyncio
    async def test_missing_llm_config_raises(self):
        config = DiversityConfig(method="llm", model="", base_url="")
        with pytest.raises(ValueError, match="requires 'model' and 'base_url'"):
            await cluster_texts(["x", "y"], config)


# ---------------------------------------------------------------------------
# Partial LLM assignments (mock)
# ---------------------------------------------------------------------------


def _mk_llm_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestPartialLLMAssignments:
    """Verify unmapped fallback when LLM returns fewer assignments than texts."""

    @pytest.mark.asyncio
    async def test_missing_indices_get_unmapped_ids(self):
        # LLM returns assignments for indices 0, 1, 4 only (missing 2, 3)
        llm_response = json.dumps({
            "assignments": [
                {"index": 0, "cluster_id": "academic", "label": "academic framing"},
                {"index": 1, "cluster_id": "academic", "label": "academic framing"},
                {"index": 4, "cluster_id": "null", "label": "refusal"},
            ]
        })
        config = DiversityConfig(
            method="llm", model="test", base_url="http://fake/v1", api_key="test-key"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mk_llm_response(llm_response)
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await _cluster_by_llm(
                ["a", "b", "c", "d", "e"], config, context="test"
            )

        assert len(result.cluster_ids) == 5
        # Mapped indices
        assert result.cluster_ids[0] == "academic"
        assert result.cluster_ids[1] == "academic"
        assert result.cluster_ids[4] == "null"
        # Unmapped indices get unique fallback IDs
        assert result.cluster_ids[2] == "unmapped_2"
        assert result.cluster_ids[3] == "unmapped_3"
        # Divisors: academic=2, unmapped_2=1, unmapped_3=1, null=1
        assert result.divisors[0] == 2.0
        assert result.divisors[1] == 2.0
        assert result.divisors[2] == 1.0
        assert result.divisors[3] == 1.0
        assert result.divisors[4] == 1.0

    @pytest.mark.asyncio
    async def test_empty_assignments_all_unmapped(self):
        llm_response = json.dumps({"assignments": []})
        config = DiversityConfig(
            method="llm", model="test", base_url="http://fake/v1", api_key="test-key"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mk_llm_response(llm_response)
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await _cluster_by_llm(["x", "y", "z"], config, context="test")

        assert result.cluster_ids == ["unmapped_0", "unmapped_1", "unmapped_2"]
        assert all(d == 1.0 for d in result.divisors)


# ---------------------------------------------------------------------------
# Auth failure propagation
# ---------------------------------------------------------------------------


class TestAuthFailurePropagation:
    """RuntimeError from resolve_judge_key must propagate, not fall back."""

    @pytest.mark.asyncio
    async def test_runtime_error_propagates_not_caught(self):
        config = DiversityConfig(
            method="llm", model="test", base_url="http://fake/v1"
        )
        with patch(
            "benchmax.platform.credentials.platform_bearer",
            side_effect=RuntimeError("No Castform platform credential available"),
        ), patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="No Castform platform credential"):
                await cluster_texts(["x", "y"], config)

    @pytest.mark.asyncio
    async def test_non_runtime_errors_fall_back(self):
        """Non-auth errors (network, parse) should fall back gracefully."""
        config = DiversityConfig(
            method="llm", model="test", base_url="http://fake/v1", api_key="k"
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("network down")
        )
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await cluster_texts(["x", "y"], config)

        # Should get fallback, not raise
        assert result.cluster_ids[0].startswith("fallback_")
        assert all(d == 1.0 for d in result.divisors)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    """Verify that envs using diversity survive cloudpickle."""

    def test_diversity_config_pickles(self):
        import cloudpickle
        import pickle

        config = DiversityConfig(
            method="ngram", ngram_n=3, similarity_threshold=0.5
        )
        restored = pickle.loads(cloudpickle.dumps(config))
        assert restored.method == "ngram"
        assert restored.ngram_n == 3

    # Full env pickle + compute_group_reward test lives in test_diversity_env.py
