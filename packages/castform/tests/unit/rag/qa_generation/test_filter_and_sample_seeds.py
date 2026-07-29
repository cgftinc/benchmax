"""Tests for _filter_and_sample_seeds in pipeline."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any
from unittest.mock import MagicMock, patch

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.qa_generation.corpus_profile import CorpusMetadataCensus, CorpusProfile
from castform.rag.qa_generation.pipeline import _filter_and_sample_seeds


def _make_census() -> CorpusMetadataCensus:
    return CorpusMetadataCensus(
        header_prevalence=0.5,
        doc_id_prevalence=0.5,
        date_prevalence=0.0,
        sequential_prevalence=0.0,
        content_length_p25=50,
        content_length_p50=100,
        content_length_p75=200,
        entity_density_p25=0.0,
        entity_density_p50=1.0,
        entity_density_p75=2.0,
        lexical_diversity_p25=0.3,
        lexical_diversity_p50=0.5,
        lexical_diversity_p75=0.7,
        metadata_regime="unstructured",
        chunk_count=10,
    )


def _make_chunk(content: str, idx: int = 0) -> Chunk:
    return Chunk(content=content, metadata=(("file", f"doc{idx}.md"),))


def _make_source_with_collection(chunks: list[Chunk]) -> Any:
    source = MagicMock()
    source.collection = ChunkCollection(chunks)
    return source


def _make_profile_with_scores(chunks: list[Chunk], scores: list[float]) -> CorpusProfile:
    profile = CorpusProfile()
    profile.census = _make_census()
    profile.chunk_suitability_scores = {c.hash: s for c, s in zip(chunks, scores)}
    return profile


# ---------------------------------------------------------------------------
# Test 1: chunks below p25 threshold are never returned
# ---------------------------------------------------------------------------


def test_filter_and_sample_seeds_excludes_bottom_quartile() -> None:
    # 8 chunks with scores 0.1, 0.2, ..., 0.8
    # p25 index = 8//4 = 2 → threshold = scores[2] = 0.3
    # chunks with score <= 0.3 must never appear in results
    chunks = [_make_chunk(f"content chunk {i} " * 20, i) for i in range(8)]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    profile = _make_profile_with_scores(chunks, scores)
    source = _make_source_with_collection(chunks)

    rng = random.Random(42)
    # Run many times to catch non-deterministic violations
    for _ in range(20):
        result = _filter_and_sample_seeds(source, n=3, min_chars=50, profile=profile, rng=rng)
        result_hashes = {c.hash for c in result}
        bad_hashes = {c.hash for c, s in zip(chunks, scores) if s <= 0.3}
        assert result_hashes.isdisjoint(bad_hashes), (
            f"Bottom-quartile chunks appeared in result: {result_hashes & bad_hashes}"
        )


# ---------------------------------------------------------------------------
# Test 2: sampling is uniform (no weighting bias among eligible chunks)
# ---------------------------------------------------------------------------


def test_filter_and_sample_seeds_uniform() -> None:
    # 8 chunks with scores [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    # threshold = sorted_scores[8//4] = sorted_scores[2] = 0.6
    # Eligible (score > 0.6): 5 chunks with scores [0.65, 0.7, 0.75, 0.8, 0.9]
    # Verify those 5 eligible chunks appear with roughly equal frequency
    chunks = [_make_chunk(f"content chunk {i} " * 20, i) for i in range(8)]
    scores = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    profile = _make_profile_with_scores(chunks, scores)
    source = _make_source_with_collection(chunks)

    eligible_hashes = {c.hash for c, s in zip(chunks, scores) if s > 0.6}
    ineligible_hashes = {c.hash for c, s in zip(chunks, scores) if s <= 0.6}

    rng = random.Random(0)
    counter: Counter[str] = Counter()
    trials = 1000
    for _ in range(trials):
        result = _filter_and_sample_seeds(source, n=1, min_chars=50, profile=profile, rng=rng)
        assert len(result) == 1
        h = result[0].hash
        assert h not in ineligible_hashes, "Ineligible chunk returned"
        counter[h] += 1

    # Each of the 5 eligible chunks should appear ~200 times; allow ±120
    for h in eligible_hashes:
        count = counter[h]
        assert 80 <= count <= 320, (
            f"Eligible chunk {h[:8]} appeared {count} times in {trials} trials — "
            "expected roughly uniform distribution"
        )


# ---------------------------------------------------------------------------
# Test 3: fallback when profile is None
# ---------------------------------------------------------------------------


def test_filter_and_sample_seeds_fallback_no_profile() -> None:
    source = MagicMock()
    fallback = [_make_chunk("fallback content " * 10, i) for i in range(3)]
    source.sample_chunks.return_value = fallback

    rng = random.Random(42)
    result = _filter_and_sample_seeds(source, n=3, min_chars=50, profile=None, rng=rng)

    source.sample_chunks.assert_called_once_with(3, min_chars=50)
    assert result == fallback


# ---------------------------------------------------------------------------
# Test 4: fallback when eligible pool < n
# ---------------------------------------------------------------------------


def test_filter_and_sample_seeds_fallback_small_pool() -> None:
    # 6 chunks but only 2 above threshold — requesting n=5 triggers fallback
    # to uniform sampling from all min_chars-eligible candidates
    chunks = [_make_chunk(f"content chunk {i} " * 20, i) for i in range(6)]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    # threshold = scores[6//4] = scores[1] = 0.2 → eligible: scores > 0.2 = 4 chunks
    # n=5 > 4 eligible → falls back to rng.sample(candidates, 5)
    profile = _make_profile_with_scores(chunks, scores)
    source = _make_source_with_collection(chunks)

    rng = random.Random(42)
    result = _filter_and_sample_seeds(source, n=5, min_chars=50, profile=profile, rng=rng)

    # Fallback draws from all candidates (all 6 meet min_chars), not just eligible
    assert len(result) == 5
    assert all(c in chunks for c in result)


# ---------------------------------------------------------------------------
# Test 5: API backend path (no collection attribute)
# ---------------------------------------------------------------------------


class _NoCollectionSource:
    """Fake source that has no collection attribute (API-backed path)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[int, int]] = []

    def sample_chunks(self, n: int, *, min_chars: int = 0) -> list[Chunk]:
        self.calls.append((n, min_chars))
        return self._chunks[:n]


def test_filter_and_sample_seeds_api_backend() -> None:
    # 10 chunks; mock compute_chunk_suitability to return predictable scores
    # so we can test filtering without depending on the real scoring logic.
    chunks = [_make_chunk(f"content api chunk {i} " * 20, i) for i in range(10)]

    # chunk_suitability_scores has 10 entries → threshold = scores[10//4] = scores[2] = 0.3
    mock_scores_dict = {c.hash: 0.1 * (i + 1) for i, c in enumerate(chunks)}
    # Eligible (live score > 0.3): chunks 3–9

    profile = CorpusProfile()
    profile.census = _make_census()
    profile.chunk_suitability_scores = mock_scores_dict

    source = _NoCollectionSource(chunks)

    # Patch compute_chunk_suitability in pipeline to return the precomputed scores
    with patch(
        "castform.rag.qa_generation.pipeline.compute_chunk_suitability",
        side_effect=lambda c, census, prof: mock_scores_dict.get(c.hash, 0.0),
    ):
        rng = random.Random(42)
        result = _filter_and_sample_seeds(source, n=3, min_chars=50, profile=profile, rng=rng)

    assert len(result) == 3
    # All returned chunks should have live suitability > threshold (0.3)
    for c in result:
        score = mock_scores_dict.get(c.hash, 0)
        assert score > 0.3, f"Returned chunk with score {score} <= threshold 0.3"

    # sample_chunks was called once with n*2=6
    assert source.calls == [(6, 50)]
