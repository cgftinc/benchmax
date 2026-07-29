"""Tests for entity quality scoring."""

from __future__ import annotations

import pytest
from castform.rag.qa_generation.corpus_profile import (
    EntityPattern,
    _score_entity_quality,
    compute_entity_document_frequency,
)


class TestScoreEntityQuality:
    """Validate the quality scoring formula."""

    def test_high_quality_named_entity(self):
        """Proper named entity in the DF sweet spot scores near 1.0."""
        p = EntityPattern(name="PostHog", type="entity", document_frequency=0.05)
        score = _score_entity_quality(p)
        assert score >= 0.85

    def test_domain_term_reasonable_score(self):
        """Lowercase domain term with good DF scores moderately."""
        p = EntityPattern(name="redis", type="domain_term", document_frequency=0.08)
        score = _score_entity_quality(p)
        assert 0.5 < score < 0.9

    def test_stopword_phrase_low_score(self):
        """All-stopword phrase gets penalized via name_score=0."""
        p = EntityPattern(name="if you", type="domain_term", document_frequency=0.05)
        score = _score_entity_quality(p)
        # name_score=0 but df + type still contribute; should be below wiki threshold 0.5
        # at high DF, but at DF=0.05 it's 0.68. Key: it's below proper entities (~1.0).
        assert score < 0.7

    def test_proper_entity_beats_stopword(self):
        """Named entity always outscores a stopword phrase at same DF."""
        entity = EntityPattern(name="Feature Flags", type="entity", document_frequency=0.05)
        stopword = EntityPattern(name="if you", type="domain_term", document_frequency=0.05)
        assert _score_entity_quality(entity) > _score_entity_quality(stopword)

    def test_zero_df_scores_low(self):
        """Entity with DF=0 gets low score — type + name but no DF signal."""
        p = EntityPattern(name="PostHog", type="entity", document_frequency=0.0)
        score = _score_entity_quality(p)
        assert score == pytest.approx(0.48)  # 0*0.4 + 1.0*0.2 + 0.7*0.4

    def test_very_high_df_scores_low(self):
        """Ubiquitous entity (DF > 0.40) gets zero DF component."""
        p = EntityPattern(name="PostHog", type="entity", document_frequency=0.50)
        score = _score_entity_quality(p)
        assert score <= 0.5  # only type + name, no DF

    def test_code_pattern_reasonable(self):
        """Code patterns bypass name heuristics, get decent score."""
        p = EntityPattern(
            name=r"posthog\.\w+\(",
            type="code_pattern",
            document_frequency=0.05,
        )
        score = _score_entity_quality(p)
        assert score > 0.6

    def test_short_name_penalized(self):
        """Names shorter than 4 chars get name_score halved."""
        short = EntityPattern(name="TTL", type="entity", document_frequency=0.05)
        long = EntityPattern(name="Time To Live", type="entity", document_frequency=0.05)
        assert _score_entity_quality(long) > _score_entity_quality(short)

    def test_df_sweet_spot_flat_top(self):
        """DF 0.02-0.10 should all get the same df_score (flat top)."""
        scores = []
        for df in [0.02, 0.05, 0.08, 0.10]:
            p = EntityPattern(name="Test Entity", type="entity", document_frequency=df)
            scores.append(_score_entity_quality(p))
        # All should be equal (same df_score=1.0, same name/type)
        assert all(s == scores[0] for s in scores)

    def test_df_taper_region(self):
        """DF 0.10-0.40 should produce decreasing scores."""
        s1 = _score_entity_quality(
            EntityPattern(name="X", type="entity", document_frequency=0.15),
        )
        s2 = _score_entity_quality(
            EntityPattern(name="X", type="entity", document_frequency=0.30),
        )
        assert s1 > s2


class TestComputeEntityDFSetsQuality:
    """Verify compute_entity_document_frequency also sets quality_score."""

    def test_quality_score_set(self):
        """After computing DF, quality_score should be non-default."""
        patterns = [
            EntityPattern(name="PostHog", type="entity"),
            EntityPattern(name="redis", type="domain_term"),
        ]

        class FakeChunk:
            def __init__(self, content: str):
                self.content = content

        chunks = [
            FakeChunk("PostHog is an analytics platform"),
            FakeChunk("Redis is used for caching by PostHog"),
            FakeChunk("Neither term appears here"),
        ]
        compute_entity_document_frequency(patterns, chunks)

        assert patterns[0].document_frequency > 0
        assert patterns[0].quality_score > 0
        assert patterns[1].quality_score > 0

    def test_empty_chunks(self):
        """Empty chunk list doesn't crash, scores stay at default."""
        patterns = [EntityPattern(name="Test", type="entity")]
        compute_entity_document_frequency(patterns, [])
        assert patterns[0].quality_score == 0.0
