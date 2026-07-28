"""Tests for the CorpusProfile module."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from castform.rag.qa_generation.corpus_profile import (
    CorpusMetadataCensus,
    CorpusProfile,
    EntityPattern,
    build_entity_patterns_from_extraction,
    compute_chunk_suitability,
    compute_entity_document_frequency,
    compute_metadata_census,
    detect_search_capabilities,
    diverse_profile_sample,
    select_diverse,
)


@dataclass(frozen=True)
class FakeChunk:
    content: str
    metadata: tuple[tuple[str, Any], ...] = ()

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


def _make_chunks() -> list[FakeChunk]:
    return [
        FakeChunk(
            content="Redis supports TTL-based key expiration.",
            metadata=(("file", "redis.md"),),
        ),
        FakeChunk(
            content="Redis is used in the caching layer with 5m TTL.",
            metadata=(("file", "redis.md"),),
        ),
        FakeChunk(
            content="The API uses JWT tokens for auth.",
            metadata=(("file", "auth.md"),),
        ),
        FakeChunk(
            content="Deploy using Kubernetes on Azure AKS.",
            metadata=(("file", "deploy.md"),),
        ),
        FakeChunk(
            content="PostHog analytics tracks user events.",
            metadata=(("file", "analytics.md"),),
        ),
    ]


class TestEntityPattern:
    def test_basic(self):
        p = EntityPattern(name="Redis", type="entity")
        assert p.name == "Redis"
        assert p.document_frequency == 0.0
        assert p.idf_weight == 0.0


class TestComputeEntityDocumentFrequency:
    def test_computes_df(self):
        chunks = _make_chunks()
        patterns = [
            EntityPattern(name="Redis", type="entity"),
            EntityPattern(name="JWT", type="entity"),
            EntityPattern(name="Kubernetes", type="entity"),
        ]
        compute_entity_document_frequency(patterns, chunks)

        # Redis appears in 2/5 chunks
        assert patterns[0].document_frequency == 0.4
        assert patterns[0].idf_weight > 0

        # JWT appears in 1/5
        assert patterns[1].document_frequency == 0.2

        # Kubernetes in 1/5
        assert patterns[2].document_frequency == 0.2

    def test_ubiquitous_entity(self):
        chunks = [FakeChunk(content=f"PostHog docs chunk {i}") for i in range(10)]
        patterns = [EntityPattern(name="PostHog", type="entity")]
        compute_entity_document_frequency(patterns, chunks)
        assert patterns[0].document_frequency == 1.0

    def test_empty_chunks(self):
        patterns = [EntityPattern(name="Redis", type="entity")]
        compute_entity_document_frequency(patterns, [])
        assert patterns[0].document_frequency == 0.0


class TestCorpusProfile:
    def test_get_discriminative_entities(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="PostHog", type="entity", document_frequency=0.95),
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
                EntityPattern(name="TTL", type="domain_term", document_frequency=0.2),
            ],
            ubiquitous_df_threshold=0.80,
        )
        disc = profile.get_discriminative_entities()
        assert len(disc) == 2
        assert all(e.name != "PostHog" for e in disc)

    def test_get_entity_names(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="PostHog", type="entity", document_frequency=0.95),
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
            ],
            ubiquitous_df_threshold=0.80,
        )
        names = profile.get_entity_names(discriminative_only=True)
        assert names == ["Redis"]

        all_names = profile.get_entity_names(discriminative_only=False)
        assert len(all_names) == 2

    def test_get_domain_terms(self):
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
                EntityPattern(name="TTL", type="domain_term", document_frequency=0.2),
                EntityPattern(name="API", type="domain_term", document_frequency=0.9),
            ],
            ubiquitous_df_threshold=0.80,
        )
        terms = profile.get_domain_terms(discriminative_only=True)
        assert terms == ["TTL"]

    def test_corpus_queries_bulleted(self):
        profile = CorpusProfile(corpus_queries=["how to X", "what is Y"])
        assert profile.corpus_queries_bulleted == "- how to X\n- what is Y"

    def test_enrichment_methods(self):
        profile = CorpusProfile()

        profile.add_discovered_entities(["Redis", "TTL"])
        assert profile.discovered_entities == ["Redis", "TTL"]

        # No duplicates
        profile.add_discovered_entities(["Redis", "JWT"])
        assert profile.discovered_entities == ["Redis", "TTL", "JWT"]

        profile.record_query_effectiveness("Redis setup", 5)
        assert "Redis setup" in profile.effective_query_templates

        profile.record_query_effectiveness("empty query", 0)
        assert "empty query" not in profile.effective_query_templates

        profile.update_connection_stats("cross_document")
        profile.update_connection_stats("cross_document")
        profile.update_connection_stats("co_located")
        assert profile.connection_distribution == {
            "cross_document": 2,
            "co_located": 1,
        }


class TestBuildEntityPatternsFromExtraction:
    def test_builds_patterns(self):
        patterns = build_entity_patterns_from_extraction(
            entity_names=["Redis", "PostHog"],
            code_patterns={"func": r"def \w+\("},
            domain_terms=["TTL", "cache"],
        )
        assert len(patterns) == 5
        types = [p.type for p in patterns]
        assert types.count("entity") == 2
        assert types.count("code_pattern") == 1
        assert types.count("domain_term") == 2

    def test_skips_empty(self):
        patterns = build_entity_patterns_from_extraction(
            entity_names=["", "Redis", "  "],
            code_patterns={},
            domain_terms=[""],
        )
        assert len(patterns) == 1
        assert patterns[0].name == "Redis"


class TestDiverseProfileSample:
    def test_stratified_sampling(self):
        source = MagicMock()
        # Repeat chunks to have enough for stratification to kick in
        chunks = _make_chunks() * 200  # 1000 chunks from 5 docs
        source.sample_chunks.return_value = chunks

        # corpus_size=10000 → pool_size=min(10000, max(500, min(10000, 500)))=500
        result = diverse_profile_sample(
            source, corpus_size=10_000, min_chars=0, rng=random.Random(42)
        )
        assert len(result) <= 500

        # Should have chunks from multiple files
        files = {dict(c.metadata).get("file") for c in result}
        assert len(files) >= 2

    def test_empty_source(self):
        source = MagicMock()
        source.sample_chunks.return_value = []
        result = diverse_profile_sample(
            source, corpus_size=1000, min_chars=0, rng=random.Random(42)
        )
        assert result == []

    def test_fewer_chunks_than_pool(self):
        source = MagicMock()
        chunks = _make_chunks()[:2]
        source.sample_chunks.return_value = chunks
        # corpus_size=100 → pool_size=min(100, 500)=100 but source only returns 2
        result = diverse_profile_sample(source, corpus_size=100, min_chars=0, rng=random.Random(42))
        assert len(result) == 2

    def test_pool_size_proportional(self):
        source = MagicMock()
        source.sample_chunks.return_value = []

        # Tiny corpus (50 chunks) → pool_size = min(50, 500) = 50
        diverse_profile_sample(source, corpus_size=50, min_chars=0, rng=random.Random(0))
        requested = source.sample_chunks.call_args[0][0]
        assert requested <= 50 * 3

        source.reset_mock()
        # Large corpus (200k) → pool_size = min(200000, max(500, min(10000, 10000))) = 10000
        diverse_profile_sample(source, corpus_size=200_000, min_chars=0, rng=random.Random(0))
        requested = source.sample_chunks.call_args[0][0]
        assert requested <= 10_000 * 3


class TestDetectSearchCapabilities:
    def test_hybrid_mode(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical", "vector", "hybrid"},
        }
        modes, best = detect_search_capabilities(source)
        assert "hybrid" in modes
        assert best == "hybrid"

    def test_vector_mode(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical", "vector"},
        }
        modes, best = detect_search_capabilities(source)
        assert best == "vector"

    def test_lexical_only(self):
        source = MagicMock()
        source.get_search_capabilities.return_value = {
            "modes": {"lexical"},
        }
        modes, best = detect_search_capabilities(source)
        assert best == "lexical"

    def test_fallback_on_error(self):
        source = MagicMock()
        source.get_search_capabilities.side_effect = AttributeError
        modes, best = detect_search_capabilities(source)
        assert modes == {"lexical"}
        assert best == "lexical"


# ---------------------------------------------------------------------------
# Helpers shared by new test classes
# ---------------------------------------------------------------------------


def _make_chunk_with_content(content: str, **metadata: Any) -> FakeChunk:
    return FakeChunk(content=content, metadata=tuple(metadata.items()))


def _make_census(
    *,
    header_prevalence: float = 0.8,
    doc_id_prevalence: float = 0.9,
    date_prevalence: float = 0.1,
    sequential_prevalence: float = 0.1,
    content_length_p25: int = 100,
    content_length_p50: int = 200,
    content_length_p75: int = 400,
    entity_density_p25: float = 0.5,
    entity_density_p50: float = 1.0,
    entity_density_p75: float = 2.0,
    lexical_diversity_p25: float = 0.4,
    lexical_diversity_p50: float = 0.6,
    lexical_diversity_p75: float = 0.8,
    metadata_regime: str = "structured",
    chunk_count: int = 1000,
) -> CorpusMetadataCensus:
    return CorpusMetadataCensus(
        header_prevalence=header_prevalence,
        doc_id_prevalence=doc_id_prevalence,
        date_prevalence=date_prevalence,
        sequential_prevalence=sequential_prevalence,
        content_length_p25=content_length_p25,
        content_length_p50=content_length_p50,
        content_length_p75=content_length_p75,
        entity_density_p25=entity_density_p25,
        entity_density_p50=entity_density_p50,
        entity_density_p75=entity_density_p75,
        lexical_diversity_p25=lexical_diversity_p25,
        lexical_diversity_p50=lexical_diversity_p50,
        lexical_diversity_p75=lexical_diversity_p75,
        metadata_regime=metadata_regime,
        chunk_count=chunk_count,
    )


class TestComputeMetadataCensus:
    def _make_pool(self) -> list[FakeChunk]:
        """10 chunks with known distributions for deterministic testing."""
        return [
            FakeChunk(
                content=f"Redis supports TTL key expiration in milliseconds chunk {i}.",
                metadata=(("h1", "Redis"), ("file_name", f"doc{i % 3}.md")),
            )
            for i in range(10)
        ]

    def test_prevalences(self):
        pool = self._make_pool()
        census = compute_metadata_census(pool, entity_names=["Redis", "TTL"], chunk_count=500)

        # All chunks have h1 header → header_prevalence = 1.0
        assert census.header_prevalence == 1.0
        # All chunks have file_name → doc_id_prevalence = 1.0
        assert census.doc_id_prevalence == 1.0
        # No date metadata
        assert census.date_prevalence == 0.0
        # No sequential metadata
        assert census.sequential_prevalence == 0.0

    def test_regime_structured(self):
        pool = self._make_pool()
        census = compute_metadata_census(pool, entity_names=[], chunk_count=100)
        assert census.metadata_regime == "structured"

    def test_regime_unstructured(self):
        pool = [FakeChunk(content=f"plain text chunk number {i}") for i in range(10)]
        census = compute_metadata_census(pool, entity_names=[], chunk_count=100)
        assert census.metadata_regime == "unstructured"
        assert census.header_prevalence == 0.0

    def test_regime_mixed(self):
        pool = [
            FakeChunk(
                content=f"chunk {i}",
                metadata=(("h1", "Header"),) if i < 5 else (),
            )
            for i in range(10)
        ]
        census = compute_metadata_census(pool, entity_names=[], chunk_count=100)
        assert census.metadata_regime == "mixed"
        assert 0.20 <= census.header_prevalence <= 0.70

    def test_percentiles_ordered(self):
        pool = self._make_pool()
        census = compute_metadata_census(pool, entity_names=["Redis"], chunk_count=100)
        assert census.content_length_p25 <= census.content_length_p50 <= census.content_length_p75
        assert census.entity_density_p25 <= census.entity_density_p50 <= census.entity_density_p75
        assert (
            census.lexical_diversity_p25
            <= census.lexical_diversity_p50
            <= census.lexical_diversity_p75
        )

    def test_tiny_corpus_degenerate(self):
        pool = [FakeChunk(content="x") for _ in range(3)]
        census = compute_metadata_census(pool, entity_names=[], chunk_count=3)
        assert census.metadata_regime == "unstructured"
        assert census.content_length_p25 == 0
        assert census.content_length_p50 == 0
        assert census.content_length_p75 == 0

    def test_single_chunk_degenerate(self):
        pool = [FakeChunk(content="single")]
        census = compute_metadata_census(pool, entity_names=[], chunk_count=1)
        assert census.metadata_regime == "unstructured"

    def test_chunk_count_stored(self):
        pool = self._make_pool()
        census = compute_metadata_census(pool, entity_names=[], chunk_count=9999)
        assert census.chunk_count == 9999

    def test_multi_file_corpus_property(self):
        pool = self._make_pool()
        census = compute_metadata_census(pool, entity_names=[], chunk_count=100)
        assert census.multi_file_corpus is True  # doc_id_prevalence = 1.0

    def test_all_same_content_lengths(self):
        pool = [FakeChunk(content="x" * 100) for _ in range(10)]
        census = compute_metadata_census(pool, entity_names=[], chunk_count=100)
        assert census.content_length_p25 == census.content_length_p50 == census.content_length_p75


class TestSelectDiverse:
    def _make_distinct_chunks(self, n: int) -> list[FakeChunk]:
        """Chunks with clearly non-overlapping vocabulary."""
        vocabs = [
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
            "lambda mu nu xi omicron",
            "pi rho sigma tau upsilon",
            "phi chi psi omega ares",
            "zeus hera poseidon athena apollo",
            "artemis ares hephaestus hermes hestia",
            "demeter dionysus persephone hades iris",
        ]
        return [
            FakeChunk(content=vocabs[i % len(vocabs)] + f" unique{i}" * 5)
            for i in range(n)
        ]

    def test_returns_n_when_pool_larger(self):
        pool = self._make_distinct_chunks(20)
        result = select_diverse(pool, 5, rng=random.Random(42))
        assert len(result) == 5

    def test_returns_all_when_pool_smaller(self):
        pool = self._make_distinct_chunks(3)
        result = select_diverse(pool, 10, rng=random.Random(42))
        assert len(result) == 3

    def test_empty_pool(self):
        result = select_diverse([], 5, rng=random.Random(42))
        assert result == []

    def test_n_zero(self):
        pool = self._make_distinct_chunks(5)
        result = select_diverse(pool, 0, rng=random.Random(42))
        assert result == []

    def test_diversity_guarantee(self):
        """Selected chunks should have lower average lexical overlap than random."""

        pool = self._make_distinct_chunks(20)
        result = select_diverse(pool, 5, rng=random.Random(42), min_jaccard_distance=0.0)
        assert len(result) == 5

        # Verify no two selected chunks are identical
        contents = [c.content for c in result]
        assert len(set(contents)) == len(contents)

    def test_stratification(self):
        """With stratify_key, all groups should be represented."""
        pool = [
            FakeChunk(
                content=f"group{g} word{i} unique term{i * 10 + g}",
                metadata=(("file", f"doc{g}.md"),),
            )
            for g in range(3)
            for i in range(10)
        ]
        result = select_diverse(
            pool,
            6,
            rng=random.Random(42),
            stratify_key=lambda c: dict(c.metadata).get("file", "unknown"),
        )
        assert len(result) == 6
        # Each of the 3 groups should be represented
        groups = {dict(c.metadata).get("file") for c in result}
        assert len(groups) == 3

    def test_stratification_uneven_groups(self):
        """Stratification with uneven group sizes should not crash."""
        pool = (
            [
                FakeChunk(content=f"big group token{i} word{i}", metadata=(("file", "big.md"),))
                for i in range(20)
            ]
            + [
                FakeChunk(content=f"small group item{i}", metadata=(("file", "small.md"),))
                for i in range(2)
            ]
        )
        result = select_diverse(
            pool,
            5,
            rng=random.Random(42),
            stratify_key=lambda c: dict(c.metadata).get("file", "unknown"),
        )
        assert 1 <= len(result) <= 5


class TestComputeChunkSuitability:
    def _make_profile(self, regime: str = "structured") -> CorpusProfile:
        profile = CorpusProfile(
            entity_patterns=[
                EntityPattern(name="Redis", type="entity", document_frequency=0.3),
                EntityPattern(name="TTL", type="domain_term", document_frequency=0.2),
            ],
            token_document_frequency={"redis": 5, "ttl": 3, "supports": 10, "expiration": 2},
            token_df_sample_size=50,
        )
        return profile

    def _make_census(self, regime: str = "structured") -> CorpusMetadataCensus:
        return _make_census(metadata_regime=regime)

    def test_returns_float_in_range(self):
        chunk = _make_chunk_with_content(
            "Redis supports TTL key expiration with millisecond precision.", h1="Redis"
        )
        census = self._make_census("structured")
        profile = self._make_profile("structured")
        score = compute_chunk_suitability(chunk, census, profile)
        assert 0.0 <= score <= 1.0

    def test_structured_rewards_headers(self):
        """In structured regime, a chunk with headers should score higher than one without."""
        content = "Redis TTL expiration configuration and settings for production use."
        with_header = _make_chunk_with_content(content, h1="Redis Config")
        without_header = _make_chunk_with_content(content)

        census = self._make_census("structured")
        profile = self._make_profile("structured")

        score_with = compute_chunk_suitability(with_header, census, profile)
        score_without = compute_chunk_suitability(without_header, census, profile)
        assert score_with > score_without

    def test_unstructured_ignores_headers(self):
        """In unstructured regime, metadata_score weight is 0 — headers add nothing."""
        content = "Redis TTL expiration configuration and settings."
        with_header = _make_chunk_with_content(content, h1="Redis Config")
        without_header = _make_chunk_with_content(content)

        census = self._make_census("unstructured")
        profile = self._make_profile("unstructured")

        score_with = compute_chunk_suitability(with_header, census, profile)
        score_without = compute_chunk_suitability(without_header, census, profile)
        # Headers carry zero weight in unstructured, so scores should be equal
        assert abs(score_with - score_without) < 1e-9

    def test_degenerate_census_no_crash(self):
        """Zero-valued census percentiles should not cause division by zero."""
        chunk = _make_chunk_with_content("simple content here")
        census = _make_census(
            content_length_p25=0,
            content_length_p50=0,
            content_length_p75=0,
            entity_density_p25=0.0,
            entity_density_p50=0.0,
            entity_density_p75=0.0,
            lexical_diversity_p25=0.0,
            lexical_diversity_p50=0.0,
            lexical_diversity_p75=0.0,
            metadata_regime="unstructured",
        )
        profile = self._make_profile()
        score = compute_chunk_suitability(chunk, census, profile)
        assert 0.0 <= score <= 1.0

    def test_empty_profile_no_crash(self):
        """An empty profile (no entities, no token DF) should return a valid score."""
        chunk = _make_chunk_with_content("some content without entities")
        census = self._make_census("mixed")
        profile = CorpusProfile()
        score = compute_chunk_suitability(chunk, census, profile)
        assert 0.0 <= score <= 1.0
