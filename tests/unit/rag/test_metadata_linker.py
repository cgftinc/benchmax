"""Tests for MetadataChunkLinker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from benchmax.rag.qa_generation.corpus_profile import CorpusProfile, EntityPattern
from benchmax.rag.qa_generation.metadata_linker import (
    MetadataChunkLinker,
    MetadataLinkerConfig,
    _token_set_jaccard,
)


@dataclass(frozen=True)
class FakeChunk:
    content: str
    metadata: tuple[tuple[str, Any], ...] = ()
    hash: str = ""

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


def _make_profile() -> CorpusProfile:
    return CorpusProfile(
        corpus_summary="Test corpus about Redis and caching.",
        corpus_queries=["Redis TTL", "cache config"],
        entity_patterns=[
            EntityPattern(name="Redis", type="entity", document_frequency=0.3, idf_weight=1.2),
            EntityPattern(name="TTL", type="domain_term", document_frequency=0.2, idf_weight=1.6),
            EntityPattern(name="API", type="entity", document_frequency=0.95, idf_weight=0.05),
        ],
        ubiquitous_df_threshold=0.80,
        token_document_frequency={
            "redis": 30,
            "caching": 10,
            "supports": 80,
            "streaming": 5,
            "messaging": 3,
            "expiration": 8,
        },
        token_df_sample_size=100,
    )


def _make_source(search_results: list[dict]) -> MagicMock:
    source = MagicMock()
    source.search_related.return_value = search_results
    return source


def _make_search_results(chunks: list[FakeChunk]) -> list[dict]:
    return [{"chunk": c, "queries": [], "same_file": False, "max_score": 1.0} for c in chunks]


class TestMetadataChunkLinker:
    def test_finds_secondaries(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "TTL Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )
        bundle = linker.link(primary, target_hop_count=2)

        assert len(bundle.secondary_chunks) == 1
        assert bundle.secondary_chunks[0].hash == "secondary"
        assert bundle.target_hop_count == 2
        assert bundle.structural_hints["linker"] == "metadata"
        assert bundle.structural_hints["confidence"] == 1.0

    def test_filters_primary_chunk(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        results = _make_search_results([primary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(source, profile)
        bundle = linker.link(primary, target_hop_count=2)

        assert len(bundle.secondary_chunks) == 0

    def test_filters_same_file(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        same_file = FakeChunk(
            content="Redis also supports pub/sub messaging patterns for events.",
            metadata=(("file", "redis.md"),),
            hash="same_file",
        )
        results = _make_search_results([same_file])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(filter_same_file=True)
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 0

    def test_filters_tiny_chunks(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        tiny = FakeChunk(
            content="Short.",
            metadata=(("file", "other.md"),),
            hash="tiny",
        )
        results = _make_search_results([tiny])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10)
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 0

    def test_respects_max_secondaries(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        chunks = [
            FakeChunk(
                content=f"Chunk {i} about Redis and caching layer details.",
                metadata=(("file", f"file_{i}.md"),),
                hash=f"chunk_{i}",
            )
            for i in range(5)
        ]
        results = _make_search_results(chunks)
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                max_secondaries=2,
                min_chunk_chars=10,
                min_coherence=0.0,
                max_secondary_similarity=1.0,  # disable diversity filter; test is about max_secondaries
            ),
        )
        bundle = linker.link(primary, target_hop_count=4)
        assert len(bundle.secondary_chunks) == 2

    def test_lookup_returns_empty_secondaries(self):
        primary = FakeChunk(content="Redis supports TTL.", hash="p")
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(source, profile)
        bundle = linker.link(primary, target_hop_count=1)
        assert len(bundle.secondary_chunks) == 0

    def test_enriches_profile(self):
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )
        linker.link(primary, target_hop_count=2)

        assert len(profile.effective_query_templates) > 0
        assert profile.connection_distribution.get("cross_document", 0) > 0

    def test_deduplicates_results(self):
        primary = FakeChunk(
            content="Redis TTL config.",
            metadata=(("file", "a.md"), ("h2", "TTL Config")),
            hash="p",
        )
        dup = FakeChunk(
            content="Duplicate chunk about Redis cache layer configuration.",
            metadata=(("file", "b.md"),),
            hash="dup",
        )
        results = _make_search_results([dup, dup])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 1


class TestHeaderFirstQueries:
    def test_header_queries_appear_first(self):
        """h2/h3 metadata should be the first queries."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration in caching.",
            metadata=(
                ("file", "redis.md"),
                ("h2", "Redis Configuration"),
                ("h3", "TTL Settings"),
            ),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(retry_confidence=0.0)
        )
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        assert queries[0] == "Redis Configuration"
        assert queries[1] == "TTL Settings"

    def test_no_plain_entity_queries(self):
        """Plain entity names should NOT appear as standalone queries."""
        primary = FakeChunk(
            content="API endpoint for Redis cache operations with TTL.",
            metadata=(("file", "redis.md"), ("h2", "Cache Operations")),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(source, profile)
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        # "Redis" and "TTL" should NOT appear alone (API is ubiquitous)
        assert "Redis" not in queries
        assert "TTL" not in queries
        assert "API" not in queries

    def test_fallback_to_metadata_queries(self):
        """With no headers and no entity matches, fall back to BM25 queries."""
        primary = FakeChunk(
            content="Some content with no matching entities at all here.",
            metadata=(("file", "other.md"), ("h1", "Installation Guide")),
            hash="primary",
        )
        source = _make_source([])
        profile = CorpusProfile(entity_patterns=[])

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(retry_confidence=0.0)
        )
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        assert "Installation Guide" in queries


class TestCompoundEntityQueries:
    def test_compound_queries_generated(self):
        """When 2+ entities found in chunk, compound query should appear."""
        primary = FakeChunk(
            content="Redis uses TTL-based expiration for cache invalidation.",
            metadata=(("file", "redis.md"),),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(retry_confidence=0.0)
        )
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        assert "Redis TTL" in queries

    def test_single_entity_no_compound(self):
        """With only one entity in chunk, no compound query generated."""
        primary = FakeChunk(
            content="Redis supports pub/sub messaging for event streaming.",
            metadata=(("file", "redis.md"),),
            hash="primary",
        )
        # Profile only has Redis matching (TTL not in content)
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(source, profile)
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        # No compound query — only header or fallback queries
        assert not any(" " in q and "Redis" in q and "TTL" in q for q in queries)


class TestEntityCoherenceReranking:
    def test_reranks_by_shared_entities(self):
        """Candidates sharing more entities with primary rank higher."""
        primary = FakeChunk(
            content="Redis uses TTL-based expiration for cache keys.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        # Chunk with both Redis and TTL
        both_entities = FakeChunk(
            content="Configure Redis TTL for session cache expiration policy.",
            metadata=(("file", "config.md"),),
            hash="both",
        )
        # Chunk with neither
        no_entities = FakeChunk(
            content="PostgreSQL supports JSONB columns for document storage.",
            metadata=(("file", "pg.md"),),
            hash="none",
        )
        results = _make_search_results([no_entities, both_entities])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10)
        )
        bundle = linker.link(primary, target_hop_count=2)

        # both_entities should rank first due to shared entity count
        assert len(bundle.secondary_chunks) == 1
        assert bundle.secondary_chunks[0].hash == "both"


class TestSameFileFiltering:
    def test_filter_same_file_true_removes_same_file(self):
        """filter_same_file=True should exclude same-file chunks."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        same_file = FakeChunk(
            content="Redis also supports pub/sub messaging patterns for events.",
            metadata=(("file", "redis.md"),),
            hash="same_file",
        )
        results = _make_search_results([same_file])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                filter_same_file=True, min_chunk_chars=10, min_coherence=0.0
            ),
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 0

    def test_filter_same_file_false_allows_same_file(self):
        """filter_same_file=False should allow same-file chunks."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        same_file = FakeChunk(
            content="Redis also supports pub/sub messaging patterns for events.",
            metadata=(("file", "redis.md"),),
            hash="same_file",
        )
        results = _make_search_results([same_file])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                filter_same_file=False, min_chunk_chars=10, min_coherence=0.0
            ),
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 1


class TestCrossQuestionDedup:
    def test_secondaries_not_reused(self):
        """Chunks used as secondaries in Q1 should be excluded from Q2."""
        primary1 = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary1",
        )
        primary2 = FakeChunk(
            content="Redis caching layer overview and configuration.",
            metadata=(("file", "overview.md"), ("h2", "Caching Layer")),
            hash="primary2",
        )
        shared_secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="shared",
        )
        results = _make_search_results([shared_secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )

        # First question uses shared_secondary
        bundle1 = linker.link(primary1, target_hop_count=2)
        assert len(bundle1.secondary_chunks) == 1
        assert bundle1.secondary_chunks[0].hash == "shared"

        # Second question should NOT get shared_secondary
        bundle2 = linker.link(primary2, target_hop_count=2)
        assert len(bundle2.secondary_chunks) == 0

    def test_primary_not_reused_as_secondary(self):
        """A chunk used as primary in Q1 shouldn't appear as secondary in Q2."""
        primary1 = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="chunk_a",
        )
        primary2 = FakeChunk(
            content="Redis caching layer overview and configuration.",
            metadata=(("file", "overview.md"), ("h2", "Caching Layer")),
            hash="chunk_b",
        )
        # Search returns chunk_a (primary1) as a result for Q2
        results = _make_search_results([primary1])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10)
        )

        linker.link(primary1, target_hop_count=2)
        bundle2 = linker.link(primary2, target_hop_count=2)
        assert len(bundle2.secondary_chunks) == 0

    def test_reset_used_hashes(self):
        """reset_used_hashes clears dedup state."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )

        linker.link(primary, target_hop_count=2)
        linker.reset_used_hashes()

        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 1


class TestCoherenceFloor:
    def test_filters_low_overlap(self):
        """Candidates with near-zero Jaccard should be filtered."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration for cache keys.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        unrelated = FakeChunk(
            content="Shakespeare wrote Hamlet during the Elizabethan era of English drama.",
            metadata=(("file", "lit.md"),),
            hash="unrelated",
        )
        results = _make_search_results([unrelated])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.02),
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 0

    def test_zero_coherence_disables_filter(self):
        """min_coherence=0 should allow all candidates through."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration for cache keys.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        unrelated = FakeChunk(
            content="Shakespeare wrote Hamlet during the Elizabethan era of English drama.",
            metadata=(("file", "lit.md"),),
            hash="unrelated",
        )
        results = _make_search_results([unrelated])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0),
        )
        bundle = linker.link(primary, target_hop_count=2)
        assert len(bundle.secondary_chunks) == 1


class TestRetryWithContentQueries:
    def test_retry_finds_secondaries_on_second_attempt(self):
        """When header queries return nothing, content queries should retry."""
        primary = FakeChunk(
            content=(
                "Redis supports TTL-based expiration for cache invalidation."
                " Configure the default timeout in redis.conf."
            ),
            metadata=(("file", "redis.md"), ("h2", "Very Obscure Header")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="Configure the default timeout for Redis cache invalidation policy.",
            metadata=(("file", "config.md"),),
            hash="secondary",
        )

        call_count = 0
        empty_results: list[dict] = []
        good_results = _make_search_results([secondary])

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First search (header queries) returns nothing.
            # Second search (content queries) finds the secondary.
            if call_count == 1:
                return empty_results
            return good_results

        source = MagicMock()
        source.search_related.side_effect = side_effect
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                min_chunk_chars=10,
                retry_confidence=0.5,
                min_coherence=0.0,
            ),
        )
        bundle = linker.link(primary, target_hop_count=2)

        assert call_count == 2  # Two search calls (primary + retry)
        assert len(bundle.secondary_chunks) == 1
        assert bundle.secondary_chunks[0].hash == "secondary"
        assert bundle.structural_hints["retried"] is True

    def test_no_retry_when_confidence_is_high(self):
        """If first attempt succeeds, no retry should happen."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                min_chunk_chars=10, retry_confidence=0.5, min_coherence=0.0
            ),
        )
        bundle = linker.link(primary, target_hop_count=2)

        assert source.search_related.call_count == 1
        assert bundle.structural_hints["retried"] is False

    def test_retry_disabled_when_threshold_zero(self):
        """retry_confidence=0 should never retry."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Very Obscure Header")),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(retry_confidence=0.0),
        )
        bundle = linker.link(primary, target_hop_count=2)

        assert source.search_related.call_count == 1
        assert bundle.structural_hints["retried"] is False

    def test_retry_merges_without_duplicates(self):
        """Retry results should not duplicate chunks from the first attempt."""
        primary = FakeChunk(
            content=(
                "Redis supports TTL-based expiration for cache invalidation."
                " Configure the default timeout in redis.conf."
            ),
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        first_hit = FakeChunk(
            content="Redis cluster configuration and failover settings.",
            metadata=(("file", "cluster.md"),),
            hash="first",
        )
        retry_dup = FakeChunk(
            content="Redis cluster configuration and failover settings.",
            metadata=(("file", "cluster.md"),),
            hash="first",  # same hash as first_hit
        )
        retry_new = FakeChunk(
            content="Default timeout configuration for Redis cache layers.",
            metadata=(("file", "timeout.md"),),
            hash="new",
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_search_results([first_hit])
            return _make_search_results([retry_dup, retry_new])

        source = MagicMock()
        source.search_related.side_effect = side_effect
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                min_chunk_chars=10,
                max_secondaries=3,
                retry_confidence=1.0,  # Always retry (need 3, first gives 1)
                min_coherence=0.0,
            ),
        )
        bundle = linker.link(primary, target_hop_count=4)

        # Should have first_hit + retry_new (not retry_dup)
        hashes = [c.hash for c in bundle.secondary_chunks]
        assert "first" in hashes
        assert "new" in hashes
        assert hashes.count("first") == 1  # No duplicates


class TestContentQueryGeneration:
    def test_first_sentence_extracted(self):
        """Content queries should include the first sentence."""
        primary = FakeChunk(
            content="PostHog makes it easy to track user behavior. More details follow.",
            metadata=(("file", "intro.md"),),
            hash="p",
        )
        profile = _make_profile()
        source = _make_source([])
        linker = MetadataChunkLinker(source, profile)
        queries = linker._generate_content_queries(primary)
        assert any("PostHog makes it easy to track user behavior" in q for q in queries)

    def test_capitalized_phrase_extracted(self):
        """Content queries should include capitalized multi-word phrases."""
        primary = FakeChunk(
            content="Configure the Session Replay feature in your project settings.",
            metadata=(("file", "replay.md"),),
            hash="p",
        )
        profile = _make_profile()
        source = _make_source([])
        linker = MetadataChunkLinker(source, profile)
        queries = linker._generate_content_queries(primary)
        assert any("Session Replay" in q for q in queries)

    def test_empty_content(self):
        primary = FakeChunk(content="", metadata=(), hash="p")
        profile = _make_profile()
        source = _make_source([])
        linker = MetadataChunkLinker(source, profile)
        assert linker._generate_content_queries(primary) == []


class TestCustomHeaderKeys:
    def test_custom_header_keys_used_in_queries(self):
        """Custom header_keys should be used instead of default h1/h2/h3."""
        primary = FakeChunk(
            content="Korean legal statute text about commercial law.",
            metadata=(
                ("file", "commercial_act.json"),
                ("hierarchy", "Commercial Act > Chapter 3"),
                ("statute_index", "Article 42"),
            ),
            hash="primary",
        )
        source = _make_source([])
        profile = CorpusProfile(entity_patterns=[])

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                header_keys=("hierarchy", "statute_index"),
                retry_confidence=0.0,
            ),
        )
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        assert "Commercial Act > Chapter 3" in queries
        assert "Article 42" in queries

    def test_default_header_keys_backward_compat(self):
        """Default header_keys should match the original h2/h3/h1 behavior."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration in caching.",
            metadata=(
                ("file", "redis.md"),
                ("h2", "Redis Configuration"),
                ("h3", "TTL Settings"),
            ),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        # No header_keys specified — should use defaults
        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(retry_confidence=0.0)
        )
        linker.link(primary, target_hop_count=2)

        call_args = source.search_related.call_args
        queries = call_args[0][1]
        assert "Redis Configuration" in queries
        assert "TTL Settings" in queries


class TestReasoningModeAwareLinking:
    def test_factual_mode_unchanged(self):
        """factual mode should behave identically to no mode."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "TTL Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )
        bundle = linker.link(primary, target_hop_count=2, reasoning_mode="factual")

        assert len(bundle.secondary_chunks) == 1
        assert bundle.structural_hints["reasoning_mode"] == "factual"

    def test_empty_mode_backward_compat(self):
        """Empty reasoning_mode should behave identically to factual."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "TTL Expiration")),
            hash="primary",
        )
        secondary = FakeChunk(
            content="The caching layer uses Redis with 5m TTL for sessions.",
            metadata=(("file", "caching.md"),),
            hash="secondary",
        )
        results = _make_search_results([secondary])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0)
        )
        bundle = linker.link(primary, target_hop_count=2)

        assert len(bundle.secondary_chunks) == 1
        assert bundle.structural_hints["reasoning_mode"] == ""

    def test_sequential_allows_same_file(self):
        """Sequential mode should allow same-file chunks through."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration"), ("index", 0)),
            hash="primary",
        )
        same_file = FakeChunk(
            content="Redis also supports pub/sub messaging patterns for events.",
            metadata=(("file", "redis.md"), ("index", 3)),
            hash="same_file",
        )
        results = _make_search_results([same_file])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                filter_same_file=True, min_chunk_chars=10, min_coherence=0.0
            ),
        )
        # With filter_same_file=True, same_file would normally be filtered.
        # Sequential mode overrides to False.
        bundle = linker.link(primary, target_hop_count=2, reasoning_mode="sequential")
        assert len(bundle.secondary_chunks) == 1
        assert bundle.secondary_chunks[0].hash == "same_file"

    def test_sequential_without_index_falls_back(self):
        """Sequential mode without index metadata should fall back to factual."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        same_file = FakeChunk(
            content="Redis also supports pub/sub messaging patterns for events.",
            metadata=(("file", "redis.md"),),
            hash="same_file",
        )
        results = _make_search_results([same_file])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                filter_same_file=True, min_chunk_chars=10, min_coherence=0.0
            ),
        )
        # No index on primary → falls back to factual → filter_same_file=True
        bundle = linker.link(primary, target_hop_count=2, reasoning_mode="sequential")
        assert len(bundle.secondary_chunks) == 0

    def test_inference_lowers_coherence(self):
        """Inference mode should lower the coherence floor."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration for cache keys.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        # Low coherence chunk — shares few tokens with primary
        low_coherence = FakeChunk(
            content="PostgreSQL supports JSONB columns for document storage and indexing.",
            metadata=(("file", "pg.md"),),
            hash="low",
        )
        results = _make_search_results([low_coherence])
        source = _make_source(results)
        profile = _make_profile()

        # With default min_coherence=0.15, this would be filtered.
        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.15),
        )
        bundle_factual = linker.link(primary, target_hop_count=2, reasoning_mode="factual")
        linker.reset_used_hashes()
        bundle_inference = linker.link(primary, target_hop_count=2, reasoning_mode="inference")

        # Inference halves coherence floor (0.15 * 0.5 = 0.075)
        # so low_coherence chunk may pass in inference but not factual
        assert len(bundle_inference.secondary_chunks) >= len(bundle_factual.secondary_chunks)

    def test_temporal_with_dates(self):
        """Temporal mode should use date diversity as a tiebreaker.

        With 70% rank preservation and 30% date diversity, a date-diverse
        candidate can overtake a same-date candidate when they have similar
        coherence ranks. We test with 3 candidates where two have similar
        coherence but different dates.
        """
        primary = FakeChunk(
            content="Meeting with Alice about project kickoff planning details.",
            metadata=(
                ("file", "thread_1"),
                ("h2", "Project Kickoff"),
                ("date_start", "2024-01-15"),
            ),
            hash="primary",
        )
        # Three candidates at similar coherence, ordered by search rank.
        # Without temporal reranking: c1, c2, c3 (search order).
        # With temporal: c3 (most date-diverse) should move up.
        c1_same_date = FakeChunk(
            content="Project kickoff logistics with Alice confirmed.",
            metadata=(("file", "thread_2"), ("date_start", "2024-01-15")),
            hash="c1",
        )
        c2_close_date = FakeChunk(
            content="Project kickoff agenda with Alice finalized.",
            metadata=(("file", "thread_3"), ("date_start", "2024-01-20")),
            hash="c2",
        )
        c3_distant_date = FakeChunk(
            content="Project kickoff retrospective with Alice completed.",
            metadata=(("file", "thread_4"), ("date_start", "2024-07-15")),
            hash="c3",
        )
        results = _make_search_results([c1_same_date, c2_close_date, c3_distant_date])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(min_chunk_chars=10, min_coherence=0.0),
        )

        # Factual: preserves search order
        linker.reset_used_hashes()
        bundle_factual = linker.link(primary, target_hop_count=4, reasoning_mode="factual")

        # Temporal: date-diverse candidate should move up
        linker.reset_used_hashes()
        bundle_temporal = linker.link(primary, target_hop_count=4, reasoning_mode="temporal")

        assert len(bundle_temporal.secondary_chunks) == 3
        # The distant-date chunk should rank higher in temporal than factual
        temporal_ranks = [c.hash for c in bundle_temporal.secondary_chunks]
        factual_ranks = [c.hash for c in bundle_factual.secondary_chunks]
        assert temporal_ranks.index("c3") <= factual_ranks.index("c3")

    def test_temporal_without_dates_falls_back(self):
        """Temporal mode without date metadata should fall back to factual."""
        primary = FakeChunk(
            content="Redis supports TTL-based expiration.",
            metadata=(("file", "redis.md"), ("h2", "Expiration")),
            hash="primary",
        )
        source = _make_source([])
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source, profile, config=MetadataLinkerConfig(retry_confidence=0.0)
        )
        bundle = linker.link(primary, target_hop_count=2, reasoning_mode="temporal")

        # Should not crash; just uses factual behavior
        assert bundle.structural_hints["reasoning_mode"] == "temporal"


class TestSecondaryDiversityThreshold:
    def test_linker_rejects_high_overlap_secondary(self):
        """Two candidates with Jaccard > 0.55 to each other → only one selected."""
        primary = FakeChunk(
            content="PostgreSQL handles ACID transactions and row-level locking.",
            metadata=(("file", "pg.md"), ("h2", "Transactions")),
            hash="primary",
        )
        # These two candidates share most tokens (Jaccard >> 0.55).
        candidate_a = FakeChunk(
            content="Redis cache session timeout expiration configuration layer setup.",
            metadata=(("file", "a.md"),),
            hash="candidate_a",
        )
        candidate_b = FakeChunk(
            content="Redis cache session timeout expiration configuration layer management.",
            metadata=(("file", "b.md"),),
            hash="candidate_b",
        )
        results = _make_search_results([candidate_a, candidate_b])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                min_chunk_chars=10,
                min_coherence=0.0,
                max_primary_similarity=1.0,  # disable primary ceiling for this test
                max_secondary_similarity=0.55,
                max_secondaries=2,
            ),
        )
        bundle = linker.link(primary, target_hop_count=3)

        # Only one of the two near-duplicate secondaries should be selected.
        assert len(bundle.secondary_chunks) == 1

    def test_linker_rejects_high_primary_overlap(self):
        """Candidate with Jaccard > 0.55 to primary → skipped."""
        # Primary and the "too similar" candidate share most tokens.
        primary = FakeChunk(
            content=(
                "Redis supports TTL-based expiration for cache keys timeout configuration setup."
            ),
            metadata=(("file", "redis.md"), ("h2", "Cache Config")),
            hash="primary",
        )
        # This candidate has very high overlap with primary → should be filtered.
        too_similar = FakeChunk(
            content=(
                "Redis supports TTL-based expiration for cache timeout configuration setup layer."
            ),
            metadata=(("file", "other.md"),),
            hash="too_similar",
        )
        # This candidate has lower overlap → should pass.
        acceptable = FakeChunk(
            content="Memcached uses slab allocation for memory management across nodes.",
            metadata=(("file", "memcached.md"),),
            hash="acceptable",
        )
        results = _make_search_results([too_similar, acceptable])
        source = _make_source(results)
        profile = _make_profile()

        linker = MetadataChunkLinker(
            source,
            profile,
            config=MetadataLinkerConfig(
                min_chunk_chars=10,
                min_coherence=0.0,
                max_primary_similarity=0.55,
                max_secondaries=2,
            ),
        )
        bundle = linker.link(primary, target_hop_count=3)

        hashes = [c.hash for c in bundle.secondary_chunks]
        assert "too_similar" not in hashes
        assert "acceptable" in hashes


class TestTokenSetJaccard:
    def test_identical_texts(self):
        assert _token_set_jaccard("hello world", "hello world") == 1.0

    def test_disjoint_texts(self):
        assert _token_set_jaccard("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        result = _token_set_jaccard("hello world foo", "hello world bar")
        # intersection={"hello","world"}, union={"hello","world","foo","bar"}
        assert abs(result - 0.5) < 0.01

    def test_empty_text(self):
        assert _token_set_jaccard("", "hello") == 0.0
        assert _token_set_jaccard("hello", "") == 0.0
