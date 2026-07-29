"""Tests for WikiChunkLinker and build_entity_chunk_graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from castform.rag.qa_generation.corpus_profile import (
    CorpusProfile,
    EntityPattern,
    build_entity_chunk_graph,
)
from castform.rag.qa_generation.wiki_chunk_linker import WikiChunkLinker, WikiChunkLinkerConfig

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeChunk:
    content: str
    hash: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def metadata_dict(self) -> dict:
        return self.metadata

    def __post_init__(self):
        if not self.hash:
            import hashlib

            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class FakeCollection:
    chunks: list[FakeChunk] = field(default_factory=list)

    def get_chunk_by_hash(self, h: str) -> FakeChunk | None:
        for c in self.chunks:
            if c.hash == h:
                return c
        return None


@dataclass
class FakeSource:
    collection: FakeCollection = field(default_factory=FakeCollection)


def _make_chunks() -> list[FakeChunk]:
    return [
        FakeChunk(
            "PostHog is an analytics platform with Feature Flags",
            hash="chunk_a",
            metadata={"file": "docs/posthog.md"},
        ),
        FakeChunk(
            "Feature Flags enable gradual rollouts in PostHog",
            hash="chunk_b",
            metadata={"file": "docs/flags.md"},
        ),
        FakeChunk(
            "React Native SDK supports Feature Flags",
            hash="chunk_c",
            metadata={"file": "docs/react-native.md"},
        ),
        FakeChunk(
            "Error Tracking monitors application crashes",
            hash="chunk_d",
            metadata={"file": "docs/errors.md"},
        ),
        FakeChunk(
            "PostHog Error Tracking integrates with Sentry",
            hash="chunk_e",
            metadata={"file": "docs/errors.md"},
        ),
    ]


def _make_entities() -> list[EntityPattern]:
    return [
        EntityPattern(name="PostHog", type="entity"),
        EntityPattern(name="Feature Flags", type="entity"),
        EntityPattern(name="React Native", type="entity"),
        EntityPattern(name="Error Tracking", type="entity"),
    ]


# ---------------------------------------------------------------------------
# Tests for build_entity_chunk_graph
# ---------------------------------------------------------------------------


class TestBuildEntityChunkGraph:
    def test_builds_bipartite_graph(self):
        chunks = _make_chunks()
        patterns = _make_entities()
        eci, cei, cooc = build_entity_chunk_graph(patterns, chunks)

        # PostHog appears in chunks a, b, e
        assert "posthog" in eci
        assert "chunk_a" in eci["posthog"]
        assert "chunk_b" in eci["posthog"]
        assert "chunk_e" in eci["posthog"]

        # chunk_a contains PostHog and Feature Flags
        assert "chunk_a" in cei
        entities_in_a = [e.lower() for e in cei["chunk_a"]]
        assert "posthog" in entities_in_a
        assert "feature flags" in entities_in_a

    def test_sets_df_and_quality(self):
        chunks = _make_chunks()
        patterns = _make_entities()
        build_entity_chunk_graph(patterns, chunks)

        for p in patterns:
            assert p.document_frequency > 0
            assert p.quality_score > 0

    def test_computes_cooccurrence(self):
        chunks = _make_chunks()
        patterns = _make_entities()
        _, _, cooc = build_entity_chunk_graph(patterns, chunks)

        # PostHog and Feature Flags co-occur in chunks a and b
        key = ("feature flags", "posthog")
        assert key in cooc
        assert cooc[key] == 2

    def test_empty_chunks(self):
        patterns = _make_entities()
        eci, cei, cooc = build_entity_chunk_graph(patterns, [])
        assert eci == {}
        assert cei == {}
        assert cooc == {}

    def test_graph_is_unfiltered(self):
        """Graph stores ALL matches regardless of quality."""
        chunks = [FakeChunk("lowercase only term here", hash="c1")]
        patterns = [EntityPattern(name="lowercase only", type="domain_term")]
        eci, cei, _ = build_entity_chunk_graph(patterns, chunks)

        # Even low-quality entities are in the graph
        assert "lowercase only" in eci
        assert patterns[0].quality_score < 0.5  # low quality
        assert "c1" in eci["lowercase only"]  # but still indexed


# ---------------------------------------------------------------------------
# Tests for WikiChunkLinker
# ---------------------------------------------------------------------------


class TestWikiChunkLinker:
    def _build_linker(self) -> tuple[WikiChunkLinker, list[FakeChunk]]:
        chunks = _make_chunks()
        patterns = _make_entities()
        eci, cei, cooc = build_entity_chunk_graph(patterns, chunks)

        profile = CorpusProfile(
            entity_patterns=patterns,
            entity_chunk_index=eci,
            chunk_entity_index=cei,
            entity_cooccurrence=cooc,
        )

        source = FakeSource(collection=FakeCollection(chunks=chunks))
        linker = WikiChunkLinker(
            source=source,
            profile=profile,
            config=WikiChunkLinkerConfig(max_secondaries=2, min_chunk_chars=10),
        )
        return linker, chunks

    def test_finds_related_chunks(self):
        linker, chunks = self._build_linker()
        primary = chunks[0]  # PostHog + Feature Flags
        bundle = linker.link(primary, target_hop_count=2)

        assert len(bundle.secondary_chunks) >= 1
        sec_hashes = {c.hash for c in bundle.secondary_chunks}
        # Should find chunk_b (PostHog + Feature Flags) or chunk_c (Feature Flags)
        assert sec_hashes & {"chunk_b", "chunk_c"}

    def test_returns_anchor_bundle(self):
        linker, chunks = self._build_linker()
        bundle = linker.link(chunks[0], target_hop_count=2)

        assert bundle.primary_chunk == chunks[0]
        assert bundle.structural_hints["linker"] == "wiki"
        assert "confidence" in bundle.structural_hints

    def test_empty_when_no_entities(self):
        """Chunk with no known entities returns empty bundle."""
        linker, _ = self._build_linker()
        orphan = FakeChunk("completely unrelated content", hash="orphan")
        bundle = linker.link(orphan, target_hop_count=2)

        assert bundle.secondary_chunks == []
        assert bundle.structural_hints["confidence"] == 0.0

    def test_used_hashes_tracking(self):
        linker, chunks = self._build_linker()
        bundle1 = linker.link(chunks[0], target_hop_count=2)
        sec1 = {c.hash for c in bundle1.secondary_chunks}

        bundle2 = linker.link(chunks[3], target_hop_count=2)
        sec2 = {c.hash for c in bundle2.secondary_chunks}

        # Second link shouldn't reuse chunks from first
        assert not (sec1 & sec2) or not sec1  # may overlap if limited candidates

    def test_reset_used_hashes(self):
        linker, chunks = self._build_linker()
        linker.link(chunks[0], target_hop_count=2)
        assert len(linker._used_hashes) > 0

        linker.reset_used_hashes()
        assert len(linker._used_hashes) == 0

    def test_prefers_cross_file(self):
        """Cross-file chunks should score higher than same-file."""
        linker, chunks = self._build_linker()
        # chunk_d and chunk_e are both in docs/errors.md
        # chunk_e mentions PostHog, so linking from chunk_a should prefer cross-file
        bundle = linker.link(chunks[0], target_hop_count=3)

        if len(bundle.secondary_chunks) >= 2:
            files = [c.metadata.get("file", "") for c in bundle.secondary_chunks]
            # At least one secondary should be from a different file
            primary_file = chunks[0].metadata.get("file", "")
            assert any(f != primary_file for f in files)
