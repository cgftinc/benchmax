"""Tests for WikiBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from benchmax.rag.qa_generation.pipeline_config import WikiPreprocessingConfig
from benchmax.rag.qa_generation.corpus_profile import EntityPattern
from benchmax.rag.qa_generation.wiki_builder import (
    WikiBuilder,
    WikiIndex,
    WikiPage,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeChunk:
    content: str
    hash: str = ""
    metadata: tuple[tuple[str, Any], ...] = ()

    def chunk_str(self) -> str:
        return self.content


def _make_entity(
    name: str, etype: str = "entity", df: float = 0.05, quality_score: float = 0.9,
) -> EntityPattern:
    return EntityPattern(
        name=name, type=etype, document_frequency=df,
        idf_weight=1.0, quality_score=quality_score,
    )


def _make_config(**kwargs: Any) -> WikiPreprocessingConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "min_chunks_per_page": 2,
        "max_page_tokens": 500,
        "max_context_tokens": 1000,
        "model": "gpt-4o-mini",
        "api_key": "test-key",
    }
    defaults.update(kwargs)
    return WikiPreprocessingConfig(**defaults)


def _make_builder(**kwargs: Any) -> WikiBuilder:
    config = _make_config(**kwargs)
    client = MagicMock()
    return WikiBuilder(config, client)


# ---------------------------------------------------------------------------
# cluster_chunks tests
# ---------------------------------------------------------------------------


class TestClusterChunks:
    def test_basic_entity_clustering(self):
        """Chunks mentioning the same entity end up in the same cluster."""
        chunks = [
            FakeChunk(content="Redis supports TTL-based expiration.", hash="h1"),
            FakeChunk(content="Redis is fast for caching.", hash="h2"),
            FakeChunk(content="PostgreSQL is a relational database.", hash="h3"),
            FakeChunk(content="Redis clusters support sharding.", hash="h4"),
        ]
        entities = [_make_entity("Redis")]
        builder = _make_builder(min_chunks_per_page=2)

        clusters = builder.cluster_chunks(chunks, entities)

        assert "Redis" in clusters
        redis_hashes = {c.hash for c in clusters["Redis"]}
        assert redis_hashes == {"h1", "h2", "h4"}
        # PostgreSQL-only chunk not included
        assert "h3" not in redis_hashes

    def test_filters_below_min_chunks(self):
        """Clusters with fewer chunks than min_chunks_per_page are excluded."""
        chunks = [
            FakeChunk(content="Redis supports TTL.", hash="h1"),
            FakeChunk(content="PostgreSQL indexes.", hash="h2"),
            FakeChunk(content="PostgreSQL vacuum is important.", hash="h3"),
        ]
        # Redis appears in only 1 chunk; PostgreSQL appears in 2
        entities = [_make_entity("Redis"), _make_entity("PostgreSQL")]
        builder = _make_builder(min_chunks_per_page=2)

        clusters = builder.cluster_chunks(chunks, entities)

        assert "Redis" not in clusters, "Redis cluster has only 1 chunk, should be filtered"
        assert "PostgreSQL" in clusters

    def test_ignores_ubiquitous_entities(self):
        """Entities with DF >= 0.80 are excluded regardless of mentions."""
        chunks = [
            FakeChunk(content="The API is everywhere.", hash="h1"),
            FakeChunk(content="Call the API again.", hash="h2"),
            FakeChunk(content="API handles requests.", hash="h3"),
        ]
        # DF=0.95 means ubiquitous — should be ignored
        entities = [_make_entity("API", df=0.95, quality_score=0.2)]
        builder = _make_builder(min_chunks_per_page=2)

        clusters = builder.cluster_chunks(chunks, entities)

        assert "API" not in clusters

    def test_code_pattern_entity_uses_regex(self):
        """Code pattern entities match via regex, not substring."""
        chunks = [
            FakeChunk(content="Use GET /api/v1/users to fetch users.", hash="h1"),
            FakeChunk(content="POST /api/v2/items creates an item.", hash="h2"),
            FakeChunk(content="No routes here, just prose.", hash="h3"),
        ]
        entities = [_make_entity(r"(?:GET|POST) /api/v\d+/\w+", etype="code_pattern", df=0.05)]
        builder = _make_builder(min_chunks_per_page=2)

        clusters = builder.cluster_chunks(chunks, entities)

        entity_name = r"(?:GET|POST) /api/v\d+/\w+"
        assert entity_name in clusters
        hashes = {c.hash for c in clusters[entity_name]}
        assert hashes == {"h1", "h2"}
        assert "h3" not in hashes

    def test_clusters_merged_when_high_overlap(self):
        """Two entity clusters with >50% overlap are merged."""
        chunks = [
            FakeChunk(content="Redis TTL expires keys.", hash="h1"),
            FakeChunk(content="Redis TTL configuration.", hash="h2"),
            FakeChunk(content="TTL is also used in DNS.", hash="h3"),
        ]
        # Both Redis and TTL appear in h1, h2 — overlap is 2/2=100% > 50%
        entities = [_make_entity("Redis"), _make_entity("TTL")]
        builder = _make_builder(min_chunks_per_page=2)

        clusters = builder.cluster_chunks(chunks, entities)

        # Should only have one merged cluster
        assert len(clusters) == 1

    def test_no_clusters_when_all_filtered(self):
        """Returns empty dict when all entities produce clusters below min size."""
        chunks = [
            FakeChunk(content="Redis is fast.", hash="h1"),
            FakeChunk(content="Unrelated content.", hash="h2"),
        ]
        entities = [_make_entity("Redis")]
        builder = _make_builder(min_chunks_per_page=3)

        clusters = builder.cluster_chunks(chunks, entities)

        assert clusters == {}


# ---------------------------------------------------------------------------
# get_wiki_context tests
# ---------------------------------------------------------------------------


class TestGetWikiContext:
    def _make_index(self) -> WikiIndex:
        """Build a small WikiIndex for testing."""
        index = WikiIndex()
        index.pages["Redis"] = WikiPage(
            title="Redis",
            content="Redis is an in-memory data store.\n\n## TTL\nKeys expire after TTL.",
            source_chunk_ids=["hash_a", "hash_b"],
            cross_links=["TTL"],
            entity_names=["Redis"],
        )
        index.pages["PostgreSQL"] = WikiPage(
            title="PostgreSQL",
            content="PostgreSQL is a relational database.",
            source_chunk_ids=["hash_c"],
            cross_links=[],
            entity_names=["PostgreSQL"],
        )
        index.chunk_to_pages = {
            "hash_a": ["Redis"],
            "hash_b": ["Redis"],
            "hash_c": ["PostgreSQL"],
        }
        return index

    def test_returns_relevant_page_for_primary_chunk(self):
        """Pages covering the primary chunk are returned."""
        index = self._make_index()
        primary = FakeChunk(content="Redis TTL config.", hash="hash_a")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [], max_tokens=2000)

        assert "Redis" in ctx
        assert "in-memory" in ctx

    def test_secondary_chunks_also_trigger_pages(self):
        """Pages covering secondary chunks are also included."""
        index = self._make_index()
        primary = FakeChunk(content="Some other chunk.", hash="hash_x")
        secondary = FakeChunk(content="Postgres related.", hash="hash_c")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [secondary], max_tokens=2000)

        assert "PostgreSQL" in ctx

    def test_empty_string_when_no_relevant_pages(self):
        """Returns empty string when no wiki pages cover the given chunks."""
        index = self._make_index()
        primary = FakeChunk(content="Totally unrelated.", hash="hash_unknown")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [], max_tokens=2000)

        assert ctx == ""

    def test_empty_string_for_empty_wiki_index(self):
        """Returns empty string when wiki index has no pages."""
        index = WikiIndex()
        primary = FakeChunk(content="Some content.", hash="hash_a")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [], max_tokens=2000)

        assert ctx == ""

    def test_pages_ranked_by_chunk_coverage(self):
        """Pages that cover more of the task's chunks appear first."""
        index = WikiIndex()
        # 'Redis' covers both chunks; 'PostgreSQL' covers only one
        index.pages["Redis"] = WikiPage(
            title="Redis",
            content="A" * 200,
            source_chunk_ids=["h1", "h2"],
            cross_links=[],
            entity_names=["Redis"],
        )
        index.pages["PostgreSQL"] = WikiPage(
            title="PostgreSQL",
            content="B" * 200,
            source_chunk_ids=["h1"],
            cross_links=[],
            entity_names=["PostgreSQL"],
        )
        index.chunk_to_pages = {
            "h1": ["Redis", "PostgreSQL"],
            "h2": ["Redis"],
        }
        primary = FakeChunk(content="primary", hash="h1")
        secondary = FakeChunk(content="secondary", hash="h2")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [secondary], max_tokens=2000)

        # Redis should appear before PostgreSQL in the output
        redis_pos = ctx.find("Redis")
        postgres_pos = ctx.find("PostgreSQL")
        assert redis_pos < postgres_pos

    def test_token_budget_limits_output(self):
        """Context is truncated to roughly max_tokens."""
        index = WikiIndex()
        long_content = "word " * 2000  # ~10 000 chars
        index.pages["Redis"] = WikiPage(
            title="Redis",
            content=long_content,
            source_chunk_ids=["h1"],
            cross_links=[],
            entity_names=["Redis"],
        )
        index.chunk_to_pages = {"h1": ["Redis"]}
        primary = FakeChunk(content="Redis.", hash="h1")
        builder = _make_builder()

        ctx = builder.get_wiki_context(index, primary, [], max_tokens=100)

        # 100 tokens * 4 chars/token = 400-char budget (plus header overhead)
        assert len(ctx) < 2000


# ---------------------------------------------------------------------------
# generate_pages tests (prompt construction + batch_process_sync call)
# ---------------------------------------------------------------------------


class TestGeneratePages:
    def test_calls_batch_process_sync_once_per_cluster(self):
        """batch_process_sync is called with one prompt per cluster."""
        chunks_a = [
            FakeChunk(content="Redis TTL.", hash="h1"),
            FakeChunk(content="Redis config.", hash="h2"),
        ]
        chunks_b = [
            FakeChunk(content="Postgres index.", hash="h3"),
            FakeChunk(content="Postgres vacuum.", hash="h4"),
        ]
        clusters = {"Redis": chunks_a, "PostgreSQL": chunks_b}

        fake_response = MagicMock()
        fake_response.answer = "## Redis\n\nRedis is fast.\n\nRelated topics:\n- PostgreSQL"
        fake_batch_result = MagicMock()
        fake_batch_result.responses = [fake_response, fake_response]

        client = MagicMock()
        config = _make_config()
        builder = WikiBuilder(config, client)

        with patch(
            "benchmax.rag.qa_generation.wiki_builder.batch_process_sync",
            return_value=fake_batch_result,
        ) as mock_bps:
            result = builder.generate_pages(clusters, "A corpus about caching.", "Caching docs.")

        mock_bps.assert_called_once()
        call_kwargs = mock_bps.call_args
        prompts_arg = call_kwargs.kwargs.get("prompts") or call_kwargs.args[2]
        assert len(prompts_arg) == 2

        assert "Redis" in result.pages
        assert "PostgreSQL" in result.pages

    def test_failed_responses_skipped(self):
        """Clusters whose LLM call fails (None response) are skipped silently."""
        clusters = {
            "Redis": [FakeChunk(content="Redis.", hash="h1"), FakeChunk(content="x", hash="h2")],
        }
        fake_batch_result = MagicMock()
        fake_batch_result.responses = [None]

        client = MagicMock()
        builder = WikiBuilder(_make_config(), client)

        with patch(
            "benchmax.rag.qa_generation.wiki_builder.batch_process_sync",
            return_value=fake_batch_result,
        ):
            result = builder.generate_pages(clusters, "summary", "desc")

        assert "Redis" not in result.pages

    def test_chunk_to_pages_index_populated(self):
        """WikiIndex.chunk_to_pages maps each source hash to the correct page title."""
        chunks = [
            FakeChunk(content="Redis TTL.", hash="hash_redis_1"),
            FakeChunk(content="Redis fast.", hash="hash_redis_2"),
        ]
        clusters = {"Redis": chunks}

        fake_response = MagicMock()
        fake_response.answer = "## Redis\n\nRedis is great.\n\nRelated topics:\n"
        fake_batch_result = MagicMock()
        fake_batch_result.responses = [fake_response]

        client = MagicMock()
        builder = WikiBuilder(_make_config(), client)

        with patch(
            "benchmax.rag.qa_generation.wiki_builder.batch_process_sync",
            return_value=fake_batch_result,
        ):
            result = builder.generate_pages(clusters, "summary", "desc")

        assert "Redis" in result.chunk_to_pages.get("hash_redis_1", [])
        assert "Redis" in result.chunk_to_pages.get("hash_redis_2", [])

    def test_returns_empty_index_for_empty_clusters(self):
        """Returns an empty WikiIndex when no clusters are provided."""
        client = MagicMock()
        builder = WikiBuilder(_make_config(), client)

        result = builder.generate_pages({}, "summary", "desc")

        assert result.pages == {}
        assert result.chunk_to_pages == {}
