"""Tests for PostgresChunkSource.search_related — dedup, neighbor skip, sorting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.postgres.source import PostgresChunkSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(file: str, index: int, content: str) -> Chunk:
    return Chunk(
        content=content,
        metadata=(("file", file), ("index", index)),
    )


def _make_source(
    chunks: list[Chunk],
    search_results_per_call: list[list[tuple[Chunk, float]]],
) -> PostgresChunkSource:
    """Build a PostgresChunkSource with mocked internals."""
    source = PostgresChunkSource.__new__(PostgresChunkSource)
    source.collection = ChunkCollection(chunks=chunks)
    source._corpus = SimpleNamespace(id="test-corpus")
    source._client = MagicMock()
    source._corpus_name = "test"

    call_idx = {"n": 0}

    async def fake_search_with_chunks(**kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx < len(search_results_per_call):
            return search_results_per_call[idx]
        return []

    source._client.search_with_chunks = fake_search_with_chunks
    source._search_capabilities = {
        "backend": "corpora",
        "modes": {"lexical"},
        "filter_ops": {
            "field": {"eq", "in", "gte", "lte", "contains_any", "contains_all"},
            "logical": {"and", "or", "not"},
        },
        "ranking": {"bm25"},
        "constraints": {"max_top_k": 1000},
        "graph_expansion": True,
    }
    return source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkipSourceChunk:
    @pytest.mark.asyncio
    async def test_skips_by_hash(self):
        src = _chunk("a.md", 0, "source content")
        other = _chunk("b.md", 0, "other content")
        source = _make_source(
            chunks=[src, other],
            search_results_per_call=[[(src, 0.9), (other, 0.8)]],
        )
        results = await source.search_related(src, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "other content"


class TestNeighborSkip:
    @pytest.mark.asyncio
    async def test_skips_adjacent_same_file(self):
        c0 = _chunk("a.md", 0, "chunk zero")
        c1 = _chunk("a.md", 1, "chunk one (source)")
        c2 = _chunk("a.md", 2, "chunk two")
        c5 = _chunk("a.md", 5, "chunk five (far)")
        source = _make_source(
            chunks=[c0, c1, c2, c5],
            search_results_per_call=[[(c0, 0.9), (c2, 0.85), (c5, 0.7)]],
        )
        results = await source.search_related(c1, ["query"], top_k=5)
        # c0 (index 0, diff 1) and c2 (index 2, diff 1) should be skipped
        assert len(results) == 1
        assert results[0]["chunk"].content == "chunk five (far)"


class TestDedup:
    @pytest.mark.asyncio
    async def test_deduplicates_by_hash(self):
        src = _chunk("a.md", 0, "source")
        dup = _chunk("b.md", 0, "dup content")
        source = _make_source(
            chunks=[src, dup],
            search_results_per_call=[
                [(dup, 0.9)],
                [(dup, 0.8)],
            ],
        )
        results = await source.search_related(src, ["q1", "q2"], top_k=5)
        assert len(results) == 1
        assert set(results[0]["queries"]) == {"q1", "q2"}
        assert results[0]["max_score"] == 0.9


class TestEmptyQueries:
    @pytest.mark.asyncio
    async def test_empty_queries_returns_empty(self):
        src = _chunk("a.md", 0, "source")
        source = _make_source(chunks=[src], search_results_per_call=[])
        results = await source.search_related(src, [], top_k=5)
        assert results == []


class TestSorting:
    @pytest.mark.asyncio
    async def test_more_queries_sorts_higher(self):
        src = _chunk("a.md", 0, "source")
        multi = _chunk("b.md", 0, "multi hit")
        single = _chunk("c.md", 0, "single hit")
        source = _make_source(
            chunks=[src, multi, single],
            search_results_per_call=[
                [(multi, 0.9), (single, 0.8)],
                [(multi, 0.7)],
            ],
        )
        results = await source.search_related(src, ["q1", "q2"], top_k=5)
        assert results[0]["chunk"].content == "multi hit"
        assert len(results[0]["queries"]) == 2
        assert results[1]["chunk"].content == "single hit"
        assert len(results[1]["queries"]) == 1


class TestSameFileFlag:
    @pytest.mark.asyncio
    async def test_detects_same_and_cross_file(self):
        src = _chunk("a.md", 0, "source")
        same = _chunk("a.md", 5, "same file far")
        diff = _chunk("b.md", 0, "different file")
        source = _make_source(
            chunks=[src, same, diff],
            search_results_per_call=[[(same, 0.8), (diff, 0.7)]],
        )
        results = await source.search_related(src, ["query"], top_k=5)
        by_content = {r["chunk"].content: r for r in results}
        assert by_content["same file far"]["same_file"] is True
        assert by_content["different file"]["same_file"] is False
