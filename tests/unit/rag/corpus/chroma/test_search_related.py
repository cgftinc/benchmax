"""Tests for ChromaChunkSource.search_related — dedup, neighbor skip, sorting."""

from __future__ import annotations

from fakes.chroma import (
    FakeCollection,
    FileAwareFakeFiles,
    NoFileFakeFiles,
    make_query_result,
    make_source,
)

from benchmax.rag.chunkers.models import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qr(docs, metas=None, distances=None):
    """Shorthand for make_query_result."""
    return make_query_result(docs, metas=metas, distances=distances)


# ---------------------------------------------------------------------------
# Skip source chunk
# ---------------------------------------------------------------------------


class TestSkipSourceChunk:
    def test_skips_by_hash(self):
        """Source chunk is skipped by matching hash."""
        primary = Chunk(
            content="source content",
            metadata=(("file_path", "a.md"), ("chunk_index", 0)),
        )
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["source content", "other content"],
                    metas=[
                        {"file_path": "a.md", "chunk_index": 0, "chunk_hash": primary.hash},
                        {"file_path": "b.md", "chunk_index": 0},
                    ],
                ),
            ],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "other content"

    def test_skips_by_content_match(self):
        """When hash doesn't match, skips by identical content."""
        primary = Chunk(content="same content", metadata=())
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["same content", "different"],
                    metas=[
                        {"file_path": "x.md", "chunk_index": 0},
                        {"file_path": "y.md", "chunk_index": 0},
                    ],
                ),
            ],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "different"


# ---------------------------------------------------------------------------
# Neighbor skip
# ---------------------------------------------------------------------------


class TestNeighborSkip:
    def test_skips_adjacent_same_file(self):
        """Adjacent chunks (index ± 1) in same file are skipped."""
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["adj prev", "adj next", "far away"],
                    metas=[
                        {"file_path": "a.md", "chunk_index": 0},
                        {"file_path": "a.md", "chunk_index": 2},
                        {"file_path": "a.md", "chunk_index": 5},
                    ],
                ),
            ],
            count=10,
        )
        source = make_source(col, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("file_path", "a.md"), ("chunk_index", 1)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "far away"

    def test_includes_non_adjacent_same_file(self):
        """Non-adjacent chunks in same file are included."""
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["far chunk"],
                    metas=[{"file_path": "a.md", "chunk_index": 10}],
                ),
            ],
            count=10,
        )
        source = make_source(col, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_deduplicates_across_queries(self):
        """Same chunk returned by multiple queries → single entry, both queries."""
        result = _qr(
            ["dup content"],
            metas=[{"file_path": "b.md", "chunk_index": 0}],
        )
        col = FakeCollection(
            query_results_per_call=[result, result],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert len(results) == 1
        assert set(results[0]["queries"]) == {"q1", "q2"}

    def test_max_score_preserved(self):
        """When deduped, the higher score is kept."""
        # First query returns with distance 0.1 (score ~0.91)
        # Second query returns same chunk with distance 0.5 (score ~0.67)
        r1 = _qr(["dup"], metas=[{"file_path": "b.md", "chunk_index": 0}], distances=[0.1])
        r2 = _qr(["dup"], metas=[{"file_path": "b.md", "chunk_index": 0}], distances=[0.5])
        col = FakeCollection(query_results_per_call=[r1, r2], count=5)
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert len(results) == 1
        # Score from distance 0.1 is higher than from 0.5
        assert results[0]["max_score"] > 0.6


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


class TestSorting:
    def test_more_queries_sorts_higher(self):
        """Chunk matched by more queries sorts first."""
        multi = _qr(
            ["multi", "single"],
            metas=[
                {"file_path": "a.md", "chunk_index": 0},
                {"file_path": "b.md", "chunk_index": 0},
            ],
        )
        single = _qr(
            ["multi"],
            metas=[{"file_path": "a.md", "chunk_index": 0}],
        )
        col = FakeCollection(
            query_results_per_call=[multi, single],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert results[0]["chunk"].content == "multi"
        assert len(results[0]["queries"]) == 2

    def test_cross_file_sorts_above_same_file(self):
        """Cross-file results sort above same-file (same query count)."""
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["same file", "cross file"],
                    metas=[
                        {"file_path": "a.md", "chunk_index": 5},
                        {"file_path": "b.md", "chunk_index": 0},
                    ],
                ),
            ],
            count=5,
        )
        source = make_source(col, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 2
        assert results[0]["same_file"] is False
        assert results[1]["same_file"] is True


# ---------------------------------------------------------------------------
# Same-file detection
# ---------------------------------------------------------------------------


class TestSameFileDetection:
    def test_same_file_flag(self):
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["same", "diff"],
                    metas=[
                        {"file_path": "a.md", "chunk_index": 5},
                        {"file_path": "b.md", "chunk_index": 0},
                    ],
                ),
            ],
            count=5,
        )
        source = make_source(col, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        by_content = {r["chunk"].content: r for r in results}
        assert by_content["same"]["same_file"] is True
        assert by_content["diff"]["same_file"] is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_queries_returns_empty(self):
        col = FakeCollection(count=5)
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, [], top_k=5)
        assert results == []

    def test_no_file_metadata_no_neighbor_skip(self):
        """Without file metadata, adjacent chunks are NOT skipped."""
        col = FakeCollection(
            query_results_per_call=[
                _qr(
                    ["adjacent", "another"],
                    metas=[
                        {"file_path": "a.md", "chunk_index": 1},
                        {"file_path": "b.md", "chunk_index": 0},
                    ],
                ),
            ],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["query"], top_k=5)
        # Both appear — no neighbor skipping without file awareness
        assert len(results) == 2

    def test_no_file_metadata_same_file_always_false(self):
        col = FakeCollection(
            query_results_per_call=[
                _qr(["result"], metas=[{"file_path": "a.md", "chunk_index": 0}]),
            ],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["query"], top_k=5)
        assert results[0]["same_file"] is False
