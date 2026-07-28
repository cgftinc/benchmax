"""Tests for PineconeChunkSource.search_related — dedup, neighbor skip, sorting."""

from __future__ import annotations

from castform.rag.chunkers.models import Chunk
from fakes.pinecone import (
    FakeIndex,
    FileAwareFakeFiles,
    make_match,
    make_source,
)


class TestSkipSourceChunk:
    def test_skips_by_pinecone_id(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("src-1", "source"), make_match("other-2", "other")],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src-1"),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "other"

    def test_skips_by_content_match(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("id-1", "same content"), make_match("id-2", "different")],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="same content", metadata=())
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "different"


class TestNeighborSkip:
    def test_skips_adjacent_same_file(self):
        index = FakeIndex(
            matches_per_call=[
                [
                    make_match("1", "adj prev", file_path="a.md", chunk_index=0),
                    make_match("2", "adj next", file_path="a.md", chunk_index=2),
                    make_match("3", "far away", file_path="a.md", chunk_index=5),
                ],
            ]
        )
        source = make_source(index, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_pinecone_id", "src"), ("file_path", "a.md"), ("chunk_index", 1)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "far away"

    def test_includes_non_adjacent_same_file(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("5", "far chunk", file_path="a.md", chunk_index=10)],
            ]
        )
        source = make_source(index, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_pinecone_id", "src"), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1

    def test_no_neighbor_skip_different_file(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("1", "other file", file_path="b.md", chunk_index=1)],
            ]
        )
        source = make_source(index, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_pinecone_id", "src"), ("file_path", "a.md"), ("chunk_index", 1)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "other file"


class TestDedup:
    def test_deduplicates_by_match_id(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("dup-1", "dup")],
                [make_match("dup-1", "dup")],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert len(results) == 1
        assert set(results[0]["queries"]) == {"q1", "q2"}

    def test_dedup_preserves_max_score(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("dup-1", "dup", score=0.5)],
                [make_match("dup-1", "dup", score=0.9)],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert results[0]["max_score"] == 0.9


class TestSorting:
    def test_more_queries_sorts_higher(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("1", "multi"), make_match("2", "single")],
                [make_match("1", "multi")],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert results[0]["chunk"].content == "multi"
        assert len(results[0]["queries"]) == 2

    def test_cross_file_sorts_above_same_file(self):
        index = FakeIndex(
            matches_per_call=[
                [
                    make_match("1", "same file", file_path="a.md", chunk_index=5),
                    make_match("2", "cross file", file_path="b.md", chunk_index=0),
                ],
            ]
        )
        source = make_source(index, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_pinecone_id", "src"), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 2
        assert results[0]["same_file"] is False
        assert results[1]["same_file"] is True


class TestSameFileDetection:
    def test_same_file_flag(self):
        index = FakeIndex(
            matches_per_call=[
                [
                    make_match("1", "same", file_path="a.md", chunk_index=5),
                    make_match("2", "diff", file_path="b.md", chunk_index=0),
                ],
            ]
        )
        source = make_source(index, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_pinecone_id", "src"), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        by_content = {r["chunk"].content: r for r in results}
        assert by_content["same"]["same_file"] is True
        assert by_content["diff"]["same_file"] is False


class TestEmptyQueries:
    def test_empty_queries_returns_empty(self):
        index = FakeIndex()
        source = make_source(index)
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, [], top_k=5)
        assert results == []


class TestVectorSearchPath:
    """Confirm search_related works via the vector path (the only mode)."""

    def test_search_related_returns_results(self):
        index = FakeIndex(
            matches_per_call=[
                [make_match("1", "result one"), make_match("2", "result two")],
            ]
        )
        source = make_source(index)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["find related"], top_k=5)
        assert len(results) == 2
        assert all("chunk" in r for r in results)
        assert all("queries" in r for r in results)

    def test_embed_fn_called_for_queries(self):
        """Verify queries are actually embedded before search."""
        calls = []

        def tracking_embed(texts):
            calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)

        index = FakeIndex(matches_per_call=[[make_match("1", "result")]])
        source = make_source(index, embed_fn=tracking_embed)
        primary = Chunk(content="source", metadata=(("_pinecone_id", "src"),))
        source.search_related(primary, ["query one", "query two"], top_k=5)
        assert len(calls) == 1
        assert calls[0] == ["query one", "query two"]
