"""Tests for TpufChunkSource.search_related — dedup, neighbor skip, sorting."""

from __future__ import annotations

from types import SimpleNamespace

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.turbopuffer.source import TpufChunkSource

# ---------------------------------------------------------------------------
# Fakes — reuse the pattern from test_tpuf_hybrid_search.py
# ---------------------------------------------------------------------------


class FakeNamespace:
    """Returns canned rows for each query call."""

    def __init__(self, rows_per_call: list[list] | None = None):
        self.rows_per_call = list(rows_per_call or [])
        self._call_idx = 0

    def query(self, **kwargs):
        if self._call_idx < len(self.rows_per_call):
            rows = self.rows_per_call[self._call_idx]
        else:
            rows = []
        self._call_idx += 1
        return SimpleNamespace(rows=list(rows))


class FileAwareFakeFiles:
    """File-aware fake that reads metadata from chunks.

    Reference: benchmax/rag/corpus/turbopuffer/files.py FileAwareness
    """

    def check(self) -> bool:
        return True

    @staticmethod
    def chunk_file_path(chunk: Chunk) -> str | None:
        return chunk.get_metadata("file_path") or chunk.get_metadata("file") or None

    @staticmethod
    def chunk_index(chunk: Chunk) -> int | None:
        val = chunk.get_metadata("chunk_index")
        if val is None:
            val = chunk.get_metadata("index")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None


class NoFileFakeFiles:
    def check(self) -> bool:
        return False

    def chunk_file_path(self, _chunk: Chunk) -> str | None:
        return None

    def chunk_index(self, _chunk: Chunk) -> int | None:
        return None


class FakeClient:
    def __init__(self, ns: FakeNamespace, embed_fn=None):
        self.ns = ns
        self.fields = ["content"]
        self.vector_field = "vector"
        self.embed_fn = embed_fn

    def build_bm25_rank_by(self, query: str):
        return ("content", "BM25", query)

    def build_vector_rank_by(self, vector: list[float]):
        return (self.vector_field, "ANN", vector)

    def build_native_hybrid_rank_by(self, query, vector, hybrid_opts):
        return [
            {"type": "bm25", "text": query, "fields": self.fields, "weight": 1.0},
            {"type": "vector", "vector": vector, "field": self.vector_field, "weight": 1.0},
        ]

    def row_to_chunk(self, row) -> Chunk:
        metadata = []
        for attr in ("file_path", "chunk_index"):
            val = getattr(row, attr, None)
            if val is not None:
                metadata.append((attr, val))
        metadata.append(("_tpuf_id", row.id))
        return Chunk(content=row.content, metadata=tuple(metadata))

    def row_content(self, row) -> str:
        return row.content


def _row(row_id: int, content: str, **attrs):
    dist = attrs.pop("dist", 0.0)
    ns = SimpleNamespace(id=row_id, content=content, **attrs)
    setattr(ns, "$dist", dist)
    return ns


def _make_source(
    ns: FakeNamespace,
    files=None,
    embed_fn=None,
) -> TpufChunkSource:
    source = TpufChunkSource.__new__(TpufChunkSource)
    source._client = FakeClient(ns=ns, embed_fn=embed_fn)
    source._files = files or NoFileFakeFiles()
    source._search_capabilities = {
        "backend": "turbopuffer",
        "modes": {"lexical"},
        "filter_ops": {
            "field": {"eq", "in", "gte", "lte"},
            "logical": {"and", "or", "not"},
        },
        "ranking": {"bm25"},
        "constraints": {"max_top_k": 10000, "vector_dimensions": None, "vector_field": "vector"},
        "graph_expansion": False,
    }
    return source


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkipSourceChunk:
    def test_skips_by_tpuf_id(self):
        ns = FakeNamespace(
            rows_per_call=[
                [_row(10, "source"), _row(20, "other")],
            ]
        )
        source = _make_source(ns)
        primary = Chunk(content="source", metadata=(("_tpuf_id", 10),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "other"

    def test_skips_by_content_match(self):
        ns = FakeNamespace(
            rows_per_call=[
                [_row(10, "same content"), _row(20, "different")],
            ]
        )
        source = _make_source(ns)
        primary = Chunk(content="same content", metadata=())
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "different"


class TestNeighborSkip:
    def test_skips_adjacent_same_file(self):
        ns = FakeNamespace(
            rows_per_call=[
                [
                    _row(1, "adj prev", file_path="a.md", chunk_index=0),
                    _row(2, "adj next", file_path="a.md", chunk_index=2),
                    _row(3, "far away", file_path="a.md", chunk_index=5),
                ],
            ]
        )
        source = _make_source(ns, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_tpuf_id", 99), ("file_path", "a.md"), ("chunk_index", 1)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "far away"

    def test_includes_non_adjacent_same_file(self):
        ns = FakeNamespace(
            rows_per_call=[
                [_row(5, "far chunk", file_path="a.md", chunk_index=10)],
            ]
        )
        source = _make_source(ns, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_tpuf_id", 99), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1


class TestDedup:
    def test_deduplicates_by_row_id(self):
        ns = FakeNamespace(
            rows_per_call=[
                [_row(1, "dup")],
                [_row(1, "dup")],
            ]
        )
        source = _make_source(ns)
        primary = Chunk(content="source", metadata=(("_tpuf_id", 99),))
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert len(results) == 1
        assert set(results[0]["queries"]) == {"q1", "q2"}


class TestSorting:
    def test_more_queries_sorts_higher(self):
        ns = FakeNamespace(
            rows_per_call=[
                [_row(1, "multi"), _row(2, "single")],
                [_row(1, "multi")],
            ]
        )
        source = _make_source(ns)
        primary = Chunk(content="source", metadata=(("_tpuf_id", 99),))
        results = source.search_related(primary, ["q1", "q2"], top_k=5)
        assert results[0]["chunk"].content == "multi"
        assert len(results[0]["queries"]) == 2

    def test_cross_file_sorts_above_same_file(self):
        ns = FakeNamespace(
            rows_per_call=[
                [
                    _row(1, "same file", file_path="a.md", chunk_index=5),
                    _row(2, "cross file", file_path="b.md", chunk_index=0),
                ],
            ]
        )
        source = _make_source(ns, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_tpuf_id", 99), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 2
        # Cross-file first (same query count, cross-file sorts higher)
        assert results[0]["same_file"] is False
        assert results[1]["same_file"] is True


class TestSameFileDetection:
    def test_same_file_flag(self):
        ns = FakeNamespace(
            rows_per_call=[
                [
                    _row(1, "same", file_path="a.md", chunk_index=5),
                    _row(2, "diff", file_path="b.md", chunk_index=0),
                ],
            ]
        )
        source = _make_source(ns, files=FileAwareFakeFiles())
        primary = Chunk(
            content="source",
            metadata=(("_tpuf_id", 99), ("file_path", "a.md"), ("chunk_index", 0)),
        )
        results = source.search_related(primary, ["query"], top_k=5)
        by_content = {r["chunk"].content: r for r in results}
        assert by_content["same"]["same_file"] is True
        assert by_content["diff"]["same_file"] is False


class TestEmptyQueries:
    def test_empty_queries_returns_empty(self):
        ns = FakeNamespace()
        source = _make_source(ns)
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, [], top_k=5)
        assert results == []
