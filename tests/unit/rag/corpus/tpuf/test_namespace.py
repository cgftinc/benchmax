"""Tests for TpufNamespace rank builders and row conversion."""

from __future__ import annotations

from types import SimpleNamespace

from benchmax.rag.corpus.turbopuffer.namespace import TpufNamespace

def _make_namespace(
    fields: list[str] | None = None,
    vector_field: str = "vector",
) -> TpufNamespace:
    """Build a TpufNamespace without network init."""
    ns = TpufNamespace.__new__(TpufNamespace)
    ns.fields = fields or ["content"]
    ns.vector_field = str(vector_field or "vector").strip() or "vector"
    ns.embed_fn = None
    return ns


# ---------------------------------------------------------------------------
# build_bm25_rank_by
# ---------------------------------------------------------------------------


class TestBuildBm25RankBy:
    def test_single_field(self):
        ns = _make_namespace(fields=["content"])
        result = ns.build_bm25_rank_by("hello world")
        assert result == ("content", "BM25", "hello world")

    def test_multi_field(self):
        ns = _make_namespace(fields=["title", "body"])
        result = ns.build_bm25_rank_by("query")
        assert result[0] == "Sum"
        inner = result[1]
        assert len(inner) == 2
        # Each inner element: ("Product", 1, (field, "BM25", query))
        assert inner[0] == ("Product", 1, ("title", "BM25", "query"))
        assert inner[1] == ("Product", 1, ("body", "BM25", "query"))

    def test_truncates_at_1024_code_points(self):
        ns = _make_namespace()
        long_query = "a" * 2000
        result = ns.build_bm25_rank_by(long_query)
        assert len(result[2]) == TpufNamespace._TPUF_BM25_MAX_QUERY_LEN


# ---------------------------------------------------------------------------
# build_vector_rank_by
# ---------------------------------------------------------------------------


class TestBuildVectorRankBy:
    def test_returns_ann_tuple(self):
        ns = _make_namespace()
        result = ns.build_vector_rank_by([0.1, 0.2, 0.3])
        assert result[1] == "ANN"
        assert result[2] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# build_hybrid_rank_by
# ---------------------------------------------------------------------------


class TestBuildNativeHybridRankBy:
    def test_default_weights(self):
        ns = _make_namespace(fields=["content"])
        result = ns.build_native_hybrid_rank_by("query", [0.1, 0.2], None)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "bm25"
        assert result[0]["weight"] == 1.0
        assert result[1]["type"] == "vector"
        assert result[1]["weight"] == 1.0

    def test_custom_weights(self):
        ns = _make_namespace()
        opts = {"lexical_weight": 0.3, "vector_weight": 0.7}
        result = ns.build_native_hybrid_rank_by("q", [0.1], opts)
        assert result[0]["weight"] == 0.3
        assert result[1]["weight"] == 0.7


# ---------------------------------------------------------------------------
# row_to_chunk
# ---------------------------------------------------------------------------


class TestRowToChunk:
    def test_basic_conversion(self):
        ns = _make_namespace(fields=["content"])
        row = SimpleNamespace(
            id=42,
            content="hello world",
            file_path="doc.md",
            chunk_index=3,
        )
        chunk = ns.row_to_chunk(row)
        assert chunk.content == "hello world"
        assert chunk.get_metadata("_tpuf_id") == 42
        assert chunk.get_metadata("file_path") == "doc.md"
        assert chunk.get_metadata("chunk_index") == 3

    def test_multi_field_content_json(self):
        ns = _make_namespace(fields=["title", "body"])
        row = SimpleNamespace(id=1, title="Title", body="Body text")
        chunk = ns.row_to_chunk(row)
        # Multi-field: content is JSON-encoded
        assert "Title" in chunk.content
        assert "Body text" in chunk.content
