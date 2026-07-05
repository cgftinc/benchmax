"""Tests for TpufChunkSource configuration variants.

Covers degraded/variant customer configurations:
- No embed_fn (lexical only) with vector mode requested
- No file metadata (pre-existing customer namespace)
- Multi-field content_attr
- Minimal row attributes (only id + content)
- search_text basic flow
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmax.rag.chunkers.models import Chunk
from benchmax.rag.corpus.search_schema.search_exceptions import (
    UnsupportedSearchModeError,
)
from benchmax.rag.corpus.search_schema.search_types import SearchSpec
from benchmax.rag.corpus.turbopuffer.source import TpufChunkSource

# ---------------------------------------------------------------------------
# Fakes — same pattern as test_tpuf_search_related.py
# ---------------------------------------------------------------------------


class FakeNamespace:
    def __init__(self, rows: list | None = None):
        self.rows = list(rows or [])

    def query(self, **kwargs):
        return SimpleNamespace(rows=list(self.rows))


class NoFileFakeFiles:
    def check(self) -> bool:
        return False

    def chunk_file_path(self, _chunk: Chunk) -> str | None:
        return None

    def chunk_index(self, _chunk: Chunk) -> int | None:
        return None

    def get_file_chunks(self, _path: str) -> list[Chunk]:
        return []

    def get_all_file_paths(self) -> list[str]:
        return []


class FakeClient:
    def __init__(self, ns: FakeNamespace, embed_fn=None, fields=None):
        self.ns = ns
        self.fields = fields or ["content"]
        self.vector_field = "vector"
        self.embed_fn = embed_fn

    def build_bm25_rank_by(self, query: str):
        if len(self.fields) == 1:
            return (self.fields[0], "BM25", query)
        return ("Sum", tuple(("Product", 1, (f, "BM25", query)) for f in self.fields))

    def build_vector_rank_by(self, vector: list[float]):
        return (self.vector_field, "ANN", vector)

    def build_native_hybrid_rank_by(self, query, vector, hybrid_opts):
        return [
            {"type": "bm25", "text": query, "fields": self.fields, "weight": 1.0},
            {"type": "vector", "vector": vector, "field": self.vector_field, "weight": 1.0},
        ]

    def row_to_chunk(self, row) -> Chunk:
        import json

        if len(self.fields) == 1:
            content = str(getattr(row, self.fields[0], ""))
        else:
            content = json.dumps({f: getattr(row, f, "") for f in self.fields}, default=str)
        try:
            raw = vars(row)
            attrs = {
                k: v
                for k, v in raw.items()
                if not k.startswith("$") and k not in ("id", self.vector_field)
            }
        except TypeError:
            attrs = {f: getattr(row, f, "") for f in self.fields}
        attrs["_tpuf_id"] = row.id
        return Chunk(content=content, metadata=tuple(attrs.items()))

    def row_content(self, row) -> str:
        import json

        if len(self.fields) == 1:
            return str(getattr(row, self.fields[0], ""))
        return json.dumps({f: getattr(row, f, "") for f in self.fields}, default=str)


def _row(row_id: int, content: str, **attrs):
    dist = attrs.pop("dist", 0.0)
    ns = SimpleNamespace(id=row_id, content=content, **attrs)
    setattr(ns, "$dist", dist)
    return ns


def _make_source(
    ns: FakeNamespace,
    files=None,
    embed_fn=None,
    fields=None,
) -> TpufChunkSource:
    source = TpufChunkSource.__new__(TpufChunkSource)
    source._client = FakeClient(ns=ns, embed_fn=embed_fn, fields=fields)
    source._files = files or NoFileFakeFiles()

    modes = {"lexical"}
    ranking = {"bm25"}
    if embed_fn is not None:
        modes |= {"vector", "hybrid"}
        ranking |= {"cosine", "rrf"}

    source._search_capabilities = {
        "backend": "turbopuffer",
        "modes": modes,
        "filter_ops": {
            "field": {"eq", "in", "gte", "lte"},
            "logical": {"and", "or", "not"},
        },
        "ranking": ranking,
        "constraints": {
            "max_top_k": 10000,
            "vector_dimensions": None,
            "vector_field": "vector",
        },
        "graph_expansion": False,
    }
    return source


# ---------------------------------------------------------------------------
# No embed_fn — vector/hybrid mode should fail cleanly
# ---------------------------------------------------------------------------


class TestNoEmbedFnModeValidation:
    def test_search_related_vector_mode_raises(self):
        ns = FakeNamespace(rows=[_row(1, "result")])
        source = _make_source(ns, embed_fn=None)
        primary = Chunk(content="seed", metadata=())
        with pytest.raises(UnsupportedSearchModeError):
            source.search_related(primary, ["query"], top_k=5, mode="vector")

    def test_search_related_hybrid_mode_raises(self):
        ns = FakeNamespace(rows=[_row(1, "result")])
        source = _make_source(ns, embed_fn=None)
        primary = Chunk(content="seed", metadata=())
        with pytest.raises(UnsupportedSearchModeError):
            source.search_related(primary, ["query"], top_k=5, mode="hybrid")

    def test_search_vector_spec_raises(self):
        ns = FakeNamespace()
        source = _make_source(ns, embed_fn=None)
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2])
        with pytest.raises(UnsupportedSearchModeError):
            source.search(spec)

    def test_search_related_lexical_still_works(self):
        ns = FakeNamespace(rows=[_row(1, "lexical result")])
        source = _make_source(ns, embed_fn=None)
        primary = Chunk(content="seed", metadata=(("_tpuf_id", 99),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "lexical result"


# ---------------------------------------------------------------------------
# No file metadata — graceful degradation
# ---------------------------------------------------------------------------


class TestNoFileMetadata:
    def test_search_related_no_neighbor_skip(self):
        """Without file metadata, adjacent chunks are NOT skipped."""
        ns = FakeNamespace(
            rows=[
                _row(1, "adjacent chunk"),
                _row(2, "another chunk"),
            ]
        )
        source = _make_source(ns, files=NoFileFakeFiles())
        primary = Chunk(content="seed", metadata=(("_tpuf_id", 99),))
        results = source.search_related(primary, ["query"], top_k=5)
        # Both should appear — no neighbor skipping without file awareness
        assert len(results) == 2

    def test_search_related_same_file_always_false(self):
        """Without file metadata, same_file is always False."""
        ns = FakeNamespace(rows=[_row(1, "result")])
        source = _make_source(ns, files=NoFileFakeFiles())
        primary = Chunk(content="seed", metadata=(("_tpuf_id", 99),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert results[0]["same_file"] is False

    def test_get_chunk_with_context_returns_fallback(self):
        """Without file metadata, context returns empty strings."""
        ns = FakeNamespace()
        source = _make_source(ns, files=NoFileFakeFiles())
        chunk = Chunk(content="some content", metadata=())
        ctx = source.get_chunk_with_context(chunk)
        assert "chunk_content" in ctx
        assert ctx["prev_chunk_preview"] == ""
        assert ctx["next_chunk_preview"] == ""

    def test_get_top_level_chunks_returns_empty(self):
        """Without file metadata, top-level returns empty."""
        ns = FakeNamespace()
        source = _make_source(ns, files=NoFileFakeFiles())
        assert source.get_top_level_chunks() == []


# ---------------------------------------------------------------------------
# Minimal row attributes (customer's pre-existing data)
# ---------------------------------------------------------------------------


class TestMinimalRowAttributes:
    def test_row_to_chunk_only_id_and_content(self):
        """Customer row has only id + content, no file metadata."""
        ns = FakeNamespace()
        source = _make_source(ns)
        row = SimpleNamespace(id=42, content="just text")
        chunk = source._client.row_to_chunk(row)
        assert chunk.content == "just text"
        assert chunk.get_metadata("_tpuf_id") == 42
        assert chunk.get_metadata("file_path") is None
        assert chunk.get_metadata("chunk_index") is None

    def test_row_to_chunk_extra_custom_fields(self):
        """Customer row has extra fields not in standard schema."""
        ns = FakeNamespace()
        source = _make_source(ns)
        row = SimpleNamespace(id=1, content="text", custom_tag="important", priority=5)
        chunk = source._client.row_to_chunk(row)
        assert chunk.get_metadata("custom_tag") == "important"
        assert chunk.get_metadata("priority") == 5


# ---------------------------------------------------------------------------
# Multi-field content_attr
# ---------------------------------------------------------------------------


class TestMultiFieldContent:
    def test_row_content_multi_field_is_json(self):
        ns = FakeNamespace()
        source = _make_source(ns, fields=["title", "body"])
        row = SimpleNamespace(id=1, title="My Title", body="My Body")
        content = source._client.row_content(row)
        assert "My Title" in content
        assert "My Body" in content

    def test_row_to_chunk_multi_field(self):
        ns = FakeNamespace()
        source = _make_source(ns, fields=["title", "body"])
        row = SimpleNamespace(id=1, title="Title", body="Body")
        chunk = source._client.row_to_chunk(row)
        assert "Title" in chunk.content
        assert "Body" in chunk.content


# ---------------------------------------------------------------------------
# search_text basic flow
# ---------------------------------------------------------------------------


class TestSearchTextFlow:
    def test_delegates_to_lexical_search(self):
        ns = FakeNamespace(rows=[_row(1, "found it")])
        source = _make_source(ns)
        results = source.search_text("find me", top_k=5)
        assert len(results) == 1
        assert results[0].content == "found it"

    def test_search_content_returns_strings(self):
        ns = FakeNamespace(rows=[_row(1, "content text")])
        source = _make_source(ns)
        spec = SearchSpec(mode="lexical", top_k=5, text_query="query")
        results = source.search_content(spec)
        assert results == ["content text"]
        assert all(isinstance(r, str) for r in results)
