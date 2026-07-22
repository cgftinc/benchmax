"""Protocol shape conformance tests for ChunkSource."""

from __future__ import annotations

from types import SimpleNamespace

from fakes.neon import (
    FakeQueryRunner,
    FakeReadClient,
    constant_embed_fn,
    make_neon_source,
    make_query_row,
    make_read_row,
)

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.pinecone.source import PineconeChunkSource
from castform.rag.corpus.search_schema.search_types import SearchCapabilities, SearchSpec
from castform.rag.corpus.source import ChunkSource

# ---------------------------------------------------------------------------
# Minimal stub — returns canned data, implements protocol structurally
# ---------------------------------------------------------------------------


class StubChunkSource:
    """Minimal ChunkSource for shape testing. No real logic."""

    _CHUNK = Chunk(content="stub chunk", metadata=(("file", "stub.md"), ("index", 0)))

    def populate_from_folder(
        self,
        docs_path: str,
        min_chars: int = 1024,
        max_chars: int = 2048,
        overlap_chars: int = 128,
        file_extensions: list[str] | None = None,
        batch_size: int = 100,
        show_summary: bool = True,
    ) -> None:
        pass

    def populate_from_chunks(self, collection, batch_size=100, show_summary=True):
        pass

    def sample_chunks(self, n: int, min_chars: int = 0) -> list[Chunk]:
        return [self._CHUNK] * min(n, 1)

    def get_chunk_with_context(self, chunk: Chunk, max_chars: int = 200) -> dict:
        return {
            "chunk_content": chunk.chunk_str(),
            "prev_chunk_preview": "(No previous chunk)",
            "next_chunk_preview": "(No next chunk)",
        }

    def get_chunk_count(self) -> int:
        return 1

    def get_top_level_chunks(self) -> list[Chunk]:
        return [self._CHUNK]

    def search_related(
        self,
        source: Chunk,
        queries: list[str],
        top_k: int = 5,
        mode=None,
        hybrid=None,
    ) -> list[dict]:
        return (
            [
                {
                    "chunk": self._CHUNK,
                    "queries": queries[:1],
                    "same_file": False,
                    "max_score": 0.5,
                }
            ]
            if queries
            else []
        )

    def search(self, spec: SearchSpec) -> list[Chunk]:
        return [self._CHUNK]

    def search_content(self, spec: SearchSpec) -> list[str]:
        return ["stub content"]

    def embed_query(self, text: str) -> list[float] | None:
        return [0.1, 0.2, 0.3]

    def search_text(
        self,
        text_query: str,
        top_k: int = 10,
        filter=None,
    ) -> list[Chunk]:
        return [self._CHUNK]

    def get_search_capabilities(self) -> SearchCapabilities:
        return {
            "backend": "stub",
            "modes": {"lexical"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_stub = StubChunkSource()


class TestStubProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(_stub, ChunkSource)

    def test_capabilities_keys(self):
        caps = _stub.get_search_capabilities()
        assert "backend" in caps
        assert "modes" in caps
        assert "filter_ops" in caps
        assert "ranking" in caps
        assert "constraints" in caps
        assert "graph_expansion" in caps

    def test_search_related_shape(self):
        result = _stub.search_related(StubChunkSource._CHUNK, ["test query"], top_k=3)
        assert isinstance(result, list)
        for item in result:
            assert "chunk" in item
            assert "queries" in item
            assert "same_file" in item
            assert "max_score" in item
            assert isinstance(item["chunk"], Chunk)

    def test_get_chunk_with_context_shape(self):
        ctx = _stub.get_chunk_with_context(StubChunkSource._CHUNK)
        assert "chunk_content" in ctx
        assert "prev_chunk_preview" in ctx
        assert "next_chunk_preview" in ctx

    def test_sample_chunks_returns_chunks(self):
        result = _stub.sample_chunks(1)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_returns_chunks(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="test")
        result = _stub.search(spec)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_text_returns_chunks(self):
        result = _stub.search_text("test", top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_get_top_level_returns_chunks(self):
        result = _stub.get_top_level_chunks()
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_get_chunk_count_returns_int(self):
        result = _stub.get_chunk_count()
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# PineconeChunkSource protocol conformance (via fakes)
# ---------------------------------------------------------------------------


def _fake_pinecone_source() -> PineconeChunkSource:
    """Build a PineconeChunkSource with fakes injected, no real Pinecone."""

    class FakeIndex:
        def query(self, **kw):
            return SimpleNamespace(
                matches=[
                    SimpleNamespace(
                        id="1",
                        metadata={
                            "content": "pinecone chunk",
                            "file_path": "pc.md",
                            "chunk_index": 0,
                        },
                        score=0.9,
                    ),
                    SimpleNamespace(
                        id="2",
                        metadata={
                            "content": "second chunk",
                            "file_path": "pc2.md",
                            "chunk_index": 0,
                        },
                        score=0.8,
                    ),
                ]
            )

        def describe_index_stats(self):
            return SimpleNamespace(dimension=3, total_vector_count=2)

    class FakeFiles:
        def check(self):
            return True

        @staticmethod
        def chunk_file_path(chunk):
            return chunk.get_metadata("file_path") or None

        @staticmethod
        def chunk_index(chunk):
            val = chunk.get_metadata("chunk_index")
            return int(val) if val is not None else None

        def get_file_chunks(self, _path):
            return []

        def get_all_file_paths(self):
            return ["pc.md"]

        def invalidate(self):
            pass

    class FakeClient:
        def __init__(self):
            self._index = FakeIndex()
            self.embed_fn = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
            self._field_mapping = {"content": "content", "file_path": "file_path"}
            self._reverse_mapping = {"content": "content", "file_path": "file_path"}
            self._vector_dim = 3

        def query(self, **kw):
            return self._index.query(**kw)

        def _get_index(self):
            return self._index

        def _pc_field(self, name):
            return self._reverse_mapping.get(name, name)

        def _internal_field(self, name):
            return self._field_mapping.get(name, name)

        def zero_vector(self):
            return [0.0] * self._vector_dim

        def vector_type(self):
            return "dense"

        def namespace_vector_count(self):
            return 2

        def match_content(self, match):
            return str((getattr(match, "metadata", {}) or {}).get("content", ""))

        def match_to_raw(self, match):
            meta = getattr(match, "metadata", {}) or {}
            content = str(meta.get("content", ""))
            attrs = {k: v for k, v in meta.items() if k != "content"}
            attrs["_pinecone_id"] = match.id
            return {
                "id": match.id,
                "content": content,
                "metadata": attrs,
                "score": getattr(match, "score", 0.0) or 0.0,
            }

    source = PineconeChunkSource.__new__(PineconeChunkSource)
    source._client = FakeClient()
    source._files = FakeFiles()
    source._search_capabilities = {
        "backend": "pinecone",
        "modes": {"vector"},
        "filter_ops": {"field": {"eq", "in", "gte", "lte"}, "logical": {"and", "or", "not"}},
        "ranking": {"cosine"},
        "constraints": {"max_top_k": 10000, "vector_dimensions": None},
        "graph_expansion": False,
    }
    return source


_pc_source = _fake_pinecone_source()
_PC_CHUNK = Chunk(
    content="pinecone chunk",
    metadata=(("file_path", "pc.md"), ("chunk_index", 0), ("_pinecone_id", "src-99")),
)


class TestPineconeProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(_pc_source, ChunkSource)

    def test_capabilities_keys(self):
        caps = _pc_source.get_search_capabilities()
        assert "backend" in caps
        assert "modes" in caps
        assert "filter_ops" in caps
        assert "ranking" in caps
        assert "constraints" in caps
        assert "graph_expansion" in caps

    def test_capabilities_vector_only(self):
        caps = _pc_source.get_search_capabilities()
        assert caps["backend"] == "pinecone"
        assert "vector" in caps["modes"]
        assert "lexical" not in caps["modes"]

    def test_search_related_shape(self):
        result = _pc_source.search_related(_PC_CHUNK, ["test query"], top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0, "search_related should return results (not vacuously empty)"
        for item in result:
            assert "chunk" in item
            assert "queries" in item
            assert "same_file" in item
            assert "max_score" in item
            assert isinstance(item["chunk"], Chunk)

    def test_get_chunk_with_context_shape(self):
        ctx = _pc_source.get_chunk_with_context(_PC_CHUNK)
        assert "chunk_content" in ctx
        assert "prev_chunk_preview" in ctx
        assert "next_chunk_preview" in ctx

    def test_search_returns_chunks(self):
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2, 0.3])
        result = _pc_source.search(spec)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_content_returns_strings(self):
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2, 0.3])
        result = _pc_source.search_content(spec)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_search_text_returns_chunks(self):
        result = _pc_source.search_text("test", top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_embed_query_returns_vector(self):
        result = _pc_source.embed_query("test")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_get_chunk_count_returns_int(self):
        result = _pc_source.get_chunk_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_top_level_returns_chunks(self):
        result = _pc_source.get_top_level_chunks()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# NeonChunkSource protocol conformance (via fakes)
# ---------------------------------------------------------------------------


_neon_source = make_neon_source(
    embed_fn=constant_embed_fn(),
    search=FakeQueryRunner(
        rows=[
            make_query_row(
                "r1",
                "neon chunk",
                metadata={"file": "other.md", "index": 5},
                source_file="other.md",
                chunk_index=5,
            )
        ]
    ),
    read_client=FakeReadClient(
        count=2,
        sample_rows=[make_read_row("s1", "sample chunk")],
        top_level_rows=[make_read_row("t1", "top chunk")],
    ),
)
_NEON_CHUNK = Chunk(content="seed chunk", metadata=(("file", "n.md"), ("index", 0)))


class TestNeonProtocolConformance:
    def test_isinstance_check(self):
        assert isinstance(_neon_source, ChunkSource)

    def test_capabilities_keys(self):
        caps = _neon_source.get_search_capabilities()
        for key in ("backend", "modes", "filter_ops", "ranking", "constraints"):
            assert key in caps
        assert "graph_expansion" in caps

    def test_capabilities_hybrid_with_embed(self):
        caps = _neon_source.get_search_capabilities()
        assert caps["backend"] == "neon"
        assert {"lexical", "vector", "hybrid"} <= caps["modes"]

    def test_search_related_shape(self):
        result = _neon_source.search_related(_NEON_CHUNK, ["test query"], top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0, "search_related should return results (not vacuously empty)"
        for item in result:
            assert {"chunk", "queries", "same_file", "max_score"} <= set(item)
            assert isinstance(item["chunk"], Chunk)

    def test_get_chunk_with_context_shape(self):
        ctx = _neon_source.get_chunk_with_context(_NEON_CHUNK)
        assert "chunk_content" in ctx
        assert "prev_chunk_preview" in ctx
        assert "next_chunk_preview" in ctx

    def test_search_returns_chunks(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="test")
        result = _neon_source.search(spec)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_search_content_returns_strings(self):
        spec = SearchSpec(mode="lexical", top_k=5, text_query="test")
        result = _neon_source.search_content(spec)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_search_text_returns_chunks(self):
        result = _neon_source.search_text("test", top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_embed_query_returns_vector(self):
        result = _neon_source.embed_query("test")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_sample_chunks_returns_chunks(self):
        result = _neon_source.sample_chunks(1)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_get_chunk_count_returns_int(self):
        result = _neon_source.get_chunk_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_top_level_returns_chunks(self):
        result = _neon_source.get_top_level_chunks()
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
