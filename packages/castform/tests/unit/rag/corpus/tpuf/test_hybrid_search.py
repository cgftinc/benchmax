from __future__ import annotations

from types import SimpleNamespace

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.search_schema.search_types import SearchSpec
from castform.rag.corpus.turbopuffer.namespace import TpufNamespace
from castform.rag.corpus.turbopuffer.source import TpufChunkSource


class FakeNamespace:
    """Fake turbopuffer namespace that returns canned rows."""

    def __init__(self, *, lexical_rows=None, vector_rows=None):
        self.lexical_rows = list(lexical_rows or [])
        self.vector_rows = list(vector_rows or [])
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        rank_by = kwargs["rank_by"]
        if isinstance(rank_by, tuple) and len(rank_by) >= 2 and rank_by[1] == "ANN":
            return SimpleNamespace(rows=list(self.vector_rows))
        return SimpleNamespace(rows=list(self.lexical_rows))


class FakeFiles:
    def check(self) -> bool:
        return False

    def chunk_file_path(self, chunk: Chunk) -> str | None:
        return None

    def chunk_index(self, chunk: Chunk) -> int | None:
        return None


class FakeClient:
    def __init__(self, *, ns: FakeNamespace, vector_field: str = "vector", embed_fn=None):
        self.ns = ns
        self.fields = ["content"]
        self.vector_field = vector_field
        self.embed_fn = embed_fn

    def build_bm25_rank_by(self, query: str):
        return ("content", "BM25", query)

    def build_vector_rank_by(self, vector: list[float]):
        return (self.vector_field, "ANN", vector)

    def build_native_hybrid_rank_by(self, query: str, vector: list[float], hybrid_opts):
        return [
            {"type": "bm25", "text": query, "fields": self.fields, "weight": 1.0},
            {"type": "vector", "vector": vector, "field": self.vector_field, "weight": 1.0},
        ]

    def row_to_chunk(self, row) -> Chunk:
        return Chunk(content=row.content, metadata=(("_tpuf_id", row.id),))

    def row_content(self, row) -> str:
        return row.content


def _make_source(
    *, ns: FakeNamespace, embed_fn=None, vector_field: str = "vector"
) -> TpufChunkSource:
    """Build a TpufChunkSource with fake internals (no network)."""
    source = TpufChunkSource.__new__(TpufChunkSource)
    source._client = FakeClient(ns=ns, vector_field=vector_field, embed_fn=embed_fn)
    source._files = FakeFiles()

    modes = {"lexical"}
    ranking = {"bm25"}
    if embed_fn is not None:
        modes |= {"vector", "hybrid"}
        ranking |= {"cosine", "rrf"}

    source._search_capabilities = {
        "backend": "turbopuffer",
        "modes": modes,
        "filter_ops": {"field": {"eq", "in", "gte", "lte"}, "logical": {"and", "or", "not"}},
        "ranking": ranking,
        "constraints": {
            "max_top_k": 10000,
            "vector_dimensions": None,
            "vector_field": vector_field,
        },
        "graph_expansion": False,
    }
    return source


def _row(row_id: int, content: str):
    return SimpleNamespace(id=row_id, content=content)


# -- Namespace rank builders --------------------------------------------------


def test_namespace_rank_builders_respect_custom_vector_attr():
    ns = TpufNamespace.__new__(TpufNamespace)
    ns.fields = ["title", "body"]
    ns.vector_field = "embedding"

    assert ns.build_vector_rank_by([0.1, 0.2]) == ("embedding", "ANN", [0.1, 0.2])
    hybrid = ns.build_native_hybrid_rank_by("query", [0.1, 0.2], None)
    assert hybrid[1]["field"] == "embedding"


# -- Capabilities reporting ---------------------------------------------------


def test_capabilities_include_hybrid_when_embed_fn_present():
    source = _make_source(
        ns=FakeNamespace(),
        embed_fn=lambda texts: [[0.1] for _ in texts],
    )
    caps = source.get_search_capabilities()
    assert caps["modes"] == {"lexical", "vector", "hybrid"}
    assert "rrf" in caps["ranking"]


def test_capabilities_lexical_only_when_no_embed_fn():
    source = _make_source(ns=FakeNamespace())
    caps = source.get_search_capabilities()
    assert caps["modes"] == {"lexical"}
    assert caps["ranking"] == {"bm25"}


# -- search_related defaults to lexical ---------------------------------------


def test_search_related_defaults_to_lexical_even_with_embeddings():
    embed_calls: list[list[str]] = []

    def embed_fn(texts: list[str]) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    ns = FakeNamespace(lexical_rows=[_row(1, "alpha result")])
    source = _make_source(ns=ns, embed_fn=embed_fn)

    results = source.search_related(Chunk(content="seed"), ["alpha"], top_k=3)

    assert len(results) == 1
    assert ns.calls[0]["rank_by"] == ("content", "BM25", "alpha")
    # embed_fn should NOT be called — default mode is lexical
    assert embed_calls == []


# -- Explicit hybrid uses RRF fusion ------------------------------------------


def test_search_related_explicit_hybrid_uses_rrf_fusion():
    embed_calls: list[list[str]] = []

    def embed_fn(texts: list[str]) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[0.3, 0.7] for _ in texts]

    ns = FakeNamespace(
        lexical_rows=[_row(1, "lexical winner"), _row(3, "lexical only")],
        vector_rows=[_row(2, "vector winner"), _row(1, "lexical winner")],
    )
    source = _make_source(ns=ns, embed_fn=embed_fn, vector_field="embedding")

    results = source.search_related(Chunk(content="seed"), ["alpha"], top_k=2, mode="hybrid")

    assert [r["chunk"].content for r in results] == ["lexical winner", "vector winner"]
    assert embed_calls == [["alpha"]]
    # Two queries: one lexical, one vector
    assert len(ns.calls) == 2
    assert ns.calls[0]["rank_by"] == ("content", "BM25", "alpha")
    assert ns.calls[1]["rank_by"] == ("embedding", "ANN", [0.3, 0.7])


def test_search_content_hybrid_uses_rrf_fusion_with_custom_vector_field():
    ns = FakeNamespace(
        lexical_rows=[_row(1, "first"), _row(3, "third")],
        vector_rows=[_row(2, "second"), _row(1, "first")],
    )
    source = _make_source(
        ns=ns,
        embed_fn=lambda texts: [[0.2, 0.8] for _ in texts],
        vector_field="embedding",
    )

    result = source.search_content(
        SearchSpec(mode="hybrid", top_k=2, text_query="query", vector_query=[0.2, 0.8])
    )

    assert result == ["first", "second"]
    assert ns.calls[0]["rank_by"] == ("content", "BM25", "query")
    assert ns.calls[1]["rank_by"] == ("embedding", "ANN", [0.2, 0.8])


# -- Probe diagnostic tool ---------------------------------------------------


def test_probe_rank_by_payloads_without_embed_fn():
    source = _make_source(ns=FakeNamespace())
    probes = source.probe_rank_by_payloads("test")

    assert probes["lexical"]["accepted"] is True
    assert probes["vector"]["accepted"] is False
    assert probes["native_hybrid"]["accepted"] is False
    assert probes["vector"]["error"] == "embed_fn unavailable"


def test_probe_rank_by_payloads_with_embed_fn():
    source = _make_source(
        ns=FakeNamespace(),
        embed_fn=lambda texts: [[0.5, 0.25] for _ in texts],
    )
    probes = source.probe_rank_by_payloads("test")

    assert probes["lexical"]["accepted"] is True
    assert probes["vector"]["accepted"] is True
    # native_hybrid may or may not be accepted depending on the namespace;
    # with our FakeNamespace it goes through the lexical path
    assert probes["native_hybrid"]["accepted"] is True
