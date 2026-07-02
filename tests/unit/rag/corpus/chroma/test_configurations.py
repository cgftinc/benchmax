"""Tests for ChromaChunkSource configuration variants.

Covers degraded/variant customer configurations:
- Vector-only mode (no Search API / no BM25)
- No file metadata (pre-existing customer collection)
- Custom embed_fn with dimension validation
- search_text / search_content flows (vector, hybrid, lexical branches)
- Capabilities reporting
- Pickle roundtrip
- BM25 downgrade error path
- content_attr behavior
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fakes.chroma import (
    FakeCollection,
    NoFileFakeFiles,
    make_query_result,
    make_source,
)

from benchmax.rag.chunkers.models import Chunk
from benchmax.rag.corpus.chroma.client import BM25_KEY
from benchmax.rag.corpus.search_schema.search_exceptions import (
    LocalEmbeddingDownloadDisallowedError,
    UnsupportedSearchModeError,
)
from benchmax.rag.corpus.search_schema.search_types import SearchSpec


def _fake_schema(*, has_bm25: bool, enabled: bool = True):
    """Build a stand-in for chromadb's Schema object.

    Mirrors the attribute chain ChromaClient._has_bm25_index walks:
    schema.keys[BM25_KEY].sparse_vector.sparse_vector_index.{enabled,config}.
    """
    if not has_bm25:
        return SimpleNamespace(keys={})
    index = SimpleNamespace(
        enabled=enabled,
        config=SimpleNamespace(embedding_function=object()),
    )
    value_type = SimpleNamespace(
        sparse_vector=SimpleNamespace(sparse_vector_index=index)
    )
    return SimpleNamespace(keys={BM25_KEY: value_type})

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_always_has_vector_mode(self):
        col = FakeCollection(count=0)
        source = make_source(col)
        caps = source.get_search_capabilities()
        assert "vector" in caps["modes"]
        assert caps["backend"] == "chroma"

    def test_vector_only_without_search_api(self):
        col = FakeCollection(count=0)
        source = make_source(col)
        caps = source.get_search_capabilities()
        assert caps["modes"] == {"vector"}

    def test_lexical_hybrid_available_on_server_with_search_api(self):
        """When Search API is available AND host is set, lexical+hybrid appear."""
        from benchmax.rag.corpus.chroma.client import ChromaClient
        from benchmax.rag.corpus.chroma.source import ChromaChunkSource

        with patch("benchmax.rag.corpus.chroma.client.has_search_api", return_value=True):
            chroma = ChromaClient(
                collection_name="t",
                host="h",
                enable_bm25=True,
            )
        chroma._collection = MagicMock()  # pre-set to avoid connection

        src = ChromaChunkSource.__new__(ChromaChunkSource)
        src._chroma = chroma
        src._files = NoFileFakeFiles()
        src._search_capabilities = {
            "backend": "chroma",
            "modes": {"vector", "lexical", "hybrid"},
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": {"cosine", "bm25"},
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
            "graph_expansion": False,
        }
        caps = src.get_search_capabilities()
        assert "lexical" in caps["modes"]
        assert "hybrid" in caps["modes"]
        assert "bm25" in caps["ranking"]


# ---------------------------------------------------------------------------
# Mode rejection
# ---------------------------------------------------------------------------


class TestModeRejection:
    def test_lexical_mode_rejected_on_vector_only(self):
        col = FakeCollection(count=0)
        source = make_source(col)
        with pytest.raises(UnsupportedSearchModeError):
            source.search(SearchSpec(mode="lexical", top_k=5, text_query="test"))

    def test_hybrid_mode_rejected_on_vector_only(self):
        col = FakeCollection(count=0)
        source = make_source(col)
        with pytest.raises(UnsupportedSearchModeError):
            source.search(
                SearchSpec(
                    mode="hybrid",
                    top_k=5,
                    text_query="test",
                    vector_query=[0.1, 0.2],
                )
            )


# ---------------------------------------------------------------------------
# No file metadata — graceful degradation
# ---------------------------------------------------------------------------


class TestNoFileMetadata:
    def test_get_chunk_with_context_returns_fallback(self):
        col = FakeCollection(count=0)
        source = make_source(col, files=NoFileFakeFiles())
        chunk = Chunk(content="some content", metadata=())
        ctx = source.get_chunk_with_context(chunk)
        assert "chunk_content" in ctx
        assert ctx["prev_chunk_preview"] == ""
        assert ctx["next_chunk_preview"] == ""

    def test_get_top_level_chunks_returns_empty(self):
        col = FakeCollection(count=0)
        source = make_source(col, files=NoFileFakeFiles())
        assert source.get_top_level_chunks() == []

    def test_search_related_no_neighbor_skip(self):
        """Without file metadata, adjacent chunks are NOT skipped."""
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(
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
        assert len(results) == 2

    def test_search_related_same_file_always_false(self):
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(["result"]),
            ],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(primary, ["query"], top_k=5)
        assert results[0]["same_file"] is False


# ---------------------------------------------------------------------------
# search_text / search_content flows
# ---------------------------------------------------------------------------


class TestSearchTextFlow:
    def test_delegates_to_vector_search_and_passes_correct_kwargs(self):
        """Verify query() receives the right kwargs for vector search."""
        col = FakeCollection(
            query_results_per_call=[make_query_result(["found it"])],
            count=5,
        )
        source = make_source(col)
        results = source.search_text("find me", top_k=5)
        assert len(results) == 1
        assert results[0].content == "found it"
        # Verify the actual kwargs passed to query()
        call_kwargs = col._last_query_kwargs
        assert call_kwargs["n_results"] == 5
        assert call_kwargs["query_texts"] == ["find me"]

    def test_search_content_returns_strings(self):
        col = FakeCollection(
            query_results_per_call=[make_query_result(["content text"])],
            count=5,
        )
        source = make_source(col)
        spec = SearchSpec(mode="vector", top_k=5, text_query="query")
        results = source.search_content(spec)
        assert results == ["content text"]
        assert all(isinstance(r, str) for r in results)

    def test_search_text_with_embed_fn_passes_vector(self):
        """When embed_fn is provided, search_text passes vector_query."""
        embed_fn = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        col = FakeCollection(
            query_results_per_call=[make_query_result(["result"])],
            count=5,
        )
        source = make_source(col, embed_fn=embed_fn)
        source.search_text("query", top_k=3)
        embed_fn.assert_called_once_with(["query"])
        # Verify vector was passed to query()
        call_kwargs = col._last_query_kwargs
        assert call_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]

    def test_search_text_prefers_hybrid_when_available(self):
        """search_text picks hybrid mode when Search API + BM25 available."""
        col = FakeCollection(count=5)
        source = make_source(col)
        # Enable hybrid/lexical. _chroma.modes is the source of truth — search_text
        # re-syncs capabilities from it after lazy collection init.
        source._chroma.modes = {"vector", "lexical", "hybrid"}
        source._search_capabilities["modes"] = {"vector", "lexical", "hybrid"}
        source._chroma.search_api = True

        mock_result = MagicMock()
        mock_result.rows.return_value = [[{"document": "hyb", "metadata": {}, "score": 0.9}]]
        source._chroma._collection = MagicMock()
        source._chroma._collection.search = MagicMock(return_value=mock_result)

        results = source.search_text("query", top_k=3)
        assert len(results) == 1
        assert results[0].content == "hyb"
        source._chroma._collection.search.assert_called_once()

    def test_search_text_falls_back_to_lexical_without_hybrid(self):
        """search_text picks lexical when hybrid is unavailable."""
        col = FakeCollection(count=5)
        source = make_source(col)
        source._chroma.modes = {"vector", "lexical"}
        source._search_capabilities["modes"] = {"vector", "lexical"}
        source._chroma.search_api = True

        mock_result = MagicMock()
        mock_result.rows.return_value = [[{"document": "lex", "metadata": {}, "score": 0.8}]]
        source._chroma._collection = MagicMock()
        source._chroma._collection.search = MagicMock(return_value=mock_result)

        results = source.search_text("query", top_k=3)
        assert len(results) == 1
        assert results[0].content == "lex"
        source._chroma._collection.search.assert_called_once()


# ---------------------------------------------------------------------------
# embed_query / dimension validation
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_returns_none_without_embed_fn(self):
        col = FakeCollection(count=0)
        source = make_source(col)
        assert source.embed_query("hello") is None

    def test_returns_vector_with_embed_fn(self):
        embed_fn = MagicMock(return_value=[[1.0, 2.0, 3.0]])
        col = FakeCollection(count=0)
        source = make_source(col, embed_fn=embed_fn)
        result = source.embed_query("hello")
        assert result == [1.0, 2.0, 3.0]

    def test_dimension_validation_consistent(self):
        embed_fn = MagicMock(return_value=[[1.0, 2.0, 3.0]])
        col = FakeCollection(count=0)
        source = make_source(col, embed_fn=embed_fn)
        source.embed_query("first")
        source.embed_query("second")  # same dim

    def test_dimension_validation_mismatch(self):
        call_count = 0

        def varying_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [[1.0, 2.0, 3.0]]
            return [[1.0, 2.0]]  # different dim!

        col = FakeCollection(count=0)
        source = make_source(col, embed_fn=varying_embed)
        source.embed_query("first")
        with pytest.raises(ValueError, match="dimension mismatch"):
            source.embed_query("second")


# ---------------------------------------------------------------------------
# Pickle roundtrip
# ---------------------------------------------------------------------------


class TestPickle:
    def test_getstate_strips_files(self):
        col = FakeCollection(count=5)
        source = make_source(col)
        state = source.__getstate__()
        assert state["_files"] is None
        # ChromaClient's own pickle strips _raw_client and _collection
        chroma_state = state["_chroma"].__getstate__()
        assert chroma_state["_raw_client"] is None
        assert chroma_state["_collection"] is None

    def test_setstate_restores_files(self):
        col = FakeCollection(count=5)
        source = make_source(col)
        state = source.__getstate__()
        from benchmax.rag.corpus.chroma.source import ChromaChunkSource

        new_src = ChromaChunkSource.__new__(ChromaChunkSource)
        new_src.__setstate__(state)
        assert new_src._files is not None

    def test_setstate_preserves_config(self):
        col = FakeCollection(count=5)
        source = make_source(col)
        state = source.__getstate__()
        from benchmax.rag.corpus.chroma.source import ChromaChunkSource

        new_src = ChromaChunkSource.__new__(ChromaChunkSource)
        new_src.__setstate__(state)
        assert new_src._chroma.collection_name == "test"
        assert new_src._chroma.host == "localhost"
        assert new_src._chroma.enable_bm25 is False


# ---------------------------------------------------------------------------
# Chroma-specific: auto-embed relaxation
# ---------------------------------------------------------------------------


class TestAutoEmbedRelaxation:
    def test_vector_mode_with_text_query_only(self):
        """Chroma auto-embeds, so vector mode with text_query is valid."""
        col = FakeCollection(
            query_results_per_call=[make_query_result(["result"])],
            count=5,
        )
        source = make_source(col)
        results = source.search(SearchSpec(mode="vector", top_k=3, text_query="hello"))
        assert len(results) == 1
        # Verify query_texts was passed (not query_embeddings)
        call_kwargs = col._last_query_kwargs
        assert call_kwargs["query_texts"] == ["hello"]
        assert "query_embeddings" not in call_kwargs


# ---------------------------------------------------------------------------
# BM25 downgrade error path
# ---------------------------------------------------------------------------


class TestBm25Downgrade:
    def test_schema_failure_downgrades_to_vector_only(self):
        """When schema creation fails, capabilities downgrade to vector-only."""
        from benchmax.rag.corpus.chroma.client import ChromaClient
        from benchmax.rag.corpus.chroma.source import ChromaChunkSource

        with patch("benchmax.rag.corpus.chroma.client.has_search_api", return_value=True):
            chroma = ChromaClient(
                collection_name="t",
                host="h",
                enable_bm25=True,
            )

        # Mock client that raises on schema-based create, succeeds without
        mock_client = MagicMock()
        mock_collection = MagicMock()

        def fake_get_or_create(**kwargs):
            if "schema" in kwargs:
                raise ValueError("Sparse vector indexing is not enabled in local")
            return mock_collection

        mock_client.get_or_create_collection = MagicMock(side_effect=fake_get_or_create)
        chroma._raw_client = mock_client

        src = ChromaChunkSource.__new__(ChromaChunkSource)
        src._chroma = chroma
        src._files = NoFileFakeFiles()
        src._search_capabilities = {
            "backend": "chroma",
            "modes": {"vector", "lexical", "hybrid"},
            "filter_ops": {
                "field": {"eq", "in", "gte", "lte"},
                "logical": {"and", "or", "not"},
            },
            "ranking": {"cosine", "bm25"},
            "constraints": {"max_top_k": 10000, "vector_dimensions": None},
            "graph_expansion": False,
        }

        # Trigger collection creation via get_search_capabilities
        # which forces lazy init when lexical is in modes
        caps = src.get_search_capabilities()

        # ChromaClient should have downgraded its own modes
        assert chroma.modes == {"vector"}
        assert chroma.ranking == {"cosine"}

        # Source capabilities should be synced
        assert caps["modes"] == {"vector"}
        assert caps["ranking"] == {"cosine"}

        # get_or_create_collection called twice: once with schema (failed),
        # once without (succeeded)
        assert mock_client.get_or_create_collection.call_count == 2

    def test_existing_collection_without_bm25_index_downgrades(self):
        """A pre-existing collection (no BM25 index) downgrades to vector-only.

        get_or_create_collection returns an existing collection as-is, so the
        schema-based create "succeeds" without applying our BM25 index. The
        index-readiness probe must catch that and drop lexical/hybrid, or a
        later query raises "key not found in schema".
        """
        from benchmax.rag.corpus.chroma.client import BM25_KEY, ChromaClient

        with patch("benchmax.rag.corpus.chroma.client.has_search_api", return_value=True):
            chroma = ChromaClient(collection_name="t", host="h", enable_bm25=True)
        assert chroma.modes == {"vector", "lexical", "hybrid"}

        existing = SimpleNamespace(schema=_fake_schema(has_bm25=False))
        mock_client = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=existing)
        chroma._raw_client = mock_client

        with patch.object(ChromaClient, "_build_schema", return_value={"sentinel": 1}):
            result = chroma.get_collection()

        assert result is existing
        assert chroma.modes == {"vector"}
        assert chroma.ranking == {"cosine"}
        # Schema branch "succeeded" -> no fallback create.
        assert mock_client.get_or_create_collection.call_count == 1
        assert BM25_KEY not in existing.schema.keys

    def test_collection_with_bm25_index_keeps_lexical_hybrid(self):
        """A collection that actually has the BM25 index keeps lexical+hybrid."""
        from benchmax.rag.corpus.chroma.client import ChromaClient

        with patch("benchmax.rag.corpus.chroma.client.has_search_api", return_value=True):
            chroma = ChromaClient(collection_name="t", host="h", enable_bm25=True)

        indexed = SimpleNamespace(schema=_fake_schema(has_bm25=True))
        mock_client = MagicMock()
        mock_client.get_or_create_collection = MagicMock(return_value=indexed)
        chroma._raw_client = mock_client

        with patch.object(ChromaClient, "_build_schema", return_value={"sentinel": 1}):
            chroma.get_collection()

        assert chroma.modes == {"vector", "lexical", "hybrid"}
        assert chroma.ranking == {"cosine", "bm25"}

    def test_index_probe_treats_missing_schema_as_not_ready(self):
        """No schema attr / None schema -> downgrade (vector is always safe)."""
        from benchmax.rag.corpus.chroma.client import ChromaClient

        assert ChromaClient._has_bm25_index(SimpleNamespace(schema=None)) is False
        assert ChromaClient._has_bm25_index(object()) is False  # no .schema attr
        # Key present but sparse index disabled -> not usable.
        assert (
            ChromaClient._has_bm25_index(
                SimpleNamespace(schema=_fake_schema(has_bm25=True, enabled=False))
            )
            is False
        )


# ---------------------------------------------------------------------------
# Chroma Cloud hosted embedding-function repair
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, cfg):
        self.configuration_json = cfg


class _FakeCloudCollection:
    """Minimal stand-in for a chromadb Collection's EF-relevant internals."""

    def __init__(self, cfg, ef=None):
        self._model = _FakeModel(cfg)
        self._embedding_function = ef


_QWEN_CFG = {
    "embedding_function": {
        "name": "chroma-cloud-qwen",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "task": None,
    }
}

_QWEN_EF_PATH = (
    "chromadb.utils.embedding_functions."
    "chroma_cloud_qwen_embedding_function.ChromaCloudQwenEmbeddingFunction"
)


class TestCloudQwenEmbeddingRepair:
    """chromadb's build_from_config rejects chroma-cloud-qwen's task=None config
    (through >=1.5.9), breaking every text query. _repair_cloud_embedding_function
    attaches a directly-built EF so _embed uses it instead of the broken loader.
    """

    def test_attaches_ef_for_cloud_qwen_config(self):
        from benchmax.rag.corpus.chroma.client import ChromaClient

        col = _FakeCloudCollection(_QWEN_CFG)
        sentinel = object()
        with patch(_QWEN_EF_PATH, return_value=sentinel) as ef_cls:
            ChromaClient._repair_cloud_embedding_function(col)
        assert col._embedding_function is sentinel
        # task=None must be forwarded (the value chromadb chokes on).
        assert ef_cls.call_args.kwargs["task"] is None

    def test_repairs_over_default_embedding_function(self):
        """A DefaultEmbeddingFunction means 'unresolved' — chromadb ignores it."""
        from benchmax.rag.corpus.chroma.client import ChromaClient

        class DefaultEmbeddingFunction:  # name is what the guard checks
            pass

        col = _FakeCloudCollection(_QWEN_CFG, ef=DefaultEmbeddingFunction())
        sentinel = object()
        with patch(_QWEN_EF_PATH, return_value=sentinel):
            ChromaClient._repair_cloud_embedding_function(col)
        assert col._embedding_function is sentinel

    def test_leaves_real_embedding_function_untouched(self):
        from benchmax.rag.corpus.chroma.client import ChromaClient

        real_ef = object()  # type name != DefaultEmbeddingFunction
        col = _FakeCloudCollection(_QWEN_CFG, ef=real_ef)
        with patch(_QWEN_EF_PATH, return_value=object()):
            ChromaClient._repair_cloud_embedding_function(col)
        assert col._embedding_function is real_ef

    def test_ignores_non_cloud_qwen_config(self):
        from benchmax.rag.corpus.chroma.client import ChromaClient

        col = _FakeCloudCollection({"embedding_function": {"name": "openai"}})
        ChromaClient._repair_cloud_embedding_function(col)
        assert col._embedding_function is None

    def test_guarded_against_broken_internals(self):
        """Any deviation in chromadb internals is a no-op, never a crash."""
        from benchmax.rag.corpus.chroma.client import ChromaClient

        class _Boom:
            @property
            def configuration_json(self):
                raise RuntimeError("internals changed")

        col = _FakeCloudCollection({})
        col._model = _Boom()
        ChromaClient._repair_cloud_embedding_function(col)  # must not raise
        assert col._embedding_function is None


# ---------------------------------------------------------------------------
# search_related / search_text honor and clamp the requested mode
# ---------------------------------------------------------------------------


def _set_ef_name(source, name):
    """Set the fake collection's configured embedding-function name.

    ``name=None`` means a collection with no embedding function at all.
    """
    cfg = {"embedding_function": {"name": name}} if name is not None else {}
    source._chroma._collection._model = SimpleNamespace(configuration_json=cfg)


class TestDenseSafety:
    """dense_embed_is_safe() — when a vector query won't download a model."""

    def _chroma(self, *, embed_fn=None, ef_name="missing"):
        source = make_source(FakeCollection(count=0), embed_fn=embed_fn)
        if ef_name != "missing":
            _set_ef_name(source, ef_name)
        return source._chroma

    def test_true_with_client_embed_fn(self):
        from unittest.mock import MagicMock

        assert self._chroma(embed_fn=MagicMock()).dense_embed_is_safe() is True

    def test_true_for_hosted_server_side_ef(self):
        assert self._chroma(ef_name="chroma-cloud-qwen").dense_embed_is_safe() is True

    def test_false_for_default_ef(self):
        # all-MiniLM, client-side download.
        assert self._chroma(ef_name="default").dense_embed_is_safe() is False

    def test_false_for_third_party_api_ef(self):
        # No model download, but we lack the provider key -> treat as unsafe.
        assert self._chroma(ef_name="openai").dense_embed_is_safe() is False

    def test_false_for_no_embedding_function(self):
        assert self._chroma(ef_name=None).dense_embed_is_safe() is False

    def test_false_when_collection_uninitialized(self):
        chroma = self._chroma()
        chroma._collection = None
        assert chroma.dense_embed_is_safe() is False


class TestSearchModeClamp:
    def test_vector_uses_vector_when_dense_is_safe(self):
        """mode='vector' uses the query() vector path when dense is safe.

        A server-side hosted embedding function (default fake = chroma-cloud-qwen)
        means a dense embed never downloads a model, so vector is honored.
        """
        col = FakeCollection(
            query_results_per_call=[make_query_result(["v"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        source._chroma.modes = {"vector", "lexical", "hybrid"}
        source._chroma.search_api = True

        primary = Chunk(content="src", metadata=())
        results = source.search_related(primary, ["q"], top_k=5, mode="vector")
        assert results[0]["chunk"].content == "v"
        # Vector path went through the legacy query() API, not collection.search().
        assert col._last_query_kwargs["query_texts"] == ["q"]

    def test_unsafe_dense_degrades_to_lexical_when_bm25_present(self):
        """Unsafe dense (default EF, no embed_fn) + BM25 -> lexical, no download.

        Covers the e2e collection: the linker's "inference" mode requests vector,
        but honoring it would download all-MiniLM. With a BM25 index present we
        use that instead — for every requested mode.
        """
        source = make_source(FakeCollection(count=5), files=NoFileFakeFiles())
        source._chroma.modes = {"vector", "lexical", "hybrid"}
        source._chroma.search_api = True
        _set_ef_name(source, "default")  # client-side all-MiniLM -> unsafe
        assert source._chroma.dense_embed_is_safe() is False

        captured: list[str | None] = []
        source._search_with_scores = lambda spec: captured.append(spec.get("mode")) or []  # type: ignore[method-assign,return-value]
        for requested in ("vector", "hybrid", None):
            captured.clear()
            source.search_related(
                Chunk(content="src", metadata=()), ["q"], top_k=3, mode=requested
            )
            assert captured == ["lexical"], requested

    def test_unsafe_dense_without_bm25_raises(self):
        """Unsafe dense + no BM25 index -> error, never a model download."""
        source = make_source(FakeCollection(count=5), files=NoFileFakeFiles())
        source._chroma.modes = {"vector"}  # vector-only, no lexical
        source._chroma.search_api = True
        _set_ef_name(source, "default")  # unsafe

        with pytest.raises(LocalEmbeddingDownloadDisallowedError):
            source.search_related(
                Chunk(content="src", metadata=()), ["q"], top_k=3, mode="vector"
            )

    def test_vector_stays_vector_when_safe_and_no_lexical(self):
        """Safe dense (hosted EF) + vector-only collection -> vector."""
        col = FakeCollection(
            query_results_per_call=[make_query_result(["v"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        source._chroma.modes = {"vector"}  # no lexical; default fake EF is hosted
        source._chroma.search_api = True

        captured: list[str | None] = []
        orig = source._search_with_scores
        source._search_with_scores = lambda spec: (captured.append(spec.get("mode")), orig(spec))[1]  # type: ignore[method-assign]
        source.search_related(
            Chunk(content="src", metadata=()), ["q"], top_k=3, mode="vector"
        )
        assert captured == ["vector"]

    def test_downgraded_modes_clamp_stale_hybrid_request_to_vector(self):
        """A stale mode='hybrid' is clamped to vector when the index is gone.

        Mirrors the linker passing best_search_mode='hybrid' against a
        collection whose capabilities were downgraded to vector-only.
        """
        col = FakeCollection(
            query_results_per_call=[make_query_result(["v"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        source._chroma.modes = {"vector"}  # downgraded
        source._chroma.search_api = True

        primary = Chunk(content="src", metadata=())
        results = source.search_related(primary, ["q"], top_k=5, mode="hybrid")
        assert results[0]["chunk"].content == "v"

    def test_hybrid_without_embed_fn_degrades_to_lexical_not_vector(self):
        """mode='hybrid' + no client embed_fn must run LEXICAL, not vector.

        Remote collections have no embed_fn, so dense query vectors can't be
        produced. Hybrid must degrade to its sparse/lexical leg (no embedding),
        not fall through to vector search — which would force chromadb to embed
        every query (slow; pulls the all-MiniLM model on a default-EF collection).
        """
        col = FakeCollection(count=5)
        source = make_source(col, files=NoFileFakeFiles())  # embed_fn=None
        source._chroma.modes = {"vector", "lexical", "hybrid"}
        source._chroma.search_api = True

        captured: list[str | None] = []

        def _capture(spec):
            captured.append(spec.get("mode"))
            return []

        source._search_with_scores = _capture  # type: ignore[method-assign]
        source.search_related(
            Chunk(content="src", metadata=()), ["q"], top_k=3, mode="hybrid"
        )
        assert captured == ["lexical"]

    def test_search_related_refreshes_modes_from_client(self):
        """search_related re-syncs _search_capabilities from _chroma.modes.

        A source whose capabilities still advertise hybrid (frozen at
        construction) but whose client was downgraded must not attempt hybrid.
        """
        col = FakeCollection(
            query_results_per_call=[make_query_result(["v"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        # Stale capabilities say hybrid; client is the source of truth (vector).
        source._search_capabilities["modes"] = {"vector", "lexical", "hybrid"}
        source._chroma.modes = {"vector"}
        source._chroma.search_api = True

        primary = Chunk(content="src", metadata=())
        results = source.search_related(primary, ["q"], top_k=5)
        assert results[0]["chunk"].content == "v"
        assert source._search_capabilities["modes"] == {"vector"}


# ---------------------------------------------------------------------------
# search_related accepts mode/hybrid kwargs
# ---------------------------------------------------------------------------


class TestSearchRelatedProtocol:
    def test_accepts_mode_kwarg(self):
        """search_related must accept mode= without TypeError."""
        col = FakeCollection(
            query_results_per_call=[make_query_result(["result"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        # Should not raise TypeError
        results = source.search_related(primary, ["query"], top_k=5, mode="vector")
        assert len(results) == 1

    def test_accepts_hybrid_kwarg(self):
        """search_related must accept hybrid= without TypeError."""
        col = FakeCollection(
            query_results_per_call=[make_query_result(["result"])],
            count=5,
        )
        source = make_source(col, files=NoFileFakeFiles())
        primary = Chunk(content="source", metadata=())
        results = source.search_related(
            primary, ["query"], top_k=5, hybrid={"vector_weight": 1.0}
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# content_attr — custom field extraction
# ---------------------------------------------------------------------------


class TestContentAttr:
    def test_single_custom_field_extracts_from_metadata(self):
        """content_attr=["description"] reads chunk text from metadata."""
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(
                    # Chroma document field (may be empty for pre-existing collections)
                    [""],
                    metas=[{"description": "the real text", "file_path": "a.md", "chunk_index": 0}],
                ),
            ],
            count=5,
        )
        source = make_source(col)
        source._chroma.content_attr = ["description"]
        results = source.search(SearchSpec(mode="vector", top_k=3, text_query="test"))
        assert len(results) == 1
        assert results[0].content == "the real text"

    def test_multi_field_content_is_json(self):
        """content_attr=["title", "body"] produces JSON-joined content."""
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(
                    [""],
                    metas=[{
                        "title": "My Title", "body": "My Body",
                        "file_path": "a.md", "chunk_index": 0,
                    }],
                ),
            ],
            count=5,
        )
        source = make_source(col)
        source._chroma.content_attr = ["title", "body"]
        results = source.search(SearchSpec(mode="vector", top_k=3, text_query="test"))
        assert len(results) == 1
        assert "My Title" in results[0].content
        assert "My Body" in results[0].content

    def test_default_content_attr_uses_document_field(self):
        """Default content_attr=["content"] uses the Chroma document field."""
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(["doc text from chroma"]),
            ],
            count=5,
        )
        source = make_source(col)
        assert source._chroma.content_attr == ["content"]
        results = source.search(SearchSpec(mode="vector", top_k=3, text_query="test"))
        assert results[0].content == "doc text from chroma"

    def test_search_content_uses_content_attr(self):
        """search_content also respects content_attr."""
        col = FakeCollection(
            query_results_per_call=[
                make_query_result(
                    [""],
                    metas=[{"description": "the text", "file_path": "a.md", "chunk_index": 0}],
                ),
            ],
            count=5,
        )
        source = make_source(col)
        source._chroma.content_attr = ["description"]
        results = source.search_content(SearchSpec(mode="vector", top_k=3, text_query="test"))
        assert results == ["the text"]
