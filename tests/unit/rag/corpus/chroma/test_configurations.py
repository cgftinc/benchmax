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

from unittest.mock import MagicMock, patch

import pytest
from fakes.chroma import (
    FakeCollection,
    NoFileFakeFiles,
    make_query_result,
    make_source,
)

from benchmax.rag.chunkers.models import Chunk
from benchmax.rag.corpus.search_schema.search_exceptions import (
    UnsupportedSearchModeError,
)
from benchmax.rag.corpus.search_schema.search_types import SearchSpec

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
        # Enable hybrid/lexical
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
