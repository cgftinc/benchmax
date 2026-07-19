"""Tests for PineconeChunkSource configuration variants.

Covers degraded/variant customer configurations:
- No file metadata (pre-existing customer index)
- Custom field mapping (bring your own index)
- search_text auto-embed flow
- search_content cloudpickle-safe flow
- Capabilities reporting (vector-only, always, regardless of embed_fn)
- Lexical/hybrid mode rejection
- Built-in Pinecone Inference embed_fn
- Embed_fn override
- Dimension mismatch surfacing
- Missing mapped field graceful handling
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.search_schema.search_exceptions import (
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import SearchSpec

from fakes.pinecone import (
    FakeIndex,
    NoFileFakeFiles,
    make_match,
    make_source,
)

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_vector_only_mode(self):
        source = make_source(FakeIndex())
        caps = source.get_search_capabilities()
        assert caps["modes"] == {"vector"}
        assert "lexical" not in caps["modes"]
        assert "hybrid" not in caps["modes"]

    def test_backend_name(self):
        source = make_source(FakeIndex())
        assert source.get_search_capabilities()["backend"] == "pinecone"

    def test_filter_ops(self):
        source = make_source(FakeIndex())
        ops = source.get_search_capabilities()["filter_ops"]
        assert "eq" in ops["field"]
        assert "and" in ops["logical"]

    def test_capabilities_vector_only_even_with_custom_embed_fn(self):
        """Pinecone is always vector-only regardless of embed_fn — unlike
        turbopuffer where embed_fn unlocks hybrid mode."""
        custom_fn = lambda texts: [[1.0, 2.0, 3.0]] * len(texts)  # noqa: E731
        source = make_source(FakeIndex(), embed_fn=custom_fn)
        caps = source.get_search_capabilities()
        assert caps["modes"] == {"vector"}
        assert "hybrid" not in caps["modes"]


# ---------------------------------------------------------------------------
# Lexical/hybrid mode rejection
# ---------------------------------------------------------------------------


class TestModeRejection:
    def test_search_lexical_raises(self):
        source = make_source(FakeIndex())
        spec = SearchSpec(mode="lexical", text_query="test", top_k=5)
        with pytest.raises(UnsupportedSearchModeError):
            source.search(spec)

    def test_search_hybrid_raises(self):
        source = make_source(FakeIndex())
        spec = SearchSpec(
            mode="hybrid",
            text_query="test",
            vector_query=[0.1, 0.2, 0.3],
            top_k=5,
        )
        with pytest.raises(UnsupportedSearchModeError):
            source.search(spec)


# ---------------------------------------------------------------------------
# No file metadata — graceful degradation
# ---------------------------------------------------------------------------


class TestNoFileMetadata:
    def test_search_related_no_neighbor_skip(self):
        """Without file metadata, adjacent chunks are NOT skipped."""
        index = FakeIndex(
            matches=[make_match("1", "adjacent chunk"), make_match("2", "another chunk")]
        )
        source = make_source(index, files=NoFileFakeFiles())
        primary = Chunk(content="seed", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert len(results) == 2

    def test_search_related_same_file_always_false(self):
        """Without file metadata, same_file is always False."""
        index = FakeIndex(matches=[make_match("1", "result")])
        source = make_source(index, files=NoFileFakeFiles())
        primary = Chunk(content="seed", metadata=(("_pinecone_id", "src"),))
        results = source.search_related(primary, ["query"], top_k=5)
        assert results[0]["same_file"] is False

    def test_get_chunk_with_context_returns_fallback(self):
        """Without file metadata, context returns empty strings."""
        source = make_source(FakeIndex(), files=NoFileFakeFiles())
        chunk = Chunk(content="some content", metadata=())
        ctx = source.get_chunk_with_context(chunk)
        assert "chunk_content" in ctx
        assert ctx["prev_chunk_preview"] == ""
        assert ctx["next_chunk_preview"] == ""

    def test_get_top_level_chunks_returns_empty(self):
        """Without file metadata, top-level returns empty."""
        source = make_source(FakeIndex(), files=NoFileFakeFiles())
        assert source.get_top_level_chunks() == []


# ---------------------------------------------------------------------------
# Custom field mapping (bring your own index)
# ---------------------------------------------------------------------------


class TestCustomFieldMapping:
    def test_match_content_uses_mapped_field(self):
        index = FakeIndex(
            matches=[
                SimpleNamespace(
                    id="1",
                    metadata={"description": "mapped text", "path": "doc.md"},
                    score=0.9,
                ),
            ]
        )
        source = make_source(
            index,
            field_mapping={"description": "content", "path": "file_path"},
        )
        results = source.search(SearchSpec(mode="vector", vector_query=[0.1, 0.2, 0.3], top_k=5))
        assert len(results) == 1
        assert results[0].content == "mapped text"
        assert results[0].get_metadata("file_path") == "doc.md"

    def test_search_content_uses_mapped_field(self):
        index = FakeIndex(
            matches=[
                SimpleNamespace(
                    id="1",
                    metadata={"description": "mapped text"},
                    score=0.9,
                ),
            ]
        )
        source = make_source(
            index,
            field_mapping={"description": "content"},
        )
        results = source.search_content(
            SearchSpec(mode="vector", vector_query=[0.1, 0.2, 0.3], top_k=5)
        )
        assert results == ["mapped text"]

    def test_missing_mapped_content_field_returns_empty_string(self):
        """When mapped content field is absent from metadata, returns empty string."""
        index = FakeIndex(
            matches=[
                SimpleNamespace(
                    id="1",
                    metadata={"title": "only title, no description"},
                    score=0.9,
                ),
            ]
        )
        source = make_source(
            index,
            field_mapping={"description": "content"},
        )
        results = source.search(SearchSpec(mode="vector", vector_query=[0.1, 0.2, 0.3], top_k=5))
        assert len(results) == 1
        assert results[0].content == ""


# ---------------------------------------------------------------------------
# Minimal row attributes (customer's pre-existing data)
# ---------------------------------------------------------------------------


class TestMinimalRowAttributes:
    def test_match_to_raw_only_id_and_content(self):
        """Customer index has only id + content, no file metadata."""
        source = make_source(FakeIndex())
        match = SimpleNamespace(id="42", metadata={"content": "just text"}, score=0.5)
        raw = source._client.match_to_raw(match)
        assert raw["content"] == "just text"
        assert raw["metadata"]["_pinecone_id"] == "42"
        assert raw["metadata"].get("file_path") is None
        assert raw["metadata"].get("chunk_index") is None

    def test_match_to_raw_extra_custom_fields(self):
        """Customer index has extra fields not in standard schema."""
        source = make_source(FakeIndex())
        match = SimpleNamespace(
            id="1",
            metadata={"content": "text", "custom_tag": "important", "priority": 5},
            score=0.5,
        )
        raw = source._client.match_to_raw(match)
        assert raw["metadata"]["custom_tag"] == "important"
        assert raw["metadata"]["priority"] == 5


# ---------------------------------------------------------------------------
# search_text auto-embed flow
# ---------------------------------------------------------------------------


class TestSearchTextFlow:
    def test_search_text_returns_chunks(self):
        index = FakeIndex(matches=[make_match("1", "found it")])
        source = make_source(index)
        results = source.search_text("find me", top_k=5)
        assert len(results) == 1
        assert results[0].content == "found it"

    def test_search_text_calls_embed_fn(self):
        """Verify that search_text actually embeds the query text."""
        calls = []

        def tracking_embed(texts):
            calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)

        index = FakeIndex(matches=[make_match("1", "result")])
        source = make_source(index, embed_fn=tracking_embed)
        source.search_text("my search query", top_k=5)
        assert len(calls) == 1
        assert calls[0] == ["my search query"]

    def test_search_content_returns_strings(self):
        index = FakeIndex(matches=[make_match("1", "content text")])
        source = make_source(index)
        spec = SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2, 0.3])
        results = source.search_content(spec)
        assert results == ["content text"]
        assert all(isinstance(r, str) for r in results)


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_returns_vector(self):
        source = make_source(FakeIndex())
        vec = source.embed_query("hello")
        assert isinstance(vec, list)
        assert len(vec) == 3

    def test_uses_embed_fn(self):
        calls = []

        def tracking_embed(texts):
            calls.append(texts)
            return [[1.0, 2.0, 3.0]] * len(texts)

        source = make_source(FakeIndex(), embed_fn=tracking_embed)
        vec = source.embed_query("test")
        assert vec == [1.0, 2.0, 3.0]
        assert calls == [["test"]]


# ---------------------------------------------------------------------------
# Built-in Pinecone Inference embed_fn
# ---------------------------------------------------------------------------


class TestBuiltInEmbed:
    def test_build_embed_fn_is_callable(self):
        """_build_pinecone_embed_fn returns a callable."""
        from castform.rag.corpus.pinecone.index_client import PineconeIndexClient

        client = PineconeIndexClient.__new__(PineconeIndexClient)
        client._api_key = "test"
        client._embed_model = "multilingual-e5-large"
        fn = client._build_pinecone_embed_fn()
        assert callable(fn)

    def test_default_embed_model_value(self):
        """PineconeIndexClient defaults embed_model to multilingual-e5-large."""
        from castform.rag.corpus.pinecone.index_client import PineconeIndexClient

        sig = inspect.signature(PineconeIndexClient.__init__)
        default = sig.parameters["embed_model"].default
        assert default == "multilingual-e5-large"

    def test_custom_embed_fn_overrides_model(self):
        """When embed_fn provided, embed_model is ignored."""
        custom_embed = lambda texts: [[9.0]] * len(texts)  # noqa: E731
        source = make_source(FakeIndex(), embed_fn=custom_embed)
        vec = source.embed_query("hello")
        assert vec == [9.0]


# ---------------------------------------------------------------------------
# Dimension mismatch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dimension mismatch
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    def test_embed_dimension_vs_index_dimension_detectable(self):
        """When embed_fn returns vectors of wrong dimension, the mismatch
        is detectable by comparing embed output to zero_vector() length.

        In production the Pinecone API returns a 400 error. Our fakes
        don't enforce this, so we verify the dimensions are inspectable
        for customers debugging dimension issues.
        """
        wrong_dim_embed = lambda texts: [[0.1, 0.2]] * len(texts)  # noqa: E731
        index = FakeIndex(dimension=5)
        source = make_source(index, embed_fn=wrong_dim_embed)

        embed_dim = len(source.embed_query("test"))
        index_dim = len(source._client.zero_vector())

        assert embed_dim == 2
        assert index_dim == 5
        assert embed_dim != index_dim  # mismatch detectable
