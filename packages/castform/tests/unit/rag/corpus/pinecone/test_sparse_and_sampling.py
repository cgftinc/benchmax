"""Sparse-index guards and ID-based sampling for the Pinecone corpus.

Sparse indexes have no fixed dimension, so every dense-vector path must
either work without a vector (sampling) or fail with an actionable error
(search — until sparse query encoding lands).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from castform.rag.corpus.pinecone.index_client import PineconeIndexClient
from fakes.pinecone import FakeIndex, make_match, make_source


def _make_client(index) -> PineconeIndexClient:
    client = PineconeIndexClient(
        api_key="test-key",
        index_name="test",
        embed_fn=lambda texts: [[0.1, 0.2, 0.3]] * len(texts),
    )
    client._index = index
    return client


class _SparseIndex:
    """describe_index_stats shape of a sparse serverless index."""

    def __init__(self):
        self.queries: list[dict] = []

    def describe_index_stats(self):
        return SimpleNamespace(
            dimension=None,
            vector_type="sparse",
            total_vector_count=15,
            namespaces={"sample-data": SimpleNamespace(vector_count=15)},
        )

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(matches=[])


class TestSparseGuards:
    def test_query_raises_actionable_error(self):
        client = _make_client(_SparseIndex())
        with pytest.raises(ValueError, match="sparse"):
            client.query(vector=[0.1, 0.2, 0.3])

    def test_query_never_reaches_index(self):
        index = _SparseIndex()
        client = _make_client(index)
        with pytest.raises(ValueError):
            client.query(vector=[0.1, 0.2, 0.3])
        assert index.queries == []

    def test_zero_vector_raises_on_missing_dimension(self):
        client = _make_client(_SparseIndex())
        with pytest.raises(ValueError, match="sparse"):
            client.zero_vector()

    def test_dense_index_unaffected(self):
        client = _make_client(FakeIndex(matches=[make_match("1", "hello")]))
        result = client.query(vector=[0.1, 0.2, 0.3])
        assert len(result.matches) == 1


class TestVectorType:
    def test_sparse_detected(self):
        client = _make_client(_SparseIndex())
        assert client.vector_type() == "sparse"

    def test_dense_default_when_absent(self):
        # Older SDK stats objects have no vector_type attribute.
        client = _make_client(FakeIndex())
        assert client.vector_type() == "dense"


class TestNamespaceHandling:
    def test_none_namespace_normalizes_to_default(self):
        client = PineconeIndexClient(
            api_key="k", index_name="i", namespace=None, embed_fn=lambda t: []
        )
        assert client._namespace == ""

    def test_namespace_vector_count_scoped(self):
        class _Index:
            def describe_index_stats(self):
                return SimpleNamespace(
                    total_vector_count=500,
                    namespaces={
                        "__default__": SimpleNamespace(vector_count=480),
                        "other": SimpleNamespace(vector_count=20),
                    },
                )

        client = _make_client(_Index())
        # Default namespace → "__default__" entry, NOT the index-wide total.
        assert client.namespace_vector_count() == 480

        named = PineconeIndexClient(
            api_key="k", index_name="i", namespace="other", embed_fn=lambda t: []
        )
        named._index = _Index()
        assert named.namespace_vector_count() == 20

    def test_namespace_vector_count_empty_namespace(self):
        class _Index:
            def describe_index_stats(self):
                return SimpleNamespace(total_vector_count=15, namespaces={})

        client = _make_client(_Index())
        assert client.namespace_vector_count() == 0


class TestSampleChunksViaIds:
    def test_samples_without_querying(self):
        matches = [make_match(f"id-{i}", f"content {i}") for i in range(5)]
        index = FakeIndex(matches=matches)
        source = make_source(index)
        chunks = source.sample_chunks(3)
        assert len(chunks) == 3
        # ID-based sampling must not issue any vector query.
        assert index._call_idx == 0

    def test_min_chars_filtering(self):
        matches = [
            make_match("long-1", "x" * 50),
            make_match("short-1", "x"),
            make_match("long-2", "y" * 50),
        ]
        source = make_source(FakeIndex(matches=matches))
        chunks = source.sample_chunks(2, min_chars=10)
        assert len(chunks) == 2
        assert all(len(c.content) >= 10 for c in chunks)

    def test_empty_index_returns_empty(self):
        source = make_source(FakeIndex(matches=[]))
        assert source.sample_chunks(5) == []


class TestContentField:
    """content_field sugar — BYO indexes whose text isn't under `content`."""

    def _movie_client(self, **kwargs) -> PineconeIndexClient:
        client = PineconeIndexClient(
            api_key="k", index_name="movies", embed_fn=lambda t: [], **kwargs
        )
        return client

    def test_maps_custom_key_to_content(self):
        client = self._movie_client(content_field="summary")
        assert client._pc_field("content") == "summary"

        match = SimpleNamespace(
            id="0",
            metadata={"title": "Avatar", "summary": "On the alien world..."},
            score=0.9,
        )
        raw = client.match_to_raw(match)
        assert raw["content"] == "On the alien world..."
        assert raw["metadata"]["title"] == "Avatar"
        assert "summary" not in raw["metadata"]  # consumed as content

    def test_fetch_to_raw_uses_custom_key(self):
        client = self._movie_client(content_field="summary")
        raw = client.fetch_to_raw(
            "0", SimpleNamespace(metadata={"summary": "text here", "year": 2009})
        )
        assert raw["content"] == "text here"

    def test_empty_or_default_is_noop(self):
        for cf in (None, "", "content"):
            client = self._movie_client(content_field=cf)
            assert client._pc_field("content") == "content"

    def test_explicit_field_mapping_composes(self):
        client = self._movie_client(
            field_mapping={"path": "file_path"}, content_field="summary"
        )
        assert client._pc_field("content") == "summary"
        assert client._pc_field("file_path") == "path"

    def test_conflicting_content_mappings_raise(self):
        # field_mapping already claims the content column with a different
        # key — ambiguous, must fail instead of silently picking a winner.
        with pytest.raises(ValueError, match="conflicts"):
            self._movie_client(
                field_mapping={"description": "content"}, content_field="summary"
            )

    def test_agreeing_content_mappings_pass(self):
        client = self._movie_client(
            field_mapping={"summary": "content"}, content_field="summary"
        )
        assert client._pc_field("content") == "summary"


class TestEmptyContentGuard:
    def test_all_empty_content_raises_with_field_listing(self):
        # Records whose text lives under `summary` while the source still
        # reads `content` — the sample-movies failure mode.
        matches = [
            make_match("0", "", title="Avatar", summary="On the alien world..."),
            make_match("1", "", title="Endgame", summary="In the aftermath..."),
        ]
        source = make_source(FakeIndex(matches=matches))
        with pytest.raises(ValueError) as exc:
            source.sample_chunks(2)
        msg = str(exc.value)
        assert "content" in msg
        assert "summary" in msg and "title" in msg
        assert "content_field" in msg

    def test_some_content_passes(self):
        matches = [
            make_match("0", "real text"),
            make_match("1", "", stray="x"),
        ]
        source = make_source(FakeIndex(matches=matches))
        chunks = source.sample_chunks(2)
        assert len(chunks) == 2
