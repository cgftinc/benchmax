"""Sparse-index guards and ID-based sampling for the Pinecone corpus.

Sparse indexes have no fixed dimension, so every dense-vector path must
either work without a vector (sampling) or fail with an actionable error
(search — until sparse query encoding lands).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmax.rag.corpus.pinecone.index_client import PineconeIndexClient

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
