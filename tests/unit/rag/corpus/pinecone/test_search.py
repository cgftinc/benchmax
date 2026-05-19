"""Tests for PineconeSearch — pickle-safe search client."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import cloudpickle
import pytest

from benchmax.rag.corpus.pinecone.search import PineconeSearch


class FakeIndex:
    def query(self, **kwargs):
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id="1", metadata={"content": "result one"}, score=0.9
                ),
                SimpleNamespace(
                    id="2", metadata={"content": "result two"}, score=0.8
                ),
            ]
        )

    def describe_index_stats(self):
        return SimpleNamespace(dimension=3)


def _make_search(embed_fn=None) -> PineconeSearch:
    ps = PineconeSearch(
        api_key="test",
        index_name="test",
        embed_fn=embed_fn or (lambda texts: [[0.1, 0.2, 0.3]] * len(texts)),
    )
    # Inject fake client to avoid real Pinecone calls
    from benchmax.rag.corpus.pinecone.index_client import PineconeIndexClient

    client = PineconeIndexClient.__new__(PineconeIndexClient)
    client._api_key = "test"
    client._index_name = "test"
    client._index_host = None
    client._namespace = ""
    client._embed_model = "test"
    client.embed_fn = embed_fn or (lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
    client._field_mapping = {"content": "content"}
    client._reverse_mapping = {"content": "content"}
    client._index = FakeIndex()
    client._known_ids = None
    client._vector_dim = 3
    ps._client = client
    return ps


class TestSearch:
    def test_returns_dicts(self):
        ps = _make_search()
        results = ps.search("query")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["content"] == "result one"
        assert results[1]["content"] == "result two"

    def test_mode_lexical_raises(self):
        ps = _make_search()
        with pytest.raises(ValueError, match="lexical"):
            ps.search("query", mode="lexical")

    def test_mode_hybrid_raises(self):
        ps = _make_search()
        with pytest.raises(ValueError, match="hybrid"):
            ps.search("query", mode="hybrid")

    def test_mode_auto_works(self):
        ps = _make_search()
        results = ps.search("query", mode="auto")
        assert len(results) == 2

    def test_mode_vector_works(self):
        ps = _make_search()
        results = ps.search("query", mode="vector")
        assert len(results) == 2


class TestEmbed:
    def test_delegates_to_embed_fn(self):
        calls = []

        def tracking(texts):
            calls.append(texts)
            return [[1.0, 2.0, 3.0]] * len(texts)

        ps = _make_search(embed_fn=tracking)
        vec = ps.embed("hello")
        assert vec == [1.0, 2.0, 3.0]
        assert calls == [["hello"]]


class TestPickle:
    def test_roundtrip_preserves_config(self):
        ps = PineconeSearch(api_key="k", index_name="idx", namespace="ns")
        data = cloudpickle.dumps(ps)
        restored = pickle.loads(data)
        assert restored._api_key == "k"
        assert restored._index_name == "idx"
        assert restored._namespace == "ns"
        assert restored._client is None  # stripped

    def test_available_modes_after_restore(self):
        ps = PineconeSearch(api_key="k", index_name="idx")
        restored = pickle.loads(cloudpickle.dumps(ps))
        assert restored.available_modes == ["vector"]
