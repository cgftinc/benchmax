"""Protocol conformance tests for SearchClient implementations."""

from __future__ import annotations

import inspect
import pickle

import cloudpickle

from castform.rag.corpus.chroma.search import ChromaSearch
from castform.rag.corpus.embed import OpenAIEmbedder
from castform.rag.corpus.pinecone.search import PineconeSearch
from castform.rag.corpus.postgres.search import PostgresSearch
from castform.rag.corpus.search_client import SearchClient
from castform.rag.corpus.turbopuffer.search import TpufSearch


def test_rollout_search_and_embedding_contracts_are_async_only():
    """Prevent reintroducing a sync retrieval/auth path."""

    for implementation in (PostgresSearch, TpufSearch, PineconeSearch, ChromaSearch):
        assert inspect.iscoroutinefunction(implementation.search)
        assert inspect.iscoroutinefunction(implementation.embed)
    assert inspect.iscoroutinefunction(OpenAIEmbedder.__call__)


class TestPineconeSearchConformance:
    def test_isinstance(self):
        ps = PineconeSearch(index_name="test")
        assert isinstance(ps, SearchClient)

    def test_available_modes(self):
        ps = PineconeSearch(index_name="test")
        assert ps.available_modes == ["vector"]

    def test_get_params_carries_no_credential(self):
        ps = PineconeSearch(index_name="idx")
        params = ps.get_params()
        assert params["backend"] == "pinecone"
        assert params["index_name"] == "idx"
        assert "api_key" not in params

    def test_pickle_roundtrip(self):
        ps = PineconeSearch(index_name="test")
        data = cloudpickle.dumps(ps)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchClient)
        assert restored.available_modes == ["vector"]

    def test_pickle_carries_no_secret(self):
        # Default token_provider reads PINECONE_API_KEY at runtime; the bundle
        # carries the var name (via partial), not a key.
        ps = PineconeSearch(index_name="test")
        data = cloudpickle.dumps(ps)
        assert len(data) < 1500
        assert b"PINECONE_API_KEY" in data


class TestChromaSearchConformance:
    def test_isinstance(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        assert isinstance(cs, SearchClient)

    def test_available_modes_default(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        assert "vector" in cs.available_modes

    def test_get_params_self_hosted(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="h", port=9000)
        params = cs.get_params()
        assert params["backend"] == "chroma"
        assert params["mode"] == "self_hosted"
        assert params["host"] == "h"
        assert params["port"] == 9000

    def test_get_params_cloud(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(
            collection_name="test",
            tenant="t",
            database="d",
            token_provider="ck-fake-key-123",
        )
        params = cs.get_params()
        assert params["backend"] == "chroma"
        assert params["mode"] == "cloud"
        assert params["tenant"] == "t"
        assert params["database"] == "d"
        assert params["api_key"] == "ck-fake-..."

    def test_pickle_carries_no_secret_default(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        data = cloudpickle.dumps(cs)
        assert len(data) < 1500
        assert b"CHROMA_API_KEY" in data

    def test_pickle_roundtrip(self):
        from castform.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchClient)
