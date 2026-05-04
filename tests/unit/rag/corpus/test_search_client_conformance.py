"""Protocol conformance tests for SearchClient implementations."""

from __future__ import annotations

import pickle

import cloudpickle

from benchmax.rag.corpus.pinecone.search import PineconeSearch
from benchmax.rag.corpus.search_client import SearchClient


class TestPineconeSearchConformance:
    def test_isinstance(self):
        ps = PineconeSearch(api_key="test", index_name="test")
        assert isinstance(ps, SearchClient)

    def test_available_modes(self):
        ps = PineconeSearch(api_key="test", index_name="test")
        assert ps.available_modes == ["vector"]

    def test_get_params_masks_key(self):
        ps = PineconeSearch(api_key="sk_secret_key_here", index_name="idx")
        params = ps.get_params()
        assert params["backend"] == "pinecone"
        assert params["index_name"] == "idx"
        assert "secret_key_here" not in params["api_key"]

    def test_pickle_roundtrip(self):
        ps = PineconeSearch(api_key="test", index_name="test")
        data = cloudpickle.dumps(ps)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchClient)
        assert restored.available_modes == ["vector"]

    def test_pickle_size_reasonable(self):
        ps = PineconeSearch(api_key="test", index_name="test")
        data = cloudpickle.dumps(ps)
        assert len(data) < 1000  # should be ~200B


class TestChromaSearchConformance:
    def test_isinstance(self):
        from benchmax.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        assert isinstance(cs, SearchClient)

    def test_available_modes_default(self):
        from benchmax.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        assert "vector" in cs.available_modes

    def test_get_params(self):
        from benchmax.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="h", port=9000)
        params = cs.get_params()
        assert params["backend"] == "chroma"
        assert params["host"] == "h"
        assert params["port"] == 9000

    def test_pickle_roundtrip(self):
        from benchmax.rag.corpus.chroma.search import ChromaSearch

        cs = ChromaSearch(collection_name="test", host="localhost")
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert isinstance(restored, SearchClient)
