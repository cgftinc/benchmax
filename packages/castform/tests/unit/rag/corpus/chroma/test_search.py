"""Tests for ChromaSearch — pickle-safe search client."""

from __future__ import annotations

import pickle

import cloudpickle
import pytest

from castform.rag.corpus.chroma.search import ChromaSearch
from castform.rag.corpus.search_schema.search_exceptions import (
    LocalEmbeddingDownloadDisallowedError,
)


def _make_search(modes=None, embed_fn=None, ef_name="chroma-cloud-qwen") -> ChromaSearch:
    """Build a ChromaSearch with a fake ChromaClient injected.

    ``ef_name`` is the collection's configured embedding function: defaults to a
    hosted server-side EF (dense-safe). Pass ``"default"`` (all-MiniLM, would
    download) or ``None`` (no EF) to exercise the unsafe path.
    """
    from types import SimpleNamespace

    from castform.rag.corpus.chroma.client import ChromaClient

    cs = ChromaSearch(
        collection_name="test",
        host="localhost",
        embed_fn=embed_fn,
    )

    # Inject fake client
    client = ChromaClient.__new__(ChromaClient)
    client.collection_name = "test"
    client.api_key = None
    client.tenant = None
    client.database = None
    client.host = "localhost"
    client.port = 8000
    client.path = None
    client.embed_fn = embed_fn
    client.content_attr = ["content"]
    client.distance_metric = "cosine"
    client.enable_bm25 = False
    client.rrf_k = 60
    client.rrf_oversample = 20
    client.rrf_max_candidates = 200
    client._raw_client = None
    cfg = {"embedding_function": {"name": ef_name}} if ef_name is not None else {}
    client._collection = SimpleNamespace(
        _model=SimpleNamespace(configuration_json=cfg)
    )
    client.get_collection = lambda: client._collection
    client._embed_dim = None
    client._total_count = None
    client.search_api = False
    client.modes = modes or {"vector"}
    client.ranking = {"cosine"}

    # Mock query_raw to return canned results
    client.query_raw = lambda **kw: [
        {"id": "1", "content": "result one", "metadata": {}, "score": 0.9},
        {"id": "2", "content": "result two", "metadata": {}, "score": 0.8},
    ]
    client.extract_content = lambda doc, meta: doc

    cs._client = client
    return cs


class TestSearch:
    def test_returns_dicts(self):
        cs = _make_search()
        results = cs.search("query")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["content"] == "result one"
        assert results[1]["content"] == "result two"

    def test_mode_auto_picks_vector_when_only_vector(self):
        cs = _make_search(modes={"vector"})
        results = cs.search("query", mode="auto")
        assert len(results) == 2

    def test_mode_auto_picks_hybrid_when_available(self):
        cs = _make_search(modes={"vector", "lexical", "hybrid"})
        cs._client.search_api = True
        cs._client.search_api_raw = lambda **kw: [
            {"id": "1", "content": "hybrid result", "metadata": {}, "score": 0.9},
        ]
        results = cs.search("query", mode="auto")
        assert results[0]["content"] == "hybrid result"

    def test_mode_auto_picks_lexical_over_vector(self):
        cs = _make_search(modes={"vector", "lexical"})
        cs._client.search_api = True
        cs._client.search_api_raw = lambda **kw: [
            {"id": "1", "content": "lexical result", "metadata": {}, "score": 0.9},
        ]
        results = cs.search("query", mode="auto")
        assert results[0]["content"] == "lexical result"

    def test_invalid_mode_raises(self):
        cs = _make_search(modes={"vector"})
        with pytest.raises(ValueError, match="does not support mode 'hybrid'"):
            cs.search("query", mode="hybrid")

    def test_unsafe_dense_uses_lexical_at_inference(self):
        """Default-EF collection + BM25: the search tool uses BM25, no download.

        Mirrors a rollout/inference search against a default (all-MiniLM)
        collection — embedding would download a model, so we use lexical instead.
        """
        cs = _make_search(modes={"vector", "lexical", "hybrid"}, ef_name="default")
        cs._client.search_api = True
        seen = {}
        cs._client.search_api_raw = lambda **kw: seen.update(kw) or [
            {"id": "1", "content": "bm25 result", "metadata": {}, "score": 0.9},
        ]
        results = cs.search("query", mode="auto")
        assert results[0]["content"] == "bm25 result"
        assert seen["mode"] == "lexical"  # never vector/hybrid -> no embed

    def test_unsafe_dense_without_bm25_raises_at_inference(self):
        """Default-EF, vector-only collection: refuse rather than download."""
        cs = _make_search(modes={"vector"}, ef_name="default")
        with pytest.raises(LocalEmbeddingDownloadDisallowedError):
            cs.search("query", mode="auto")

    def test_explicit_vector_on_unsafe_collection_degrades(self):
        """Even an explicit vector request degrades to BM25 when unsafe."""
        cs = _make_search(modes={"vector", "lexical"}, ef_name="default")
        cs._client.search_api = True
        seen = {}
        cs._client.search_api_raw = lambda **kw: seen.update(kw) or []
        cs.search("query", mode="vector")
        assert seen["mode"] == "lexical"


class TestEmbed:
    def test_returns_none_without_embed_fn(self):
        cs = _make_search(embed_fn=None)
        cs._client.embed = lambda text: None
        assert cs.embed("test") is None

    def test_returns_vector_with_embed_fn(self):
        fn = lambda texts: [[1.0, 2.0]] * len(texts)  # noqa: E731
        cs = _make_search(embed_fn=fn)
        cs._client.embed = lambda text: fn([text])[0]
        assert cs.embed("test") == [1.0, 2.0]


class TestAvailableModes:
    def test_reflects_client_modes(self):
        cs = _make_search(modes={"vector", "lexical", "hybrid"})
        assert sorted(cs.available_modes) == ["hybrid", "lexical", "vector"]

    def test_vector_only(self):
        cs = _make_search(modes={"vector"})
        assert cs.available_modes == ["vector"]


class TestPickle:
    def test_roundtrip_preserves_config_self_hosted(self):
        cs = ChromaSearch(collection_name="col", host="h", port=9000)
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert restored._collection_name == "col"
        assert restored._host == "h"
        assert restored._port == 9000
        assert restored._client is None

    def test_roundtrip_preserves_config_cloud(self):
        cs = ChromaSearch(
            collection_name="col",
            tenant="t",
            database="d",
            token_provider="ck-key",
        )
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert restored._collection_name == "col"
        assert restored._tenant == "t"
        assert restored._database == "d"
        assert restored._client is None

    def test_roundtrip_size(self):
        cs = ChromaSearch(collection_name="test", host="localhost")
        assert len(cloudpickle.dumps(cs)) < 1500
