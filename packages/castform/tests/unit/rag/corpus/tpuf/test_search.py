"""Tests for TpufSearch — pickle-safe search client."""

from __future__ import annotations

import pickle

import cloudpickle
import pytest
from castform.rag.corpus.search_client import SearchClient
from castform.rag.corpus.turbopuffer.search import TpufSearch


class TestConformance:
    def test_isinstance(self):
        ts = TpufSearch(namespace="t")
        assert isinstance(ts, SearchClient)

    def test_pickle_roundtrip(self):
        ts = TpufSearch(namespace="ns", region="r")
        data = cloudpickle.dumps(ts)
        restored = pickle.loads(data)
        assert restored._namespace == "ns"
        assert restored._client is None
        # No credential stored on the instance — nothing to freeze.
        assert not hasattr(restored, "_api_key")

    def test_get_params_carries_no_credential(self):
        ts = TpufSearch(namespace="ns")
        params = ts.get_params()
        assert params["backend"] == "turbopuffer"
        assert "api_key" not in params


class TestAvailableModes:
    def test_lexical_only_without_embed_fn(self):
        ts = TpufSearch(namespace="t")
        assert ts.available_modes == ["lexical"]

    def test_all_modes_with_embed_fn(self):
        async def embed(texts):
            return [[0.1]] * len(texts)

        ts = TpufSearch(
            namespace="t",
            embed_fn=embed,
        )
        assert sorted(ts.available_modes) == ["hybrid", "lexical", "vector"]


class TestModeValidation:
    @pytest.mark.asyncio
    async def test_vector_without_embed_fn_raises(self):
        ts = TpufSearch(namespace="t")
        with pytest.raises(ValueError, match="vector"):
            await ts.search("query", mode="vector")

    @pytest.mark.asyncio
    async def test_hybrid_without_embed_fn_raises(self):
        ts = TpufSearch(namespace="t")
        with pytest.raises(ValueError, match="hybrid"):
            await ts.search("query", mode="hybrid")


class TestEmbed:
    @pytest.mark.asyncio
    async def test_returns_none_without_embed_fn(self):
        ts = TpufSearch(namespace="t")
        assert await ts.embed("hello") is None

    @pytest.mark.asyncio
    async def test_returns_vector_with_embed_fn(self):
        async def embed(texts):
            return [[1.0, 2.0]] * len(texts)

        ts = TpufSearch(
            namespace="t",
            embed_fn=embed,
        )
        assert await ts.embed("hello") == [1.0, 2.0]
