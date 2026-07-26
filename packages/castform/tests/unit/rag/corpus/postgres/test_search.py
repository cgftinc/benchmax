"""Tests for PostgresSearch — pickle-safe search client."""

from __future__ import annotations

import pickle
import subprocess
import sys
from types import SimpleNamespace

import cloudpickle
import pytest

from castform.platform.credentials import runtime_platform_bearer
from castform.rag.corpus.postgres.models import CorpusChunk, SearchResult
from castform.rag.corpus.postgres.search import PostgresSearch
from castform.rag.corpus.search_client import SearchClient


def test_postgres_search_import_does_not_require_tqdm():
    """Rollout-only search stays usable with base Castform dependencies."""

    code = """
import sys

class BlockTqdm:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tqdm' or fullname.startswith('tqdm.'):
            raise ModuleNotFoundError('tqdm intentionally unavailable')
        return None

sys.meta_path.insert(0, BlockTqdm())
from castform.rag.corpus.postgres.search import PostgresSearch
assert PostgresSearch('corpus', 'https://example.invalid').available_modes == ['lexical']
"""
    subprocess.run([sys.executable, "-c", code], check=True)


class TestConformance:
    def test_isinstance(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert isinstance(cs, SearchClient)

    def test_defaults_to_runtime_platform_bearer(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert cs._token_provider is runtime_platform_bearer

    def test_pickle_roundtrip(self):
        cs = PostgresSearch(corpus_name="cn", base_url="http://b")
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert restored._corpus_name == "cn"
        assert restored._client is None
        # No credential is stored on the instance — nothing to freeze.
        assert not hasattr(restored, "_api_key")

    def test_get_params_carries_no_credential(self):
        cs = PostgresSearch(corpus_name="c", base_url="http://b")
        params = cs.get_params()
        assert params["backend"] == "corpora"
        assert params["corpus_name"] == "c"
        assert "api_key" not in params

    @pytest.mark.asyncio
    async def test_runtime_name_resolution_never_creates_a_corpus(self):
        class ReadOnlyClient:
            def __init__(self) -> None:
                self.resolved: list[str] = []

            async def aget_corpus_by_name(self, name: str):
                self.resolved.append(name)
                return SimpleNamespace(id="corpus-id")

            def get_or_create_corpus(self, name: str):
                raise AssertionError("runtime search attempted corpus provisioning")

        client = ReadOnlyClient()
        search = PostgresSearch(corpus_name="existing", base_url="http://t")
        search._client = client  # type: ignore[assignment]

        assert await search._get_corpus_id() == "corpus-id"
        assert client.resolved == ["existing"]


class TestAvailableModes:
    def test_lexical_only(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert cs.available_modes == ["lexical"]


class TestSearch:
    @pytest.mark.asyncio
    async def test_uses_async_name_resolution_and_search_only(self):
        class AsyncOnlyClient:
            async def aget_corpus_by_name(self, name: str):
                assert name == "existing"
                return SimpleNamespace(id="corpus-id")

            async def asearch(self, **kwargs):
                assert kwargs == {
                    "corpus_id": "corpus-id",
                    "query": "needle",
                    "limit": 3,
                }
                return SearchResult(
                    results=[
                        CorpusChunk(
                            id="chunk-id",
                            content="found",
                            metadata={"file": "doc.md"},
                            score=0.75,
                        )
                    ],
                    total=1,
                    query="needle",
                )

            def get_corpus_by_name(self, name: str):
                raise AssertionError("sync corpus resolution was used")

            def search(self, **kwargs):
                raise AssertionError("sync search was used")

        search = PostgresSearch(corpus_name="existing", base_url="http://t")
        search._client = AsyncOnlyClient()  # type: ignore[assignment]

        assert await search.search("needle", top_k=3) == [
            {
                "content": "found",
                "source": "doc.md",
                "metadata": {"file": "doc.md"},
                "score": 0.75,
            }
        ]


class TestModeValidation:
    @pytest.mark.asyncio
    async def test_vector_raises(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            await cs.search("query", mode="vector")

    @pytest.mark.asyncio
    async def test_hybrid_raises(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            await cs.search("query", mode="hybrid")


class TestEmbed:
    @pytest.mark.asyncio
    async def test_always_none(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert await cs.embed("hello") is None
