"""Tests for PostgresSearch — pickle-safe search client."""

from __future__ import annotations

import pickle
import subprocess
import sys

import cloudpickle
import pytest

from castform.platform.credentials import platform_bearer
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

    def test_defaults_to_platform_bearer(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert cs._token_provider is platform_bearer

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


class TestAvailableModes:
    def test_lexical_only(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert cs.available_modes == ["lexical"]


class TestModeValidation:
    def test_vector_raises(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            cs.search("query", mode="vector")

    def test_hybrid_raises(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            cs.search("query", mode="hybrid")


class TestEmbed:
    def test_always_none(self):
        cs = PostgresSearch(corpus_name="t", base_url="http://t")
        assert cs.embed("hello") is None
