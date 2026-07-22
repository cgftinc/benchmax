"""Protocol conformance tests for SearchClient implementations."""

from __future__ import annotations

import logging
import pickle
import sys
import types

import cloudpickle
import pytest

from benchmax.bundle import dump_bundle, load_bundle
from benchmax.envs import BaseEnv
from castform.rag.corpus.embed import platform_embed_fn
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA
from castform.rag.corpus.neon.schema import NeonTableSpec
from castform.rag.corpus.neon.search import NeonSearch, surfaced_score
from castform.rag.corpus.pinecone.search import PineconeSearch
from castform.rag.corpus.search_client import SearchClient

# Defined at module scope so cloudpickle can pickle the bundle env by value when
# the test module is registered as a local module (dump_bundle enforces this).
_TEST_MODULE = sys.modules[__name__]


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


# --- Neon SearchClient conformance (Slice 4) ---------------------------------

# One canned candidate row set: (id, content, metadata, source_file, chunk_index,
# native_score) — the shape NeonClient's candidate SELECT returns.
_CANNED_ROWS = [
    ("h1", "alpha content", {"lang": "en"}, "a.md", 0, -3.5),
    ("h2", "beta content", {"lang": "en"}, "b.md", 1, -2.0),
]

SENTINEL_RO_DSN = "postgresql://benchmax_ro@sentinel-host:5432/corpus"


def _dummy_embed_fn():
    return lambda texts: [[0.0] * 8 for _ in texts]


class _FakeNeonClient:
    """Stands in for NeonClient: the candidate-SQL builders return dummies (they
    are never executed) and the execute seam returns canned rows."""

    def __init__(self, rows):
        self.rows = rows
        self.bm25_setup = None

    def vector_candidates_sql(self, spec, where=None):
        return (None, {})

    def bm25_candidates_sql(self, spec, where=None, *, schema):
        return (None, {})

    def execute(self, query, params=None):
        return self.rows

    def execute_read_txn(self, query, params=None, *, session_setup=None):
        self.bm25_setup = session_setup
        return self.rows


class _SearchEnv(BaseEnv):
    """Minimal env carrying a SearchClient as a constructor arg (bundle carrier)."""

    reward_keys = ("score",)

    def __init__(self, *, search_client):
        super().__init__()
        self.search_client = search_client

    async def create_dataset(self, split, base_dir):
        raise NotImplementedError

    async def compute_reward(self, rollout):
        return {"score": 0.0}


class _FakeOpenAI:
    """Records construction; returns one zero vector per input (embedder warming)."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.embeddings = types.SimpleNamespace(create=self._create)
        _FakeOpenAI.instances.append(self)

    def _create(self, *, model, input):
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.0] * 8) for _ in input]
        )


def _warm_patch(monkeypatch):
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        "castform.rag.corpus.embed.resolve_judge_key",
        lambda api_key, base_url: api_key or "tok",
    )
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")


def _stub_backend(monkeypatch, client, rows):
    monkeypatch.setattr(client, "_get_client", lambda: _FakeNeonClient(rows))
    monkeypatch.setattr(client, "_resolve_spec", lambda c: NeonTableSpec("corpus", 1))


class TestNeonSearchConformance:
    def test_isinstance(self):
        assert isinstance(NeonSearch("corpus"), SearchClient)

    def test_available_modes_without_embedder(self):
        assert NeonSearch("corpus").available_modes == ["lexical"]

    def test_available_modes_with_embedder(self):
        ns = NeonSearch("corpus", embed_fn=_dummy_embed_fn())
        assert ns.available_modes == ["hybrid", "lexical", "vector"]

    def test_get_params_carries_no_credential(self):
        ns = NeonSearch("corpus", dsn_provider=lambda: "postgresql://ro@h/db")
        params = ns.get_params()
        assert params == {
            "backend": "neon",
            "table": "corpus",
            "schema": CORPUS_SCHEMA,
        }
        assert "postgresql" not in "".join(str(v) for v in params.values())

    def test_pickle_roundtrip(self):
        restored = pickle.loads(cloudpickle.dumps(NeonSearch("corpus")))
        assert isinstance(restored, SearchClient)
        assert restored.available_modes == ["lexical"]

    def test_pickle_carries_no_secret_default(self):
        # Default RO provider reads NEON_CORPUS_DSN_RO at runtime; the pickle
        # carries the var NAME (env_token partial), never a DSN.
        data = cloudpickle.dumps(NeonSearch("corpus"))
        assert b"NEON_CORPUS_DSN_RO" in data
        assert b"postgresql://" not in data


def test_neon_bundle_roundtrip_reference_path(monkeypatch):
    # Self-serve path: the RO DSN is NOT baked — the bundle carries the env-var name.
    _warm_patch(monkeypatch)
    ns = NeonSearch("corpus", embed_fn=platform_embed_fn(api_key="tok"), dsn_provider=None)
    assert ns.embed("warm me")  # WARM the embedder before pickling (B2)

    bundle = dump_bundle(
        _SearchEnv, constructor_args={"search_client": ns}, local_modules=[_TEST_MODULE]
    )
    assert b"NEON_CORPUS_DSN_RO" in bundle.pickled  # var name travels
    assert b"postgresql://" not in bundle.pickled  # no DSN literal baked

    env = load_bundle(bundle)  # dump -> load -> instantiate
    client = env.search_client
    assert client._embed_fn._client is None  # warmed client dropped across the pickle

    _stub_backend(monkeypatch, client, _CANNED_ROWS)
    results = client.search("hello", mode="lexical", top_k=2)  # -> search
    assert [r["content"] for r in results] == ["alpha content", "beta content"]
    assert results[0]["source"] == "a.md"
    assert results[0]["score"] == surfaced_score(0)


def test_neon_bundle_roundtrip_baked_path(monkeypatch, caplog):
    # Platform path: the RESOLVED RO DSN IS baked into the pickle by design.
    _warm_patch(monkeypatch)
    with pytest.warns(UserWarning) as warned:
        ns = NeonSearch(
            "corpus",
            embed_fn=platform_embed_fn(api_key="tok"),
            dsn_provider=SENTINEL_RO_DSN,
        )
    assert ns.embed("warm me")
    assert all(SENTINEL_RO_DSN not in str(w.message) for w in warned)  # warning omits the DSN

    with caplog.at_level(logging.DEBUG):
        bundle = dump_bundle(
            _SearchEnv,
            constructor_args={"search_client": ns},
            local_modules=[_TEST_MODULE],
        )
    assert SENTINEL_RO_DSN.encode() in bundle.pickled  # baked by design
    assert "benchmax_ro" in SENTINEL_RO_DSN  # sanity: the sentinel is RO-scoped
    assert b"benchmax_writer" not in bundle.pickled  # never the RW role
    assert SENTINEL_RO_DSN not in caplog.text  # nothing logs the DSN

    env = load_bundle(bundle)
    client = env.search_client
    assert client._dsn_provider() == SENTINEL_RO_DSN  # baked provider yields the RO DSN

    _stub_backend(monkeypatch, client, _CANNED_ROWS)
    assert client.search("hello", mode="lexical", top_k=1)[0]["content"] == "alpha content"
