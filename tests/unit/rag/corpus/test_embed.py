"""Tests for the platform-backed ``embed_fn`` helper."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import cloudpickle

from benchmax.rag.corpus.embed import DEFAULT_EMBED_MODEL, platform_embed_fn


class _FakeOpenAI:
    """Records construction args + embeddings calls; returns a vector per input."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)
        _FakeOpenAI.instances.append(self)

    def _create(self, *, model, input):
        self.calls.append({"model": model, "input": list(input)})
        # one vector per input row, length = len(text) so we can assert ordering
        data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
        return SimpleNamespace(data=data)


def _patch(monkeypatch, *, key="tok-seam"):
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    # bypass the credential seam (no network / no real token needed)
    monkeypatch.setattr(
        "benchmax.rag.corpus.embed.resolve_judge_key",
        lambda api_key, base_url: api_key or key,
    )


def test_returns_one_vector_per_input_in_order(monkeypatch):
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)

    fn = platform_embed_fn()
    out = fn(["a", "bbb"])

    assert out == [[1.0], [3.0]]
    client = _FakeOpenAI.instances[0]
    assert client.base_url == "https://llm.test.example/v1"
    assert client.api_key == "tok-seam"
    assert client.calls[0]["model"] == DEFAULT_EMBED_MODEL
    assert client.calls[0]["input"] == ["a", "bbb"]


def test_client_built_lazily_and_once(monkeypatch):
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)

    fn = platform_embed_fn()
    assert _FakeOpenAI.instances == []  # nothing constructed until first call

    fn(["x"])
    fn(["y"])
    assert len(_FakeOpenAI.instances) == 1  # reused across calls


def test_base_url_resolves_at_call_time_from_config(monkeypatch):
    # Cross-env story: a sandbox sets CASTFORM_BASE_DOMAIN, which must be read when the
    # fn runs, not when it was authored.
    _patch(monkeypatch)
    fn = platform_embed_fn()  # authored with no base_url

    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.dev")
    monkeypatch.delenv("CASTFORM_LLM_URL", raising=False)
    fn(["q"])

    assert _FakeOpenAI.instances[0].base_url == "https://llm.castform.dev/v1"


def test_explicit_overrides_win(monkeypatch):
    _patch(monkeypatch)
    fn = platform_embed_fn(
        model="custom-embed", base_url="https://override/v1", api_key="sk-explicit"
    )
    fn(["q"])

    client = _FakeOpenAI.instances[0]
    assert client.base_url == "https://override/v1"
    assert client.api_key == "sk-explicit"
    assert client.calls[0]["model"] == "custom-embed"


def test_cloudpickle_roundtrip_before_first_call(monkeypatch):
    # The env bundle is cloudpickled before it ever runs; the fn must survive with no
    # live client captured.
    fn = platform_embed_fn(api_key="tok-seam")
    restored = pickle.loads(cloudpickle.dumps(fn))

    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)
    assert restored(["zz"]) == [[2.0]]
