"""Tests for Benchmax's explicit-auth OpenAI embedding callable."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import cloudpickle
import pytest
from benchmax.auth import InjectedAuth, StaticBearerAuth, bind_model_auth
from benchmax.rag.embed import DEFAULT_EMBED_MODEL, OpenAIEmbedder


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)
        self.closed = False

    async def _create(self, *, model, input, timeout):
        self.calls.append({"model": model, "input": list(input), "timeout": timeout})
        data = [SimpleNamespace(embedding=[float(len(text))]) for text in input]
        return SimpleNamespace(data=data)

    async def close(self) -> None:
        self.closed = True


def _patch_client(monkeypatch):
    created: list[tuple[dict, _FakeClient]] = []

    def create(**kwargs):
        client = _FakeClient()
        created.append((kwargs, client))
        return client

    monkeypatch.setattr(
        "benchmax.rag.embed.create_async_openai_client",
        create,
    )
    return created


@pytest.mark.asyncio
async def test_returns_one_vector_per_input_with_explicit_auth(monkeypatch):
    created = _patch_client(monkeypatch)
    auth = StaticBearerAuth("sk-explicit")
    embedder = OpenAIEmbedder(
        model=DEFAULT_EMBED_MODEL,
        base_url="https://model.example/v1",
        auth=auth,
    )

    assert await embedder(["a", "bbb"]) == [[1.0], [3.0]]

    kwargs, client = created[0]
    assert kwargs["auth"] is auth
    assert kwargs["base_url"] == "https://model.example/v1"
    assert kwargs["model"] == DEFAULT_EMBED_MODEL
    assert client.calls == [
        {
            "model": DEFAULT_EMBED_MODEL,
            "input": ["a", "bbb"],
            "timeout": 60.0,
        }
    ]
    assert client.closed


@pytest.mark.asyncio
async def test_uses_one_request_scoped_client_per_call(monkeypatch):
    created = _patch_client(monkeypatch)
    embedder = OpenAIEmbedder(
        model=DEFAULT_EMBED_MODEL,
        base_url="https://model.example/v1",
        auth=StaticBearerAuth("sk-explicit"),
    )

    await embedder(["x"])
    await embedder(["y"])

    assert len(created) == 2
    assert all(client.closed for _, client in created)


@pytest.mark.asyncio
async def test_injected_auth_remains_bound_at_call_time(monkeypatch):
    created = _patch_client(monkeypatch)
    embedder = OpenAIEmbedder(
        model=DEFAULT_EMBED_MODEL,
        base_url="https://llm.castform.dev/v1",
        auth=InjectedAuth("embedding"),
    )

    with bind_model_auth({"embedding": StaticBearerAuth("runtime-token")}):
        await embedder(["q"])

    assert created[0][0]["auth"] == InjectedAuth("embedding")


@pytest.mark.asyncio
async def test_empty_input_skips_client_creation(monkeypatch):
    created = _patch_client(monkeypatch)
    embedder = OpenAIEmbedder(
        model=DEFAULT_EMBED_MODEL,
        base_url="https://model.example/v1",
        auth=StaticBearerAuth("sk-explicit"),
    )

    assert await embedder([]) == []
    assert created == []


@pytest.mark.asyncio
async def test_cloudpickle_roundtrip_contains_auth_declaration_not_client(monkeypatch):
    created = _patch_client(monkeypatch)
    embedder = OpenAIEmbedder(
        auth=StaticBearerAuth("sk-explicit"),
        model="custom-embed",
        base_url="https://model.example/v1",
    )
    restored = pickle.loads(cloudpickle.dumps(embedder))

    assert await restored(["zz"]) == [[2.0]]
    assert created[0][0]["auth"] == StaticBearerAuth("sk-explicit")
    assert created[0][0]["model"] == "custom-embed"
