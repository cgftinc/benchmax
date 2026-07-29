"""Async-only corpus client and source tests."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.postgres.client import CorpusClient
from castform.rag.corpus.postgres.exceptions import AuthenticationError, CorpusAPIError
from castform.rag.corpus.postgres.source import PostgresChunkSource


def test_corpus_client_public_io_is_async_only():
    methods = (
        "create_corpus",
        "list_corpora",
        "get_corpus",
        "get_corpus_by_name",
        "delete_corpus",
        "get_or_create_corpus",
        "upload_chunks",
        "list_corpus_chunks",
        "search",
        "search_with_chunks",
        "close",
    )
    for name in methods:
        assert inspect.iscoroutinefunction(getattr(CorpusClient, name))
    assert inspect.iscoroutinefunction(CorpusClient._request)
    for obsolete in (
        "alist_corpora",
        "aget_corpus_by_name",
        "asearch",
        "asearch_with_chunks",
        "aclose",
        "_arequest",
    ):
        assert not hasattr(CorpusClient, obsolete)


def test_postgres_source_network_io_is_async_only():
    methods = (
        "populate_from_folder",
        "populate_from_chunks",
        "populate_from_existing_corpus",
        "populate_from_existing_corpus_name",
        "search_related",
        "search",
        "search_text",
        "search_content",
        "close",
    )
    for name in methods:
        assert inspect.iscoroutinefunction(getattr(PostgresChunkSource, name))
    for obsolete in ("asearch_related", "aclose"):
        assert not hasattr(PostgresChunkSource, obsolete)


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, json={"ok": True})


class _FakeAsyncClient:
    """Returns queued responses in order and records calls — stands in for the
    lazily-built ``httpx.AsyncClient`` so ``_request`` runs without a network."""

    is_closed = False

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path))
        return self._responses[len(self.calls) - 1]


@pytest.fixture
def no_async_sleep(monkeypatch):
    """Capture awaited backoff durations instead of actually sleeping."""
    slept: list[float] = []

    async def _sleep(delay):
        slept.append(delay)

    monkeypatch.setattr("castform.rag.corpus.postgres.client.asyncio.sleep", _sleep)
    return slept


def _wire_async(monkeypatch, client: CorpusClient, responses) -> _FakeAsyncClient:
    fake = _FakeAsyncClient(responses)
    monkeypatch.setattr(client, "_get_async_client", lambda: fake)
    return fake


async def test_request_retries_on_429_then_succeeds(monkeypatch, no_async_sleep):
    client = CorpusClient(base_url="http://corpora", token_provider=lambda: "tok")
    fake = _wire_async(monkeypatch, client, [_resp(429, {"Retry-After": "2"}), _resp(200)])

    resp = await client._request("POST", "/v1/corpora/c/chunks", json={})

    assert resp.status_code == 200
    assert len(fake.calls) == 2
    assert no_async_sleep == [2.0]  # honored the server's Retry-After


async def test_missing_credential_surfaces_as_auth_error():
    def _no_cred() -> str:
        raise RuntimeError("No Castform platform credential available")

    client = CorpusClient(base_url="https://corpora.invalid", token_provider=_no_cred)
    with pytest.raises(AuthenticationError, match="No Castform platform credential"):
        await client._request("GET", "/health")


async def test_request_429_without_header_uses_exponential_backoff(monkeypatch, no_async_sleep):
    client = CorpusClient(
        base_url="http://corpora",
        token_provider=lambda: "tok",
        retry_backoff_seconds=0.5,
    )
    _wire_async(monkeypatch, client, [_resp(429), _resp(200)])

    resp = await client._request("POST", "/x")

    assert resp.status_code == 200
    assert no_async_sleep == [0.5]  # 0.5 * 2**0 fallback when Retry-After absent


async def test_request_network_error_retries_then_surfaces(monkeypatch, no_async_sleep):
    client = CorpusClient(base_url="http://corpora", token_provider=lambda: "tok", max_retries=3)

    class _AlwaysFails:
        is_closed = False

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def request(self, method, path, **kwargs):
            self.calls.append((method, path))
            raise httpx.ConnectError("boom")

    fake = _AlwaysFails()
    monkeypatch.setattr(client, "_get_async_client", lambda: fake)

    with pytest.raises(CorpusAPIError):
        await client._request("POST", "/x")
    assert len(fake.calls) == 3  # attempts 1+2 retried, attempt 3 surfaces
    assert no_async_sleep == [0.5, 1.0]  # exponential backoff between attempts


async def test_request_429_exhausts_retries(monkeypatch, no_async_sleep):
    client = CorpusClient(
        base_url="http://corpora",
        token_provider=lambda: "tok",
        max_retries=3,
    )
    fake = _wire_async(
        monkeypatch,
        client,
        [_resp(429, {"Retry-After": "1"})] * 3,
    )

    response = await client._request("POST", "/x")

    assert response.status_code == 429
    assert len(fake.calls) == 3
    assert no_async_sleep == [1.0, 1.0]
    with pytest.raises(CorpusAPIError):
        client._handle_response_errors(response)


async def test_upload_chunks_uses_bounded_async_concurrency(monkeypatch):
    chunks = ChunkCollection(chunks=[Chunk(content=f"chunk {index}") for index in range(4)])
    client = CorpusClient(base_url="http://corpora", token_provider=lambda: "tok")
    active = 0
    max_active = 0

    async def fake_request(method, path, **kwargs):
        nonlocal active, max_active
        assert method == "POST"
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        ids = [row["id"] for row in kwargs["json"]["chunks"]]
        return httpx.Response(
            200,
            json={"insertedCount": len(ids), "chunkIds": ids},
        )

    monkeypatch.setattr(client, "_request", fake_request)

    result = await client.upload_chunks(
        "corpus-id",
        chunks,
        batch_size=1,
        max_workers=2,
        show_progress=False,
    )

    assert result.inserted_count == 4
    assert len(result.chunk_ids) == 4
    assert max_active == 2


# --- async source retrieval -------------------------------------------------


def _chunk(file: str, index: int, content: str) -> Chunk:
    return Chunk(content=content, metadata=(("file", file), ("index", index)))


def _make_async_source(chunks, search_results_per_call) -> PostgresChunkSource:
    source = PostgresChunkSource.__new__(PostgresChunkSource)
    source.collection = ChunkCollection(chunks=chunks)
    source._corpus = SimpleNamespace(id="test-corpus")
    source._client = MagicMock()
    source._corpus_name = "test"

    call_idx = {"n": 0}

    async def fake_search_with_chunks(**kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx < len(search_results_per_call):
            return search_results_per_call[idx]
        return []

    source._client.search_with_chunks = fake_search_with_chunks
    return source


async def test_search_related_dedups_and_aggregates():
    src = _chunk("a.md", 0, "source")
    dup = _chunk("b.md", 0, "dup content")
    source = _make_async_source(
        chunks=[src, dup],
        search_results_per_call=[[(dup, 0.9)], [(dup, 0.8)]],
    )

    results = await source.search_related(src, ["q1", "q2"], top_k=5)

    assert len(results) == 1
    assert set(results[0]["queries"]) == {"q1", "q2"}
    assert results[0]["max_score"] == 0.9  # max across both query hits


async def test_search_related_skips_source_and_neighbors():
    c0 = _chunk("a.md", 0, "chunk zero")
    c1 = _chunk("a.md", 1, "chunk one (source)")
    c2 = _chunk("a.md", 2, "chunk two")
    c5 = _chunk("a.md", 5, "chunk five (far)")
    source = _make_async_source(
        chunks=[c0, c1, c2, c5],
        search_results_per_call=[[(c0, 0.9), (c2, 0.85), (c5, 0.7)]],
    )

    results = await source.search_related(c1, ["query"], top_k=5)

    # source c1 plus same-file neighbors c0/c2 (index diff 1) are skipped.
    assert len(results) == 1
    assert results[0]["chunk"].content == "chunk five (far)"
