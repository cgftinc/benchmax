"""Async twins of the corpus client/search unit tests.

The sync seams (`_request` + `time.sleep`, `search_related`) are covered in
``test_client_retry.py`` / ``test_search_related.py`` and stay there. These
exercise the Step-2 async paths: ``_arequest`` retry/backoff and
``asearch_related`` (which shares ``_accumulate_related``/``_sorted_related``
with the sync path, so parity here guards against sync/async drift).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from benchmax.rag.chunkers.models import Chunk, ChunkCollection
from benchmax.rag.corpus.postgres.client import CorpusClient
from benchmax.rag.corpus.postgres.exceptions import CorpusAPIError
from benchmax.rag.corpus.postgres.source import PostgresChunkSource


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, json={"ok": True})


class _FakeAsyncClient:
    """Returns queued responses in order and records calls — stands in for the
    lazily-built ``httpx.AsyncClient`` so ``_arequest`` runs without a network."""

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

    monkeypatch.setattr("benchmax.rag.corpus.postgres.client.asyncio.sleep", _sleep)
    return slept


def _wire_async(monkeypatch, client: CorpusClient, responses) -> _FakeAsyncClient:
    fake = _FakeAsyncClient(responses)
    monkeypatch.setattr(client, "_get_async_client", lambda: fake)
    return fake


async def test_arequest_retries_on_429_then_succeeds(monkeypatch, no_async_sleep):
    client = CorpusClient(base_url="http://corpora", token_provider=lambda: "tok")
    fake = _wire_async(
        monkeypatch, client, [_resp(429, {"Retry-After": "2"}), _resp(200)]
    )

    resp = await client._arequest("POST", "/v1/corpora/c/chunks", json={})

    assert resp.status_code == 200
    assert len(fake.calls) == 2
    assert no_async_sleep == [2.0]  # honored the server's Retry-After


async def test_arequest_429_without_header_uses_exponential_backoff(
    monkeypatch, no_async_sleep
):
    client = CorpusClient(
        base_url="http://corpora", token_provider=lambda: "tok", retry_backoff_seconds=0.5
    )
    _wire_async(monkeypatch, client, [_resp(429), _resp(200)])

    resp = await client._arequest("POST", "/x")

    assert resp.status_code == 200
    assert no_async_sleep == [0.5]  # 0.5 * 2**0 fallback when Retry-After absent


async def test_arequest_network_error_retries_then_surfaces(monkeypatch, no_async_sleep):
    client = CorpusClient(
        base_url="http://corpora", token_provider=lambda: "tok", max_retries=3
    )

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
        await client._arequest("POST", "/x")
    assert len(fake.calls) == 3  # attempts 1+2 retried, attempt 3 surfaces
    assert no_async_sleep == [0.5, 1.0]  # exponential backoff between attempts


# --- asearch_related parity (shared aggregation with the sync path) ----------


def _chunk(file: str, index: int, content: str) -> Chunk:
    return Chunk(content=content, metadata=(("file", file), ("index", index)))


def _make_async_source(chunks, asearch_results_per_call) -> PostgresChunkSource:
    source = PostgresChunkSource.__new__(PostgresChunkSource)
    source.collection = ChunkCollection(chunks=chunks)
    source._corpus = SimpleNamespace(id="test-corpus")
    source._client = MagicMock()
    source._corpus_name = "test"

    call_idx = {"n": 0}

    async def fake_asearch_with_chunks(**kwargs):
        idx = call_idx["n"]
        call_idx["n"] += 1
        if idx < len(asearch_results_per_call):
            return asearch_results_per_call[idx]
        return []

    source._client.asearch_with_chunks = fake_asearch_with_chunks
    return source


async def test_asearch_related_dedups_and_aggregates_like_sync():
    src = _chunk("a.md", 0, "source")
    dup = _chunk("b.md", 0, "dup content")
    source = _make_async_source(
        chunks=[src, dup],
        asearch_results_per_call=[[(dup, 0.9)], [(dup, 0.8)]],
    )

    results = await source.asearch_related(src, ["q1", "q2"], top_k=5)

    assert len(results) == 1
    assert set(results[0]["queries"]) == {"q1", "q2"}
    assert results[0]["max_score"] == 0.9  # max across both query hits


async def test_asearch_related_skips_source_and_neighbors():
    c0 = _chunk("a.md", 0, "chunk zero")
    c1 = _chunk("a.md", 1, "chunk one (source)")
    c2 = _chunk("a.md", 2, "chunk two")
    c5 = _chunk("a.md", 5, "chunk five (far)")
    source = _make_async_source(
        chunks=[c0, c1, c2, c5],
        asearch_results_per_call=[[(c0, 0.9), (c2, 0.85), (c5, 0.7)]],
    )

    results = await source.asearch_related(c1, ["query"], top_k=5)

    # source c1 plus same-file neighbors c0/c2 (index diff 1) are skipped.
    assert len(results) == 1
    assert results[0]["chunk"].content == "chunk five (far)"
