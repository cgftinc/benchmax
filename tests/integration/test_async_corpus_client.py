"""Integration tests for the async Postgres corpus client (asearch / _arequest / HTTP2).

Hits the real Corpora API — requires credentials and a stable, non-empty test corpus.
These exercise the async provider interaction added on ``qa-gen-async-refactor``:
real-``await`` ``_arequest`` over the HTTP/2 client, ``asearch`` query construction +
response parse, and sync/async parity vs ``search``. ``asearch_with_chunks`` /
``asearch_related`` ride the same ``_arequest`` transport, so they are covered
transitively (a dedicated related-path test needs an ingested corpus + local
ChunkCollection fixture — see HARDENING-FOLLOWUPS.md).

Run with:

    PLATFORM_API_KEY=... BENCHMAX_TEST_CORPUS_ID=... CASTFORM_BASE_DOMAIN=castform.dev \
        uv run pytest -m integration tests/integration/test_async_corpus_client.py -v

Credentials / target:
- ``PLATFORM_API_KEY``        — platform API key sent as the Bearer token.
- ``BENCHMAX_TEST_CORPUS_ID`` — id of a stable, non-empty corpus to search.
- base URL derives from ``benchmax.config.platform_url()`` (``CASTFORM_BASE_DOMAIN`` /
  ``CASTFORM_PLATFORM_URL``); defaults to ``https://api.castform.com``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from benchmax import config
from benchmax.rag.corpus.postgres.client import CorpusClient

_api_key = os.environ.get("PLATFORM_API_KEY", "")
_corpus_id = os.environ.get("BENCHMAX_TEST_CORPUS_ID", "")
_base_url = os.environ.get("CASTFORM_CORPORA_URL") or config.platform_url()
_query = os.environ.get("BENCHMAX_TEST_QUERY", "the")

pytestmark = pytest.mark.integration


def _skip_if_no_creds() -> None:
    if not _api_key or not _corpus_id:
        pytest.skip(
            "PLATFORM_API_KEY and BENCHMAX_TEST_CORPUS_ID required for corpus "
            "integration tests"
        )


def _client() -> CorpusClient:
    return CorpusClient(base_url=_base_url, token_provider=lambda: _api_key)


class TestAsyncCorpusClient:
    """E2E tests that send real requests to the Corpora search API."""

    async def test_asearch_returns_results(self):
        _skip_if_no_creds()
        client = _client()
        try:
            result = await client.asearch(_corpus_id, _query, limit=5)
        finally:
            await client.aclose()

        assert result.query == _query
        assert result.total >= 0
        assert len(result.results) <= 5
        for chunk in result.results:
            assert chunk.id
            assert isinstance(chunk.content, str)

    async def test_asearch_matches_sync_search(self):
        """Parity: the async path returns the same ids/order/total as the sync path
        for an identical query (same endpoint, same BM25 backend)."""
        _skip_if_no_creds()
        client = _client()
        try:
            sync_result = client.search(_corpus_id, _query, limit=5)
            async_result = await client.asearch(_corpus_id, _query, limit=5)
        finally:
            await client.aclose()
            client.close()

        assert async_result.total == sync_result.total
        # If the backend tie-breaks equal BM25 scores nondeterministically, relax this
        # to comparing sorted id sets.
        assert [c.id for c in async_result.results] == [
            c.id for c in sync_result.results
        ]

    async def test_asearch_http2_concurrency_no_errors(self):
        """Fire concurrent asearch calls over the shared HTTP/2 client and expect 0
        errors — regression guard for the RemoteProtocolError seen under the old
        sync-per-thread path."""
        _skip_if_no_creds()
        client = _client()
        queries = [f"{_query} {i}" for i in range(10)]
        try:
            results = await asyncio.gather(
                *(client.asearch(_corpus_id, q, limit=3) for q in queries)
            )
        finally:
            await client.aclose()

        assert len(results) == len(queries)
        assert all(r.query == q for r, q in zip(results, queries))
