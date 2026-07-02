"""Integration tests for the async Postgres corpus client (asearch / _arequest / HTTP2).

Hits the real Corpora API using the local castform CLI session (or PLATFORM_API_KEY).
Exercises the async provider interaction added on qa-gen-async-refactor: real-``await``
``_arequest`` over the HTTP/2 client, ``asearch`` query construction + response parse, and
sync/async parity vs ``search``. ``asearch_with_chunks`` / ``asearch_related`` ride the same
``_arequest`` transport, so they are covered transitively.

Run against staging with the local CLI session (after ``castform login``):

    CASTFORM_BASE_DOMAIN=castform.dev \
    CASTFORM_CREDENTIALS_PATH=~/.castform/staging-credentials.json \
        uv run pytest -m integration tests/integration/test_async_corpus_client.py -v

Auth: ``token_provider`` defaults to ``benchmax.platform.credentials.platform_bearer``,
which resolves ``PLATFORM_API_KEY`` / ``ACT_AS_TOKEN_PATH`` or the cached ``~/.castform``
session (session support requires the current ``platform.credentials`` — present on main).
Skips if no creds resolve. Corpus: auto-discovered via ``list_corpora()`` (override with
``BENCHMAX_TEST_CORPUS_ID``; query via ``BENCHMAX_TEST_QUERY``, default ``"the"``).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from benchmax import config
from benchmax.platform.credentials import platform_bearer
from benchmax.rag.corpus.postgres.client import CorpusClient

_base_url = os.environ.get("CASTFORM_CORPORA_URL") or config.platform_url()
_query = os.environ.get("BENCHMAX_TEST_QUERY", "the")

pytestmark = pytest.mark.integration


def _skip_if_no_creds() -> None:
    """Skip unless ``platform_bearer`` resolves a token (env key or local CLI session)."""
    try:
        token = platform_bearer()
    except Exception as exc:  # noqa: BLE001 — any failure means no usable creds
        pytest.skip(f"no platform creds ({exc}); run `castform login`")
    if not token:
        pytest.skip("no platform creds; run `castform login` or set PLATFORM_API_KEY")


def _client() -> CorpusClient:
    return CorpusClient(base_url=_base_url, token_provider=platform_bearer)


def _resolve_corpus_id(client: CorpusClient) -> str | None:
    override = os.environ.get("BENCHMAX_TEST_CORPUS_ID")
    if override:
        return override
    corpora = client.list_corpora()
    return corpora[0].id if corpora else None


class TestAsyncCorpusClient:
    """E2E tests that send real requests to the Corpora search API."""

    async def test_asearch_returns_results(self):
        _skip_if_no_creds()
        client = _client()
        try:
            corpus_id = _resolve_corpus_id(client)
            if not corpus_id:
                pytest.skip("no corpus available to search")
            result = await client.asearch(corpus_id, _query, limit=5)
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
            corpus_id = _resolve_corpus_id(client)
            if not corpus_id:
                pytest.skip("no corpus available to search")
            sync_result = client.search(corpus_id, _query, limit=5)
            async_result = await client.asearch(corpus_id, _query, limit=5)
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
        try:
            corpus_id = _resolve_corpus_id(client)
            if not corpus_id:
                pytest.skip("no corpus available to search")
            queries = [f"{_query} {i}" for i in range(10)]
            results = await asyncio.gather(
                *(client.asearch(corpus_id, q, limit=3) for q in queries)
            )
        finally:
            await client.aclose()

        assert len(results) == len(queries)
        assert all(r.query == q for r, q in zip(results, queries))
