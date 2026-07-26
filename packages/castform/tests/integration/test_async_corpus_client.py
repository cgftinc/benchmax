"""Integration tests for the async-only Postgres corpus client.

Hits the real Corpora API using the local castform CLI session (or CASTFORM_API_KEY).
Exercises real awaited requests over the HTTP/2 client, search request/response
handling, and concurrent retrieval through the sole public async API.

Run against staging with the local CLI session (after ``castform login``):

    CASTFORM_BASE_DOMAIN=castform.dev \
    CASTFORM_CREDENTIALS_PATH=~/.castform/staging-credentials.json \
        uv run pytest -m integration tests/integration/test_async_corpus_client.py -v

Auth: ``token_provider`` defaults to ``runtime_platform_bearer``, which resolves
``CASTFORM_API_KEY`` / ``ACT_AS_TOKEN_PATH`` or the cached ``~/.castform``
session (session support requires the current ``platform.credentials`` — present on main).
Skips if no creds resolve. Corpus: auto-discovered via ``list_corpora()`` (override with
``BENCHMAX_TEST_CORPUS_ID``; query via ``BENCHMAX_TEST_QUERY``, default ``"the"``).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from castform import config
from castform.platform.credentials import runtime_platform_bearer
from castform.rag.corpus.postgres.client import CorpusClient

_base_url = os.environ.get("CASTFORM_CORPORA_URL") or config.platform_url()
_query = os.environ.get("BENCHMAX_TEST_QUERY", "the")

pytestmark = pytest.mark.integration


def _skip_if_no_creds() -> None:
    """Skip unless the runtime bearer resolves an env key or local CLI session."""
    try:
        token = runtime_platform_bearer()
    except Exception as exc:  # noqa: BLE001 — any failure means no usable creds
        pytest.skip(f"no platform creds ({exc}); run `castform login`")
    if not token:
        pytest.skip("no platform creds; run `castform login` or set CASTFORM_API_KEY")


def _client() -> CorpusClient:
    return CorpusClient(base_url=_base_url, token_provider=runtime_platform_bearer)


async def _resolve_corpus_id(client: CorpusClient) -> str | None:
    override = os.environ.get("BENCHMAX_TEST_CORPUS_ID")
    if override:
        return override
    corpora = await client.list_corpora()
    return corpora[0].id if corpora else None


class TestAsyncCorpusClient:
    """E2E tests that send real requests to the Corpora search API."""

    async def test_asearch_returns_results(self):
        _skip_if_no_creds()
        client = _client()
        try:
            corpus_id = await _resolve_corpus_id(client)
            if not corpus_id:
                pytest.skip("no corpus available to search")
            result = await client.search(corpus_id, _query, limit=5)
        finally:
            await client.close()

        assert result.query == _query
        assert result.total >= 0
        assert len(result.results) <= 5
        for chunk in result.results:
            assert chunk.id
            assert isinstance(chunk.content, str)

    async def test_search_http2_concurrency_no_errors(self):
        """Fire concurrent search calls over the shared HTTP/2 client and expect 0
        errors — regression guard for the RemoteProtocolError seen under the old
        sync-per-thread path."""
        _skip_if_no_creds()
        client = _client()
        try:
            corpus_id = await _resolve_corpus_id(client)
            if not corpus_id:
                pytest.skip("no corpus available to search")
            queries = [f"{_query} {i}" for i in range(10)]
            results = await asyncio.gather(
                *(client.search(corpus_id, q, limit=3) for q in queries)
            )
        finally:
            await client.close()

        assert len(results) == len(queries)
        assert all(r.query == q for r, q in zip(results, queries))
