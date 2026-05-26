"""Tests for CorpusClient._request rate-limit (429) backoff.

A bulk corpus upload runs many batch POSTs in parallel, so the Corpora API's
per-user rate limit returns 429s mid-run. The client must honor Retry-After and
retry within its budget rather than raising and aborting the whole upload.
"""

from __future__ import annotations

import httpx
import pytest

from benchmax.rag.corpus.postgres.client import CorpusClient
from benchmax.rag.corpus.postgres.exceptions import CorpusAPIError


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, json={"ok": True})


def _sequenced_request(monkeypatch, client: CorpusClient, responses: list[httpx.Response]):
    """Wire client._http_client.request to return `responses` in order and
    record how many times it was called."""
    calls: list[tuple[str, str]] = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return responses[len(calls) - 1]

    monkeypatch.setattr(client._http_client, "request", fake_request)
    return calls


@pytest.fixture
def no_sleep(monkeypatch):
    """Capture sleep durations instead of actually waiting."""
    slept: list[float] = []
    monkeypatch.setattr(
        "benchmax.rag.corpus.postgres.client.time.sleep",
        lambda d: slept.append(d),
    )
    return slept


def test_retries_on_429_then_succeeds(monkeypatch, no_sleep):
    client = CorpusClient(base_url="http://corpora", token_provider=lambda: "tok")
    calls = _sequenced_request(
        monkeypatch,
        client,
        [_resp(429, {"Retry-After": "2"}), _resp(200)],
    )

    resp = client._request("POST", "/v1/corpora/c/chunks", json={})

    assert resp.status_code == 200
    assert len(calls) == 2
    assert no_sleep == [2.0]  # honored the server's Retry-After


def test_429_without_header_uses_exponential_backoff(monkeypatch, no_sleep):
    client = CorpusClient(
        base_url="http://corpora",
        token_provider=lambda: "tok",
        retry_backoff_seconds=0.5,
    )
    _sequenced_request(monkeypatch, client, [_resp(429), _resp(200)])

    resp = client._request("POST", "/x")

    assert resp.status_code == 200
    assert no_sleep == [0.5]  # 0.5 * 2**0 fallback when Retry-After is absent


def test_429_exhausts_retries_then_surfaces_error(monkeypatch, no_sleep):
    client = CorpusClient(
        base_url="http://corpora",
        token_provider=lambda: "tok",
        max_retries=3,
    )
    calls = _sequenced_request(
        monkeypatch,
        client,
        [_resp(429, {"Retry-After": "1"})] * 3,
    )

    resp = client._request("POST", "/x")

    # Budget exhausted: the final 429 is returned and converts to CorpusAPIError.
    assert resp.status_code == 429
    assert len(calls) == 3  # attempts 1 and 2 retried; attempt 3 returned
    with pytest.raises(CorpusAPIError):
        client._handle_response_errors(resp)
