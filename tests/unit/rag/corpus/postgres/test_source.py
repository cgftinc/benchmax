"""Tests for PostgresChunkSource.__init__ — keyless / optional base_url wiring."""

from __future__ import annotations

from benchmax import config
from benchmax.platform.credentials import platform_bearer
from benchmax.rag.corpus.postgres.source import PostgresChunkSource


def test_keyless_construction_uses_seam_and_default_base_url() -> None:
    """corpus_name alone → bearer resolves via the seam, base_url from config."""
    source = PostgresChunkSource(corpus_name="my-docs")

    assert source._client is None
    client = source._get_client()
    # No static key baked: the client resolves the bearer per request via the
    # platform credential seam.
    assert client.token_provider is platform_bearer
    # base_url falls back to the session-derived platform URL.
    assert client.base_url == config.platform_url()
    assert source._corpus_name == "my-docs"


def test_explicit_key_and_url_override() -> None:
    """An explicit key wins (fixed-value provider) and base_url is honored."""
    source = PostgresChunkSource(
        corpus_name="my-docs",
        api_key="sk_explicit",
        base_url="https://corpora.example.test",
    )

    client = source._get_client()
    assert client.token_provider() == "sk_explicit"
    assert client.base_url == "https://corpora.example.test"
