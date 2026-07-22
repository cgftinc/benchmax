"""Contract #2: separate read vs write DSN provider surfaces.

Env-var names + separation are frozen; Slice 4 wires the lazy resolution
(``as_token_provider``/``env_token``): the default seam reads the env at call
time (nothing baked), the read and write surfaces default to distinct env vars.
"""

from __future__ import annotations

from castform.rag.corpus.neon.credentials import (
    READ_DSN_ENV_VAR,
    WRITE_DSN_ENV_VAR,
    resolve_read_dsn_provider,
    resolve_write_dsn_provider,
)


def test_read_and_write_env_vars_distinct() -> None:
    assert READ_DSN_ENV_VAR == "NEON_CORPUS_DSN_RO"
    assert WRITE_DSN_ENV_VAR == "NEON_CORPUS_DSN_RW"
    assert READ_DSN_ENV_VAR != WRITE_DSN_ENV_VAR


def test_read_provider_resolves_lazily(monkeypatch) -> None:
    # Default seam: nothing baked, the RO DSN is read from the env per call.
    provider = resolve_read_dsn_provider()
    monkeypatch.setenv(READ_DSN_ENV_VAR, "postgresql://ro@host/db")
    assert provider() == "postgresql://ro@host/db"


def test_write_provider_defaults_to_write_env(monkeypatch) -> None:
    provider = resolve_write_dsn_provider()
    monkeypatch.setenv(WRITE_DSN_ENV_VAR, "postgresql://rw@host/db")
    assert provider() == "postgresql://rw@host/db"
