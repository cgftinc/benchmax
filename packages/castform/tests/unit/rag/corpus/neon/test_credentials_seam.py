"""Contract #2: separate read vs write DSN provider surfaces.

Env-var names + separation are frozen here (pass); lazy resolution is an xfail
skeleton filled by Slice 4.
"""

from __future__ import annotations

import pytest

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


@pytest.mark.xfail(reason="DSN resolution built in Slice 4", strict=False)
def test_read_provider_resolves_lazily() -> None:
    provider = resolve_read_dsn_provider("postgresql://ro@host/db")
    assert callable(provider)


@pytest.mark.xfail(reason="DSN resolution built in Slice 4", strict=False)
def test_write_provider_resolves_lazily() -> None:
    provider = resolve_write_dsn_provider("postgresql://rw@host/db")
    assert callable(provider)
