"""Contract #2: separate read vs write DSN provider surfaces.

Env-var names + separation are frozen here (pass); lazy resolution is an xfail
skeleton that must raise NotImplementedError until Slice 4.
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


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 4")
def test_read_provider_resolves_lazily() -> None:
    resolve_read_dsn_provider("postgresql://ro@host/db")


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 4")
def test_write_provider_resolves_lazily() -> None:
    resolve_write_dsn_provider("postgresql://rw@host/db")
