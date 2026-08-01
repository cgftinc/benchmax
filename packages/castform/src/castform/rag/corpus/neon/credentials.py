"""Neon DSN credential seam — separate read and write provider surfaces.

Contract-freeze artifact (Slice A). Signatures only; lazy resolution and the
psycopg connection are wired in Slice 4.

The Neon connection string (DSN) rides the same ``str | TokenProvider | None``
seam turbopuffer uses for its API key (see ``rag/corpus/turbopuffer/search.py``
and ``platform/credentials.py``): ``None`` reads the DSN from the environment at
runtime (self-serve path), a ``str`` bakes a literal DSN into the pickled env
(platform-orchestrated path), and a callable is invoked per connection.

Read and write are deliberately *separate* provider surfaces. A single provider
cannot be both the RW-ingest role (DDL + INSERT into version tables) and the
RO-search role (SELECT only, no DDL/DML) — the RO grant is what the sandbox
rollout env is handed, so it cannot mutate the corpus even if compromised.
"""

from __future__ import annotations

from castform.platform.credentials import TokenProvider, as_token_provider, env_token

READ_DSN_ENV_VAR = "NEON_CORPUS_DSN_RO"
"""Env var holding the read-only (search) DSN. RO grant: SELECT only."""

WRITE_DSN_ENV_VAR = "NEON_CORPUS_DSN_RW"
"""Env var holding the read-write (ingest) DSN. RW grant: DDL + DML."""


def resolve_read_dsn_provider(
    dsn_provider: str | TokenProvider | None = None,
) -> TokenProvider:
    """Return the lazily-resolved read-only DSN provider for search.

    Mirrors ``as_token_provider(value, env_token(READ_DSN_ENV_VAR))``. The
    returned callable yields a DSN string carrying only the RO grant.
    Design-lock stub: the resolution wiring lands in Slice 4.
    """
    raise NotImplementedError("read DSN resolution is built in Slice 4")


def resolve_write_dsn_provider(
    dsn_provider: str | TokenProvider | None = None,
) -> TokenProvider:
    """Return the lazily-resolved read-write DSN provider for ingest.

    Mirrors ``as_token_provider(value, env_token(WRITE_DSN_ENV_VAR))``. The
    returned callable yields a DSN string carrying the RW grant.
    Design-lock stub: the resolution wiring lands in Slice 4.
    """
    raise NotImplementedError("write DSN resolution is built in Slice 4")


__all__ = [
    "READ_DSN_ENV_VAR",
    "WRITE_DSN_ENV_VAR",
    "TokenProvider",
    "as_token_provider",
    "env_token",
    "resolve_read_dsn_provider",
    "resolve_write_dsn_provider",
]
