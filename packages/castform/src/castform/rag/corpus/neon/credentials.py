"""Neon DSN credential seam — separate read and write provider surfaces.

The lazy resolution is wired in Slice 4: each ``resolve_*`` returns a per-call
provider via ``as_token_provider(dsn_provider, env_token(<VAR>))`` — the DSN is
resolved on connect (or read from the env), never at construction.

The Neon connection string (DSN) rides the same ``str | TokenProvider | None``
seam turbopuffer uses for its API key (see ``rag/corpus/turbopuffer/search.py``
and ``platform/credentials.py``): ``None`` reads the DSN from the environment at
runtime (self-serve path), a ``str`` bakes a literal DSN into the pickled env
(platform-orchestrated path), and a callable is invoked per connection.

Read and write are deliberately *separate* provider surfaces. A single provider
cannot be both the RW-ingest role (DDL + INSERT into version tables) and the
RO-search role (SELECT only, no DDL/DML) — the RO grant is what the sandbox
rollout env is handed, so it cannot mutate the corpus even if compromised.

The DSN resolves *only* from these explicit vars — a generic ``DATABASE_URL``
never satisfies a Neon DSN (NB5). Neon is DDL-capable, so a broad fallback could
silently point ingest/provisioning at an unintended database; the seam has no
fallback chain, and ``env_token`` raises when the var is unset.
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

    ``None`` yields the runtime seam ``env_token(READ_DSN_ENV_VAR)`` (the DSN is
    read from the environment per connection, nothing baked); a ``str`` bakes a
    literal RO DSN into the closure (the platform-orchestrated path, since a Ray
    actor can't read the env at runtime); a callable is used as-is. The returned
    provider yields a DSN string carrying only the RO grant (SELECT).
    """
    return as_token_provider(dsn_provider, env_token(READ_DSN_ENV_VAR))


def resolve_write_dsn_provider(
    dsn_provider: str | TokenProvider | None = None,
) -> TokenProvider:
    """Return the lazily-resolved read-write DSN provider for ingest.

    Same resolution rules as :func:`resolve_read_dsn_provider`, but the default
    seam is ``env_token(WRITE_DSN_ENV_VAR)`` and the yielded DSN carries the RW
    grant (DDL + DML). Kept a separate surface so a search-only handle can never
    be handed the ingest role.
    """
    return as_token_provider(dsn_provider, env_token(WRITE_DSN_ENV_VAR))


__all__ = [
    "READ_DSN_ENV_VAR",
    "WRITE_DSN_ENV_VAR",
    "TokenProvider",
    "as_token_provider",
    "env_token",
    "resolve_read_dsn_provider",
    "resolve_write_dsn_provider",
]
