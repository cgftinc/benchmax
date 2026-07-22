"""Neon connection + SQL execution seam.

Contract-freeze artifact (Slice A). Signatures only; psycopg and pgvector are
imported lazily inside methods (never at module load) so this module — and the
whole ``neon`` package — imports without the ``neon`` extra installed, matching
the pickle-safe, lazy-import discipline of ``turbopuffer/search.py`` and
``corpus/embed.py``.
"""

from __future__ import annotations

from typing import Any

from castform.platform.credentials import TokenProvider


class NeonClient:
    """Thin connection + SQL-execution wrapper over a resolved Neon DSN.

    Args:
        dsn_provider: Callable resolving a DSN string per connection. The read
            path passes an RO provider, the ingest path an RW provider.
    """

    def __init__(self, dsn_provider: TokenProvider) -> None:
        self._dsn_provider = dsn_provider
        self._conn: Any = None

    def _connect(self) -> Any:
        """Open (or reuse) a psycopg connection, registering pgvector adapters.

        Design-lock stub: connection + adapter registration land in Slice 4.
        The real body imports ``psycopg`` and ``pgvector.psycopg`` lazily here.
        """
        raise NotImplementedError("Neon connection is built in Slice 4")

    def execute(
        self, sql: str, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute a parameterized statement and return rows.

        Design-lock stub: built in Slice 4.
        """
        raise NotImplementedError("Neon SQL execution is built in Slice 4")
