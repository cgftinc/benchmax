"""Neon connection + SQL execution seam.

Contract-freeze artifact (Slice A). Signatures only; psycopg and pgvector are
imported lazily inside methods (never at module load) so this module — and the
whole ``neon`` package — imports without the ``neon`` extra installed, matching
the pickle-safe, lazy-import discipline of ``turbopuffer/search.py`` and
``corpus/embed.py``.

The execute seam accepts **composable SQL** (``psycopg.sql.Composable``), never a
raw ``str`` (B4): every identifier reaches the driver as ``sql.Identifier`` and
every literal as a bound parameter or ``sql.Literal``, so dynamic table/view/index
names cannot be injection vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from castform.platform.credentials import TokenProvider

if TYPE_CHECKING:
    from psycopg import sql


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
        self, query: sql.Composable, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute one composable statement with bound params and return rows.

        Design-lock stub: built in Slice 4.
        """
        raise NotImplementedError("Neon SQL execution is built in Slice 4")

    def execute_in_transaction(self, statements: list[sql.Composable]) -> None:
        """Run *statements* as one all-or-nothing transaction (B5).

        Used to publish a version so the ledger update and the ``CREATE OR
        REPLACE VIEW`` commit or roll back together. Statements run on the shared
        connection; on the first failure the whole transaction is rolled back and
        the error re-raised, leaving no partial state. The *orchestration* is real
        (and unit-testable against an injected connection); only opening the
        connection (``_connect``) is the Slice 4 stub.
        """
        conn = self._conn if self._conn is not None else self._connect()
        try:
            for statement in statements:
                conn.execute(statement)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
