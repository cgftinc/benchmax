"""Contract #1/#7 (B5): version-lifecycle transaction seam + retention.

Retention validation is frozen here (pass). The transaction seam and RRF fusion
are xfail skeletons that must raise NotImplementedError. The transaction test
passes real ``Composable`` statements (not raw strings) and a fake connection
that raises on the SECOND statement, encoding the atomicity contract: the ledger
update and view replacement must roll back together, leaving no partial state.
"""

from __future__ import annotations

import pytest
from psycopg import sql

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.schema import DEFAULT_RETENTION, RetentionPolicy
from castform.rag.corpus.neon.search import fuse_rrf


class _FailingConn:
    """Fake conn that raises on the second statement and records rollback/commit."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.rolled_back = False
        self.committed = False

    def execute(self, statement: sql.Composable) -> None:
        self.executed.append(str(statement))
        if len(self.executed) == 2:
            raise RuntimeError("view swap failed mid-transaction")

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_retention_keeps_rollback_target() -> None:
    assert isinstance(DEFAULT_RETENTION, RetentionPolicy)
    assert DEFAULT_RETENTION.keep_activated >= 2
    assert DEFAULT_RETENTION.keep_ready >= 1


def test_activation_rolls_back_atomically() -> None:
    client = NeonClient(lambda: "postgresql://rw@host/db")
    conn = _FailingConn()
    client._conn = conn  # type: ignore[attr-defined]
    statements = [
        sql.SQL(
            "UPDATE neon_corpus_versions SET is_current = true "
            "WHERE logical_name = {} AND version = {}"
        ).format(sql.Literal("mycorpus"), sql.Literal(2)),
        sql.SQL("CREATE OR REPLACE VIEW {} AS SELECT id FROM {}").format(
            sql.Identifier("mycorpus"), sql.Identifier("mycorpus__v2")
        ),
    ]
    # The failing second statement rolls the whole transaction back and re-raises;
    # nothing is committed, so no partial state (is_current) survives.
    with pytest.raises(RuntimeError):
        client.execute_in_transaction(statements)
    assert conn.rolled_back is True
    assert conn.committed is False


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_fuse_rrf_single_owner() -> None:
    fuse_rrf([["a", "b"], ["b", "c"]])
