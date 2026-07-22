"""Contract #1/#7 (B5): version-lifecycle transaction seam + retention.

Retention validation and the atomic-activation rollback are covered here: the
transaction test passes real ``Composable`` statements (not raw strings) and a
fake connection that raises on the SECOND statement, so the ledger update and
view replacement must roll back together, leaving no partial state. Slice 4 adds
``read_in_snapshot`` (shared advisory lock + SET LOCAL + work in one read txn) and
implements ``fuse_rrf`` (RRF with a deterministic chunk_id tie-break), both
asserted below.
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


class _ReadConn:
    """Fake conn recording statements; the last execute returns a row cursor."""

    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.statements: list[str] = []
        self.committed = False

    def execute(self, statement: sql.Composable, params=None):
        self.statements.append(str(statement))

        class _Cur:
            description = [("col",)]

            def fetchall(_self):
                return self.rows

        return _Cur()

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:  # pragma: no cover - not hit on the happy path
        pass


def test_read_in_snapshot_locks_then_runs_work_in_one_txn() -> None:
    client = NeonClient(lambda: "postgresql://ro@host/db")
    conn = _ReadConn([("h1",), ("h2",)])
    client._conn = conn  # type: ignore[attr-defined]

    def work(c):
        return c.execute(sql.SQL("SELECT id FROM v"), {"top_k": 5}).fetchall()

    rows = client.read_in_snapshot(
        "mycorpus",
        work,
        session_setup=[sql.SQL("SET LOCAL lakebase_bm25.prefilter = on")],
    )
    assert rows == [("h1",), ("h2",)]
    # Shared advisory lock FIRST, then the SET LOCAL, then work's SELECT — one txn.
    assert "pg_advisory_xact_lock_shared" in conn.statements[0]
    assert "SET LOCAL lakebase_bm25.prefilter = on" in conn.statements[1]
    assert "SELECT id FROM v" in conn.statements[2]
    assert conn.committed is True


def test_read_in_snapshot_rejects_raw_string_setup() -> None:
    client = NeonClient(lambda: "postgresql://ro@host/db")
    with pytest.raises(TypeError):
        client.read_in_snapshot(
            "c", lambda conn: None, session_setup=["SET x"]  # type: ignore[list-item]
        )


def test_fuse_rrf_single_owner() -> None:
    # b is hit by both lists (rank 1 then rank 0), a and c once each. Scores:
    # b = 1/61 + 1/60, a = 1/60, c = 1/61 -> order b, a, c (a > c since 1/60 > 1/61).
    fused = fuse_rrf([["a", "b"], ["b", "c"]])
    assert [cid for cid, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == 1 / 61 + 1 / 60


def test_fuse_rrf_tie_break_is_chunk_id_ascending() -> None:
    # Equal single-hit scores must break deterministically on chunk_id ascending.
    fused = fuse_rrf([["z", "a"], []])
    assert [cid for cid, _ in fused] == ["z", "a"]  # z rank0 > a rank1
    same = fuse_rrf([["b"], ["a"]])  # both rank 0 -> tie -> id asc
    assert [cid for cid, _ in same] == ["a", "b"]
