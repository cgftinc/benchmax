"""Contract #5: scan_chunks total, pageable, deterministic, snapshot-consistent.

The typed order key is frozen (pass). The iterator drives the REAL
``NeonClient.scan_in_snapshot`` — one transaction on a dedicated connection holding
the shared advisory lock — via a fake ``psycopg.connect`` that pins a snapshot at
transaction begin. So we prove BOTH multi-page ordering (three rows at
``batch_size`` spanning two pages) AND that a concurrent activation mid-scan cannot
inject rows from a different version.
"""

from __future__ import annotations

from typing import Any

import psycopg
from fakes.neon import make_neon_source, make_read_row

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.source import SCAN_ORDER_BY


def test_scan_order_key_frozen() -> None:
    # Typed non-null columns, not JSONB extraction.
    assert SCAN_ORDER_BY == ("source_file", "chunk_index", "id")


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.description = [("col",)]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeScanConn:
    """A dedicated scan connection: pins a version snapshot at transaction begin.

    Models the correctness guarantee — a concurrent ``activate`` swaps the LIVE
    current version, but a real shared advisory lock would block it until the scan
    transaction ends, so the pinned snapshot (and every page read from it) stays on
    one version. Statement order inside the transaction is [lock, ledger, pages...].
    """

    closed = 0
    broken = False

    def __init__(self, versions: dict[int, list[tuple[Any, ...]]], current: int = 1):
        self._versions = {
            v: sorted(rows, key=lambda r: (r[3], r[4], r[0]))
            for v, rows in versions.items()
        }
        self._current = current
        self._pinned: list[tuple[Any, ...]] | None = None
        self._pinned_current: int | None = None
        self._txn_exec = 0
        self.transactions_entered = 0
        self.locked = False
        self.closed_count = 0

    def transaction(self):
        outer = self

        class _Txn:
            def __enter__(self):
                outer.transactions_entered += 1
                outer._pinned = list(outer._versions[outer._current])
                outer._pinned_current = outer._current
                outer._txn_exec = 0
                return self

            def __exit__(self, *exc):
                outer._pinned = None
                return False

        return _Txn()

    def activate(self, version: int) -> None:
        # Concurrent activation swaps the live current; the shared lock a real scan
        # holds would block this until the scan txn ends, so _pinned is unaffected.
        self._current = version

    def execute(self, query: Any, params: dict[str, Any] | None = None) -> _Cursor:
        params = params or {}
        if "batch_size" in params:
            return _Cursor(self._page(params))
        self._txn_exec += 1
        if self._txn_exec == 1:  # advisory lock
            assert self._pinned is not None, "lock must be acquired inside the scan txn"
            self.locked = True
            return _Cursor([])
        # ledger resolution (2nd statement): is_current reflects the pinned version.
        return _Cursor(
            [(v, "activated", v == self._pinned_current) for v in sorted(self._versions)]
        )

    def _page(self, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        rows = self._pinned or []
        after = None
        if "after_file" in params:
            after = (params["after_file"], params["after_index"], params["after_id"])
        start = 0
        if after is not None:
            start = next(
                (i for i, r in enumerate(rows) if (r[3], r[4], r[0]) > after), len(rows)
            )
        return rows[start : start + params["batch_size"]]

    def close(self) -> None:
        self.closed_count += 1


def _scan_client(conn: _FakeScanConn, monkeypatch) -> NeonClient:
    monkeypatch.setattr(psycopg, "connect", lambda dsn, **kw: conn)
    client = NeonClient.__new__(NeonClient)
    client._dsn_provider = lambda: "dsn://unused"
    client._vector_registered = False
    return client


def test_scan_chunks_deterministic_across_pages(monkeypatch) -> None:
    # Three rows, out of natural order, spanning >1 page at batch_size=1.
    rows = [
        make_read_row("h3", "c3", source_file="b.md", chunk_index=2),
        make_read_row("h2", "c2", source_file="a.md", chunk_index=10),
        make_read_row("h1", "c1", source_file="a.md", chunk_index=2),
    ]
    conn = _FakeScanConn({1: rows}, current=1)
    source = make_neon_source(read_client=_scan_client(conn, monkeypatch))
    first = [c.hash for c in source.scan_chunks(batch_size=1)]
    second = [c.hash for c in source.scan_chunks(batch_size=1)]
    assert first == second
    assert first == ["h1", "h2", "h3"]  # (a.md,2) < (a.md,10) < (b.md,2)


def test_scan_holds_snapshot_against_concurrent_activation(monkeypatch) -> None:
    v1 = [
        make_read_row("a", "v1-a", source_file="a.md", chunk_index=0),
        make_read_row("b", "v1-b", source_file="a.md", chunk_index=1),
    ]
    v2 = [make_read_row("z", "v2-z", source_file="a.md", chunk_index=0)]
    conn = _FakeScanConn({1: v1, 2: v2}, current=1)
    source = make_neon_source(read_client=_scan_client(conn, monkeypatch))

    gen = source.scan_chunks(batch_size=1)
    first = next(gen)  # reads v1 'a' inside the open, locked transaction
    conn.activate(2)  # a concurrent activation swaps the live current to v2
    rest = [c.hash for c in gen]  # continues within the SAME pinned snapshot

    assert first.hash == "a"
    assert [first.hash] + rest == ["a", "b"]  # only v1 rows; v2's 'z' never injected
    assert conn.transactions_entered == 1  # ONE transaction for the whole iterator
    assert conn.locked  # shared advisory lock acquired inside that transaction
    assert conn.closed_count == 1  # dedicated connection closed exactly once
