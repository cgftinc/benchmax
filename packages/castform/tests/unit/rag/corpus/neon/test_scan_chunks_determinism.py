"""Contract #5: scan_chunks total, pageable, deterministic ordering.

The typed order key is frozen (pass). The iterator drives the REAL
``NeonClient.scan_chunks`` keyset loop (its real ``scan_page_sql`` params) through a
fake paging connection, so multi-page ordering is genuinely exercised — three rows
at ``batch_size=2`` span two pages — and the source maps each row to a Chunk.
"""

from __future__ import annotations

from typing import Any

from fakes.neon import make_neon_source, make_read_row

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.source import SCAN_ORDER_BY


def test_scan_order_key_frozen() -> None:
    # Typed non-null columns, not JSONB extraction.
    assert SCAN_ORDER_BY == ("source_file", "chunk_index", "id")


class _FakePagingCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows
        self.description = [("col",)]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakePagingConn:
    """Keyset-pages READ_COLUMNS rows by the real scan query's bound params."""

    closed = 0
    broken = False

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        # Sort by the frozen (source_file, chunk_index, id) total order.
        self._rows = sorted(rows, key=lambda r: (r[3], r[4], r[0]))

    def execute(self, query: Any, params: dict[str, Any] | None = None):
        params = params or {}
        limit = params["batch_size"]
        after = None
        if "after_file" in params:
            after = (params["after_file"], params["after_index"], params["after_id"])
        start = 0
        if after is not None:
            start = next(
                (i for i, r in enumerate(self._rows) if (r[3], r[4], r[0]) > after),
                len(self._rows),
            )
        return _FakePagingCursor(self._rows[start : start + limit])

    def commit(self) -> None:
        pass


def _paging_client(rows: list[tuple[Any, ...]]) -> NeonClient:
    client = NeonClient.__new__(NeonClient)
    client._conn = _FakePagingConn(rows)
    client._dsn_provider = lambda: ""  # never used: the conn is never dead
    client._vector_registered = False
    return client


def test_scan_chunks_deterministic_across_pages() -> None:
    # Three rows, out of natural order, spanning >1 page at batch_size=2.
    rows = [
        make_read_row("h3", "c3", source_file="b.md", chunk_index=2),
        make_read_row("h2", "c2", source_file="a.md", chunk_index=10),
        make_read_row("h1", "c1", source_file="a.md", chunk_index=2),
    ]
    source = make_neon_source(read_client=_paging_client(rows))
    first = [c.hash for c in source.scan_chunks(batch_size=2)]
    second = [c.hash for c in source.scan_chunks(batch_size=2)]
    assert first == second
    assert first == ["h1", "h2", "h3"]  # (a.md,2) < (a.md,10) < (b.md,2)
