"""Contract #4: multi-query search_related dedup + full 3-tuple ordering.

Covers the sync ``NeonChunkSource.search_related`` and its async twin
``NeonAsyncChunkSource.asearch_related`` (which shares the pure dedup/order/mapping
helpers, so parity here guards against sync/async drift), plus the async-protocol
separation (``asearch_related`` is NOT on the runtime-checkable sync protocol).

Controlled hits are injected — chunk C hit by BOTH queries (ranks 0 and 2), chunk
D by one query — so the full ``(query_count, cross_file, max_score)`` order and the
max-reciprocal-rank dedup (with ``native_score`` from the winning hit) are asserted
unconditionally, no vacuous ``if``.
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from fakes.neon import (
    FakeAsyncQueryRunner,
    FakeQueryRunner,
    make_async_neon_source,
    make_neon_source,
    make_query_row,
)

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.neon.async_source import _AsyncQueryRunner
from castform.rag.corpus.neon.query import NeonQueryRequest


def _chunk(content: str, source_file: str, chunk_index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata=(("file", source_file), ("index", chunk_index)),
    )


# Origin in a.md; C cross-file (b.md) hit by q1@rank0 and q2@rank2; D same-file
# (a.md, far index) hit by q1@rank1 only.
_ORIGIN = _chunk("origin", "a.md", 0)
_C = _chunk("cross-file hit", "b.md", 4)
_D = _chunk("same-file hit", "a.md", 9)


def _rows_by_query() -> dict[str, list]:
    return {
        "q1": [
            make_query_row(
                _C.hash,
                "cross-file hit",
                metadata={"file": "b.md", "index": 4},
                source_file="b.md",
                chunk_index=4,
                surfaced_score=1 / 60,
                native_score=-2.0,
                rank=0,
            ),
            make_query_row(
                _D.hash,
                "same-file hit",
                metadata={"file": "a.md", "index": 9},
                source_file="a.md",
                chunk_index=9,
                surfaced_score=1 / 61,
                native_score=-2.5,
                rank=1,
            ),
        ],
        "q2": [
            make_query_row(
                _C.hash,
                "cross-file hit",
                metadata={"file": "b.md", "index": 4},
                source_file="b.md",
                chunk_index=4,
                surfaced_score=1 / 62,
                native_score=-3.0,
                rank=2,
            ),
        ],
    }


def _assert_dedup_and_order(results: list[dict]) -> None:
    for item in results:
        assert set(item) >= {"chunk", "queries", "same_file", "max_score", "native_score"}

    keys = [(len(r["queries"]), not r["same_file"], r["max_score"]) for r in results]
    assert keys == sorted(keys, reverse=True)

    # C: hit by both queries, dedups to the max reciprocal rank 1/60, ranks first.
    top = results[0]
    assert top["chunk"].hash == _C.hash
    assert top["max_score"] == 1 / 60
    assert len(top["queries"]) == 2
    assert top["same_file"] is False
    # native_score comes from the winning (max-surfaced) hit — q1 rank 0, not q2.
    assert top["native_score"] == -2.0

    # D: single query, same-file, ranks after C.
    assert results[1]["chunk"].hash == _D.hash
    assert results[1]["same_file"] is True


def test_search_related_dedup_and_full_tuple_order():
    source = make_neon_source(search=FakeQueryRunner(rows_by_query=_rows_by_query()))
    _assert_dedup_and_order(source.search_related(_ORIGIN, ["q1", "q2"], top_k=5))


def test_search_related_skips_source_and_adjacent_neighbors():
    origin = _chunk("origin", "a.md", 5)
    rows = {
        "q": [
            make_query_row(origin.hash, source_file="a.md", chunk_index=5),  # source
            make_query_row("adj", source_file="a.md", chunk_index=6),  # neighbor (diff 1)
            make_query_row("far", "far body", source_file="a.md", chunk_index=9),
        ]
    }
    source = make_neon_source(search=FakeQueryRunner(rows_by_query=rows))
    results = source.search_related(origin, ["q"], top_k=5)
    assert [r["chunk"].hash for r in results] == ["far"]


async def test_asearch_related_matches_sync_contract():
    source = make_async_neon_source(
        search=FakeAsyncQueryRunner(rows_by_query=_rows_by_query())
    )
    results = await source.asearch_related(_ORIGIN, ["q1", "q2"], top_k=5)
    _assert_dedup_and_order(results)


def test_async_protocol_is_separate_from_sync():
    from castform.rag.corpus.neon.async_source import (
        AsyncChunkSource,
        NeonAsyncChunkSource,
    )
    from castform.rag.corpus.neon.source import NeonChunkSource
    from castform.rag.corpus.source import ChunkSource

    sync = NeonChunkSource.__new__(NeonChunkSource)
    asynchronous = NeonAsyncChunkSource.__new__(NeonAsyncChunkSource)

    # The sync source conforms to the sync protocol but NOT the async one —
    # asearch_related is deliberately absent from its runtime-checkable surface.
    assert isinstance(sync, ChunkSource)
    assert not isinstance(sync, AsyncChunkSource)
    # The async source conforms to the separate async protocol.
    assert isinstance(asynchronous, AsyncChunkSource)


# ---------------------------------------------------------------------------
# Real _AsyncQueryRunner: per-op connection lifecycle (prepare_threshold + close)
# ---------------------------------------------------------------------------


class _FakeAsyncCursor:
    def __init__(self, rows: list[tuple], description: Any = None) -> None:
        self._rows = rows
        self.description = description

    async def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeAsyncConn:
    """Per-op async connection: statement order is [lock, ledger, leg]."""

    def __init__(self, *, ledger, legs, fail=False) -> None:
        self._ledger = ledger
        self._legs = legs
        self._fail = fail
        self._logical_exec = 0
        self.closed_count = 0

    def transaction(self):
        class _Txn:
            async def __aenter__(self_):
                return self_

            async def __aexit__(self_, *exc):
                return False

        return _Txn()

    async def execute(self, query, params=None):
        params = params or {}
        if self._fail:
            raise RuntimeError("boom")
        if "top_k" in params:  # a retrieval leg
            return _FakeAsyncCursor(self._legs, description=[("c",)])
        self._logical_exec += 1
        if self._logical_exec == 1:  # advisory lock (result unused)
            return _FakeAsyncCursor([])
        return _FakeAsyncCursor(self._ledger, description=[("c",)])  # ledger

    async def close(self):
        self.closed_count += 1


async def _noop_register(_conn) -> None:  # skip pgvector registration in tests
    pass


def _runner() -> _AsyncQueryRunner:
    return _AsyncQueryRunner(
        lambda: "dsn://unused",
        logical_name="c",
        schema="s",
        text_search_config="pg_catalog.english",
    )


async def test_async_runner_uses_no_prepare_and_closes_once_on_success(monkeypatch):
    conn = _FakeAsyncConn(
        ledger=[(1, "activated", True)],
        legs=[("h1", "body", {}, "f.md", 0, -1.0)],
    )
    captured: dict[str, Any] = {}

    async def fake_connect(dsn, **kw):
        captured["dsn"] = dsn
        captured["kw"] = kw
        return conn

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", fake_connect)
    runner = _runner()
    monkeypatch.setattr(runner, "_register_vector", _noop_register)

    rows = await runner.query_rows(NeonQueryRequest(mode="lexical", top_k=5, text="q"))

    assert [r.chunk_id for r in rows] == ["h1"]
    assert captured["kw"]["prepare_threshold"] is None  # no server-side auto-PREPARE
    assert conn.closed_count == 1  # per-op connection closed exactly once


async def test_async_runner_closes_once_on_failure(monkeypatch):
    conn = _FakeAsyncConn(ledger=[], legs=[], fail=True)

    async def fake_connect(dsn, **kw):
        return conn

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", fake_connect)
    runner = _runner()
    monkeypatch.setattr(runner, "_register_vector", _noop_register)

    with pytest.raises(RuntimeError, match="boom"):
        await runner.query_rows(NeonQueryRequest(mode="lexical", top_k=5, text="q"))
    assert conn.closed_count == 1  # closed even when the query body raises
