"""Unit tests for the crash-resumable streaming ingest (fake backend, no DB)."""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from castform.rag.chunkers.models import Chunk  # noqa: E402
from castform.rag.corpus.neon.schema import NeonTableSpec  # noqa: E402

from build_corpus import (  # noqa: E402
    NeonIngestError,
    _build_and_activate,
    _finalize_version,
    _resolve_build_version,
    _stream_insert,
)

_SPEC = NeonTableSpec("t", 1)


def _chunk(i: int) -> Chunk:
    return Chunk(
        content=f"chunk content number {i}", metadata=(("index", i),), hash=f"{i:064x}"
    )


class _CountingEmbed:
    """Records every text it embeds so a test can prove skipped chunks are not re-embedded."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[0.0] * 4 for _ in texts]


class _FakeCursor:
    def __init__(self, store: set[str]) -> None:
        self._store = store

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def executemany(self, _sql: object, rows: list[tuple]) -> None:
        for row in rows:  # id is row[0]; a set models ON CONFLICT (id) DO NOTHING
            self._store.add(row[0])


class _FakeConn:
    def __init__(self, store: set[str]) -> None:
        self._store = store
        self.autocommit = False
        self.commits = 0
        self.index_stmts: list[object] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._store)

    def commit(self) -> None:
        self.commits += 1

    def execute(self, stmt: object, _params: object = None) -> None:
        self.index_stmts.append(stmt)


class _FakeClient:
    """In-memory stand-in for NeonClient covering only the ingest seam."""

    def __init__(
        self, present: set[str] | None = None, ledger: list[tuple] | None = None
    ) -> None:
        self.store: set[str] = set(present or set())
        self._ledger = ledger or []
        self.conn = _FakeConn(self.store)
        self.ready = False
        self.vacuumed = False
        self.activated = False
        self.table_created = bool(present)  # rows present => table already exists
        self.created_table_calls = 0
        self.allocated: list[int] = []

    def read_ledger_sql(self) -> str:
        return "LEDGER"

    def _live_conn(self) -> _FakeConn:
        return self.conn

    def create_extensions_sql(self) -> list[str]:
        return []

    def register_vector_types(self) -> None:
        pass

    def create_ledger_sql(self) -> list[str]:
        return []

    def allocate_version(self, spec: object) -> None:
        self.allocated.append(spec.version)

    def create_table_sql(self, _spec: object) -> str:
        return "CREATE TABLE t"

    def activate(self, _spec: object, _grant: object) -> None:
        self.activated = True

    def create_ann_index_sql(self, _spec: object) -> str:
        return "ANN"

    def create_bm25_index_sql(self, _spec: object) -> str:
        return "BM25"

    def create_aux_indexes_sql(self, _spec: object) -> list[str]:
        return ["GIN", "SCAN", "TSV"]

    def vacuum(self, _spec: object) -> None:
        self.vacuumed = True

    def mark_ready_sql(self, _spec: object) -> str:
        return "READY"

    def execute(self, query: object, _params: object = None) -> list[tuple]:
        s = query if isinstance(query, str) else query.as_string(None)
        if s == "LEDGER":
            return list(self._ledger)
        if s == "READY":
            self.ready = True
            return []
        if "CREATE TABLE" in s:
            self.table_created = True
            self.created_table_calls += 1
            return []
        if "information_schema.tables" in s:
            return [(1,)] if self.table_created else []
        if "pg_indexes" in s:
            return []  # no indexes exist yet -> finalize creates them
        if "SELECT id FROM" in s:
            return [(cid,) for cid in self.store]
        return []


def test_fresh_build_inserts_all_and_finalizes() -> None:
    client = _FakeClient()
    chunks = [_chunk(i) for i in range(10)]
    emb = _CountingEmbed()
    _stream_insert(client, _SPEC, chunks, emb, batch_size=3)
    assert client.store == {c.hash for c in chunks}
    assert len(emb.seen) == 10
    assert client.conn.commits == ceil(10 / 3)  # one commit per batch
    _finalize_version(client, _SPEC, expected=10)
    assert client.ready and client.vacuumed


def test_resume_skips_present_and_only_embeds_missing() -> None:
    chunks = [_chunk(i) for i in range(10)]
    present = {chunks[i].hash for i in range(6)}  # first 6 already committed
    client = _FakeClient(present=present)
    emb = _CountingEmbed()
    _stream_insert(client, _SPEC, chunks, emb, batch_size=3)
    assert client.store == {c.hash for c in chunks}
    # only the 4 missing chunks were embedded — no re-embedding of paid-for rows
    assert len(emb.seen) == 4
    assert set(emb.seen) == {chunks[i].content for i in range(6, 10)}


def test_partial_corpus_is_not_finalized() -> None:
    client = _FakeClient()
    chunks = [_chunk(i) for i in range(10)]
    _stream_insert(client, _SPEC, chunks, _CountingEmbed(), batch_size=5)
    with pytest.raises(NeonIngestError, match="refusing to finalize"):
        _finalize_version(client, _SPEC, expected=15)  # expects more than present
    assert not client.ready


def test_resolve_build_version_classifies_ledger_state() -> None:
    resume = _FakeClient(ledger=[(1, "activated", False), (2, "building", False)])
    assert _resolve_build_version(resume, "t") == (2, "resume")

    # a ready-but-not-current version outranks resume: publish it, don't re-embed
    finalized = _FakeClient(ledger=[(1, "activated", True), (2, "ready", False)])
    assert _resolve_build_version(finalized, "t") == (2, "finalized")

    activated = _FakeClient(ledger=[(1, "activated", True)])
    assert _resolve_build_version(activated, "t") == (2, "fresh")

    empty = _FakeClient(ledger=[])
    assert _resolve_build_version(empty, "t") == (1, "fresh")


def test_resume_creates_missing_table() -> None:
    """A building row whose table was never created (crash between allocate and
    create_table) gets its table (re)created before the resume inserts."""
    client = _FakeClient(ledger=[(1, "building", False)])
    assert client.table_created is False
    chunks = [_chunk(i) for i in range(4)]
    emb = _CountingEmbed()
    version, present = _build_and_activate(client, "t", chunks, emb, batch_size=2)
    assert version == 1
    assert client.created_table_calls == 1  # table (re)created on resume
    assert present == 4 and client.activated and client.ready
    assert len(emb.seen) == 4  # a fresh table => everything embedded


def test_finalized_version_activates_without_reembed() -> None:
    """A ready-but-unpublished version is activated directly — never re-embedded."""
    chunks = [_chunk(i) for i in range(4)]
    client = _FakeClient(present={c.hash for c in chunks}, ledger=[(1, "ready", False)])
    emb = _CountingEmbed()
    version, present = _build_and_activate(client, "t", chunks, emb, batch_size=2)
    assert version == 1 and present == 4
    assert client.activated
    assert emb.seen == []  # nothing re-embedded
    assert client.created_table_calls == 0  # existing table reused


def test_finalized_version_refuses_partial() -> None:
    """A ready version short of the expected rows is not published."""
    chunks = [_chunk(i) for i in range(4)]
    client = _FakeClient(present={chunks[0].hash}, ledger=[(1, "ready", False)])
    with pytest.raises(NeonIngestError, match="ready version"):
        _build_and_activate(client, "t", chunks, _CountingEmbed(), batch_size=2)
    assert not client.activated
