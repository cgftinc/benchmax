"""Slice 2: fake-backed behavioral tests for the Neon client SQL surface.

These are the fully-seeded behavior tests deferred from Slice A (which shipped
only strict xfail stubs). A faked psycopg connection records the composable SQL
the client emits and returns canned rows, so we assert the *shapes* and *ordering*
of the real statements without a live database:

- the B14 build lifecycle order (extensions CASCADE -> register vector -> table ->
  populate -> ANN/BM25 indexes after load -> VACUUM in autocommit -> mark ready);
- the ANN opclass/operator match and the BM25 ``to_bm25query`` + ``ASC`` polarity;
- ``%s`` / named-placeholder binding everywhere (no interpolated values, B4);
- the bounded scale-to-zero reconnect (B16);
- the version-ledger prune seam + the concurrent allocation/prune advisory-lock
  race invariant (frozen in Slice A, implemented + tested here).

Rendering: ``Composable.as_string(None)`` materializes a statement to its final
SQL text so substring assertions are meaningful (``str()`` would show the repr).
"""

from __future__ import annotations

import re

import psycopg
import pytest
from psycopg import sql

from castform.rag.corpus.neon import client as client_mod
from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.schema import (
    NeonTableSpec,
    NeonVersionRecord,
    ReadGrantSpec,
    RetentionPolicy,
    VIEW_COLUMNS,
    index_names,
    physical_table_name,
)

SPEC = NeonTableSpec(logical_name="mycorpus", version=2)
GRANT = ReadGrantSpec(schema="corpora", view="mycorpus", ro_role="ro")


def render(composable: sql.Composable) -> str:
    """Render a composable to final SQL text (no live connection needed)."""
    return composable.as_string(None)


# --- fakes --------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list | None = None, description: object = None) -> None:
        self._rows = rows or []
        self.description = description

    def fetchall(self) -> list:
        return self._rows

    def executemany(self, query: sql.Composable, seq: list) -> None:
        self.conn.executed.append(("EXECUTEMANY", render(query), list(seq)))

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeConn:
    """Records executed SQL (rendered) + autocommit-at-execute; returns canned rows.

    ``responses`` maps a rendered-SQL substring to ``(rows, description)`` so a read
    can return seeded rows; everything else returns an empty result set.
    """

    def __init__(self, responses: dict[str, tuple[list, object]] | None = None) -> None:
        self.executed: list = []
        self.exec_autocommit: list[bool] = []
        self.autocommit = False
        self.committed = 0
        self.rolled_back = 0
        self.closed = 0
        self.broken = False
        self._responses = responses or {}

    def execute(self, query: sql.Composable, params: dict | None = None) -> _FakeCursor:
        text = render(query)
        self.executed.append(text)
        self.exec_autocommit.append(self.autocommit)
        for substr, (rows, desc) in self._responses.items():
            if substr in text:
                return _FakeCursor(rows, desc)
        return _FakeCursor([], None)

    def cursor(self) -> _FakeCursor:
        cur = _FakeCursor()
        cur.conn = self  # type: ignore[attr-defined]
        return cur

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


def _rendered(conn: _FakeConn) -> list[str]:
    """Executed entries as flat strings (EXECUTEMANY tuples flattened to their SQL)."""
    return [e[1] if isinstance(e, tuple) else e for e in conn.executed]


def _first_index(items: list[str], needle: str) -> int:
    for i, item in enumerate(items):
        if needle in item:
            return i
    raise AssertionError(f"{needle!r} not found in {items}")


# --- B4: composable-only execute seam -----------------------------------------


def test_execute_rejects_nothing_but_binds_params_dict() -> None:
    conn = _FakeConn(responses={"count": ([(7,)], [("count",)])})
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    rows = c.execute(c.count_sql("mycorpus"))
    assert rows == [(7,)]
    assert conn.committed == 1  # single-statement reads commit + release snapshot


# --- B14: build lifecycle order -----------------------------------------------


def test_build_version_lifecycle_order(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn()
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    # Isolate pgvector registration into an ordered marker (no real adapters).
    monkeypatch.setattr(
        c, "_register_vector", lambda cn: cn.executed.append("REGISTER_VECTOR")
    )

    c.build_version(SPEC, rows=[("id0", "text", {}, [0.1], "a.md", 0)])

    order = _rendered(conn)
    ext = _first_index(order, "CREATE EXTENSION")
    reg = _first_index(order, "REGISTER_VECTOR")
    tbl = _first_index(order, 'CREATE TABLE "')  # the physical table, not the ledger
    load = _first_index(order, 'INSERT INTO "mycorpus__v2"')  # populate (executemany)
    ann = _first_index(order, "lakebase_ann")
    bm25 = _first_index(order, "lakebase_bm25")
    vac = _first_index(order, "VACUUM")
    ready = _first_index(order, "state = 'ready'")

    # extensions -> register vector -> table -> load -> ann/bm25 -> vacuum -> ready
    assert ext < reg < tbl < load < ann
    assert load < bm25
    assert max(ann, bm25) < vac < ready
    # allocation (insert 'building' ledger row) happens before the table is created.
    assert _first_index(order, "'mycorpus', 2, 'building'") < tbl


def test_create_extensions_cascade() -> None:
    c = NeonClient(lambda: "dsn")
    stmts = [render(s) for s in c.create_extensions_sql()]
    assert any("lakebase_vector" in s and "CASCADE" in s for s in stmts)
    assert any("lakebase_text" in s and "CASCADE" in s for s in stmts)
    assert all("IF NOT EXISTS" in s for s in stmts)


def test_register_vector_only_flagged_after_call() -> None:
    calls: list[object] = []
    c = NeonClient(lambda: "dsn")
    c._register_vector = lambda cn: calls.append(cn)  # type: ignore[method-assign]
    conn = _FakeConn()
    c._conn = conn
    assert c._vector_registered is False
    c.register_vector_types()
    assert c._vector_registered is True
    assert calls == [conn]


# --- B14: managed table + regconfig safety ------------------------------------


def test_create_table_shape_and_bound_regconfig() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.create_table_sql(SPEC))
    assert '"mycorpus__v2"' in text
    assert "id text PRIMARY KEY" in text
    assert "metadata jsonb NOT NULL DEFAULT '{}'::jsonb" in text
    assert "embedding vector(3072)" in text
    # regconfig is a bound literal cast, never interpolated (B4).
    assert "'pg_catalog.english'::regconfig" in text
    assert "content_tsv tsvector GENERATED ALWAYS AS" in text


def test_insert_uses_placeholders_only() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.insert_rows_sql(SPEC))
    assert text.count("%s") == 6  # one per physical column, content_tsv excluded
    assert "content_tsv" not in text
    assert '"mycorpus__v2"' in text


# --- B14/B1: ANN opclass + operator match -------------------------------------


def test_ann_index_opclass() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.create_ann_index_sql(SPEC))
    assert "USING lakebase_ann" in text
    assert "embedding vector_cosine_ops" in text
    assert index_names("mycorpus", 2)["ann"] in text.replace('"', "")


def test_vector_candidate_operator_matches_cosine_opclass() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.vector_candidates_sql(SPEC))
    # cosine opclass binds <=>, NOT <-> (which is L2 and would skip the ANN index).
    assert "embedding <=> %(vector)s" in text
    assert "<->" not in text
    assert "AS native_score" in text
    assert text.rstrip().endswith("LIMIT %(top_k)s")
    assert "ORDER BY embedding <=> %(vector)s ASC" in text


# --- B13: BM25 index storage params + to_bm25query polarity -------------------


def test_bm25_index_storage_params_not_gucs() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.create_bm25_index_sql(SPEC))
    assert "USING lakebase_bm25" in text
    assert "content_tsv tsvector_bm25_ops" in text
    # k1/b are index storage params (WITH), literal — never GUCs or placeholders.
    assert "WITH (k1 = 1.2, b = 0.75)" in text
    assert "SET " not in text
    assert "%s" not in text


def test_bm25_candidate_to_bm25query_asc_polarity() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.bm25_candidates_sql(SPEC))
    # scored column is the <@> LEFT operand; query text is a bound param through
    # to_tsvector with the same baked config; index is the SECOND arg (regclass).
    assert "content_tsv <@> to_bm25query(" in text
    assert "to_tsvector('pg_catalog.english'::regconfig, %(text)s)" in text
    assert "::regclass)" in text
    # negative-score polarity => best candidates come first under ASC.
    assert "ORDER BY content_tsv <@> to_bm25query(" in text
    assert text.rstrip().endswith("ASC LIMIT %(top_k)s")


def test_hybrid_returns_two_independent_candidate_queries() -> None:
    c = NeonClient(lambda: "dsn")
    vec, bm25 = c.hybrid_candidates_sql(SPEC)
    assert "<=> %(vector)s" in render(vec)
    assert "to_bm25query(" in render(bm25)


def test_candidate_filter_is_spliced_not_built() -> None:
    c = NeonClient(lambda: "dsn")
    where = sql.SQL("metadata @> {}").format(sql.Placeholder("f0"))
    text = render(c.vector_candidates_sql(SPEC, where=where))
    assert "WHERE metadata @> %(f0)s" in text


# --- B14: VACUUM in autocommit, outside any transaction -----------------------


def test_vacuum_runs_in_autocommit_and_restores() -> None:
    conn = _FakeConn()
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    c.vacuum(SPEC)
    vac_i = _first_index(_rendered(conn), "VACUUM ANALYZE")
    assert conn.exec_autocommit[vac_i] is True  # autocommit ON while VACUUM runs
    assert conn.autocommit is False  # restored afterwards
    assert '"mycorpus__v2"' in _rendered(conn)[vac_i]


# --- B16: bounded scale-to-zero reconnect -------------------------------------


def _patch_connect(monkeypatch: pytest.MonkeyPatch, queue: list[_FakeConn]) -> dict:
    counters = {"connects": 0, "dsn": 0}

    def fake_connect(dsn: str) -> _FakeConn:
        counters["connects"] += 1
        return queue.pop(0)

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    return counters


def test_reconnect_when_cached_conn_already_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _FakeConn(responses={"count": ([(3,)], [("count",)])})
    counters = _patch_connect(monkeypatch, [live])
    dead = _FakeConn()
    dead.broken = True  # autosuspend killed the cached socket

    def dsn() -> str:
        counters["dsn"] += 1
        return "dsn"

    c = NeonClient(dsn)
    c._conn = dead
    rows = c.execute(c.count_sql("mycorpus"))
    assert rows == [(3,)]
    assert counters["connects"] == 1  # reconnected proactively
    assert counters["dsn"] == 1  # DSN re-resolved on reconnect
    assert dead.executed == []  # never touched the dead handle


def test_reconnect_once_on_operational_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _FakeConn(responses={"count": ([(9,)], [("count",)])})
    _patch_connect(monkeypatch, [live])

    class _DiesOnUse(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            self.broken = True
            raise psycopg.OperationalError("server closed the connection")

    c = NeonClient(lambda: "dsn")
    c._conn = _DiesOnUse()
    assert c.execute(c.count_sql("mycorpus")) == [(9,)]


def test_live_conn_error_is_not_retried() -> None:
    class _LockError(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            self.attempts = getattr(self, "attempts", 0) + 1
            raise psycopg.OperationalError("lock not available")  # broken stays False

    conn = _LockError()
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(psycopg.OperationalError):
        c.execute(c.count_sql("mycorpus"))
    assert conn.attempts == 1  # a live-conn error is a real failure, not retried


def test_reconnect_is_bounded_second_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AlwaysDead(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            self.broken = True
            raise psycopg.OperationalError("dead")

    second = _AlwaysDead()
    counters = _patch_connect(monkeypatch, [second])
    c = NeonClient(lambda: "dsn")
    c._conn = _AlwaysDead()
    with pytest.raises(psycopg.OperationalError):
        c.execute(c.count_sql("mycorpus"))
    assert counters["connects"] == 1  # exactly one bounded reconnect, no storm


def test_vector_types_reregistered_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _FakeConn()
    _patch_connect(monkeypatch, [fresh])
    registered: list[object] = []
    c = NeonClient(lambda: "dsn")
    c._register_vector = lambda cn: registered.append(cn)  # type: ignore[method-assign]
    c._vector_registered = True  # a search connection had registered before
    got = c._connect()
    assert got is fresh
    assert registered == [fresh]  # adapters re-applied on the fresh socket


# --- B5: activation / rollback under the advisory lock ------------------------


def test_activation_statement_shape() -> None:
    c = NeonClient(lambda: "dsn")
    stmts = [render(s) for s in c.activate_version_sql(SPEC, GRANT)]
    assert "pg_advisory_xact_lock" in stmts[0]
    assert any("is_current = false" in s for s in stmts)  # clear prior current
    assert any("state = 'activated', is_current = true" in s for s in stmts)
    assert any(
        "CREATE OR REPLACE VIEW" in s and "security_invoker = false" in s for s in stmts
    )
    assert any("GRANT USAGE ON SCHEMA" in s for s in stmts)
    assert any("GRANT SELECT ON" in s for s in stmts)


def test_rollback_repoints_current_guarded_by_activated() -> None:
    c = NeonClient(lambda: "dsn")
    stmts = [render(s) for s in c.rollback_version_sql("mycorpus", 1)]
    assert "pg_advisory_xact_lock" in stmts[0]
    # rollback flips is_current, only onto an already-activated version.
    assert any("is_current = true" in s and "state = 'activated'" in s for s in stmts)
    assert any("CREATE OR REPLACE VIEW" in s and '"mycorpus__v1"' in s for s in stmts)


def test_activation_rolls_back_atomically() -> None:
    """The frozen atomicity contract still holds through the Slice 2 executor."""

    class _FailSecond(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            super().execute(query, params)
            if len(self.executed) == 2:
                raise RuntimeError("view swap failed")
            return _FakeCursor()

    conn = _FailSecond()
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(RuntimeError):
        c.execute_in_transaction(
            [sql.SQL("SELECT 1"), sql.SQL("SELECT 2"), sql.SQL("SELECT 3")]
        )
    assert conn.rolled_back == 1
    assert conn.committed == 0


# --- B5: ledger prune seam + concurrent allocation/prune race safety ----------


def _records(*rows: tuple[int, str, bool]) -> list[NeonVersionRecord]:
    return [
        NeonVersionRecord(logical_name="mycorpus", version=v, state=st, is_current=cur)
        for v, st, cur in rows
    ]


def test_advisory_lock_key_identical_across_operations() -> None:
    """Allocation, activation, rollback, and prune must contend on ONE lock key."""
    c = NeonClient(lambda: "dsn")
    keys = {
        render(c.allocate_version_sql("mycorpus", 3)[0]),
        render(c.activate_version_sql(SPEC, GRANT)[0]),
        render(c.rollback_version_sql("mycorpus", 1)[0]),
        render(
            c.prune_versions_sql(
                "mycorpus", _records((1, "activated", False), (2, "retired", False))
            )[0]
        ),
    }
    assert len(keys) == 1
    assert "pg_advisory_xact_lock(hashtext('mycorpus'))" in keys.pop()


def test_plan_prune_never_touches_current_or_building() -> None:
    c = NeonClient(lambda: "dsn")
    records = _records(
        (1, "activated", False),
        (2, "activated", False),
        (3, "building", False),  # in-flight build: must be untouched
        (4, "activated", True),  # live version: must be untouched
    )
    retire, drop = c.plan_prune(records)
    assert 3 not in retire and 3 not in drop  # concurrent build protected
    assert 4 not in retire and 4 not in drop  # current protected


def test_plan_prune_respects_retention_minimums() -> None:
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    records = _records(
        (1, "ready", False),
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", False),
        (5, "activated", True),
    )
    retire, _ = c.plan_prune(records)
    # keep current(5) + 2 most-recent activated(4,3); retire the oldest activated(2).
    assert retire == [2]
    assert 1 not in retire  # the single ready version is retained


def test_prune_versions_sql_is_lock_first_and_guards_retire() -> None:
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    records = _records(
        (1, "activated", False),
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", True),
    )
    stmts = [render(s) for s in c.prune_versions_sql("mycorpus", records)]
    assert "pg_advisory_xact_lock" in stmts[0]  # serialize before any drop
    assert any("DROP TABLE IF EXISTS" in s and '"mycorpus__v1"' in s for s in stmts)
    # every retire is guarded so a raced-in current can never be retired.
    for s in stmts:
        if "SET state = 'retired'" in s:
            assert "is_current = false" in s


def test_prune_versions_sql_empty_when_nothing_prunable() -> None:
    c = NeonClient(lambda: "dsn")
    records = _records((1, "activated", True), (2, "activated", False))
    assert c.prune_versions_sql("mycorpus", records) == []


def test_live_prune_locks_before_reading_ledger_and_excludes_raced_current() -> None:
    """The race invariant: acquire the lock, THEN read the ledger under it.

    The under-lock snapshot reflects an activation that raced ahead — the version
    that just became ``is_current`` (here v1, the oldest, which a stale pre-lock
    planner might have targeted) is excluded, so its physical table is never
    dropped. Concurrent allocation contends on the same lock, so it fully
    serializes with the prune.
    """
    ledger_rows = [
        (1, "activated", True),  # raced-in current: oldest, but now live
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", False),
        (5, "activated", False),
    ]
    conn = _FakeConn(
        responses={
            "SELECT version, state, is_current": (
                ledger_rows,
                [("version",), ("state",), ("is_current",)],
            )
        }
    )
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    c._conn = conn

    dropped = c.prune(conn_logical := "mycorpus")

    order = _rendered(conn)
    lock_i = _first_index(order, "pg_advisory_xact_lock")
    read_i = _first_index(order, "SELECT version, state, is_current")
    assert lock_i < read_i  # lock acquired BEFORE the ledger snapshot is read

    # v5,v4 kept (retention) + current-protected v1; v3,v2 retired.
    assert set(dropped) == {2, 3}
    v1_table = physical_table_name(conn_logical, 1)
    assert not any(f'DROP TABLE IF EXISTS "{v1_table}"' in s for s in order)
    assert conn.committed == 1


def test_live_prune_rolls_back_on_error() -> None:
    class _FailOnDrop(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            cur = super().execute(query, params)
            if "DROP TABLE" in render(query):
                raise RuntimeError("drop failed")
            return cur

    conn = _FailOnDrop(
        responses={
            "SELECT version, state, is_current": (
                [
                    (1, "activated", False),  # retirable -> triggers a DROP
                    (2, "activated", False),
                    (3, "activated", False),
                    (4, "activated", True),
                ],
                [("version",), ("state",), ("is_current",)],
            )
        }
    )
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    c._conn = conn
    with pytest.raises(RuntimeError):
        c.prune("mycorpus")
    assert conn.rolled_back == 1
    assert conn.committed == 0


# --- B9: deterministic scan ---------------------------------------------------


def test_scan_page_sql_first_page_has_no_cursor() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.scan_page_sql("mycorpus", after=False))
    assert "WHERE" not in text
    assert "ORDER BY source_file, chunk_index, id" in text
    assert text.rstrip().endswith("LIMIT %(batch_size)s")


def test_scan_page_sql_keyset_cursor_is_row_tuple() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.scan_page_sql("mycorpus", after=True))
    assert (
        "(source_file, chunk_index, id) > "
        "(%(after_file)s, %(after_index)s, %(after_id)s)" in text
    )


def test_scan_chunks_is_deterministic_and_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    id_i = VIEW_COLUMNS.index("id")
    file_i = VIEW_COLUMNS.index("source_file")
    index_i = VIEW_COLUMNS.index("chunk_index")

    def _row(file: str, idx: int, cid: str) -> tuple:
        row = ["x"] * len(VIEW_COLUMNS)
        row[id_i], row[file_i], row[index_i] = cid, file, idx
        return tuple(row)

    rows = sorted(
        [_row("b.md", 2, "h3"), _row("a.md", 10, "h2"), _row("a.md", 2, "h1")],
        key=lambda r: (r[file_i], r[index_i], r[id_i]),
    )

    def fake_execute(query: sql.Composable, params: dict) -> list:
        limit = params["batch_size"]
        if "after_file" in params:
            after = (params["after_file"], params["after_index"], params["after_id"])
            start = next(
                (
                    i
                    for i, r in enumerate(rows)
                    if (r[file_i], r[index_i], r[id_i]) > after
                ),
                len(rows),
            )
        else:
            start = 0
        return rows[start : start + limit]

    c = NeonClient(lambda: "dsn")
    monkeypatch.setattr(c, "execute", fake_execute)
    first = [r[id_i] for r in c.scan_chunks("mycorpus", batch_size=2)]
    second = [r[id_i] for r in c.scan_chunks("mycorpus", batch_size=2)]
    assert first == second == ["h1", "h2", "h3"]  # (a.md,2)<(a.md,10)<(b.md,2)


def test_scan_chunks_rejects_bad_batch_size() -> None:
    c = NeonClient(lambda: "dsn")
    with pytest.raises(ValueError):
        list(c.scan_chunks("mycorpus", batch_size=0))


# --- reads --------------------------------------------------------------------


def test_read_query_shapes_use_view_and_placeholders() -> None:
    c = NeonClient(lambda: "dsn")
    assert 'FROM "mycorpus"' in render(c.count_sql("mycorpus"))
    sample = render(c.sample_sql("mycorpus"))
    assert "length(content) >= %(min_chars)s" in sample
    assert "ORDER BY random() LIMIT %(n)s" in sample
    neigh = render(c.neighbors_sql("mycorpus"))
    assert "chunk_index IN (%(prev_index)s, %(next_index)s)" in neigh
    assert "DISTINCT ON (source_file)" in render(c.top_level_sql("mycorpus"))


# --- contract: the client stays Chunk-free ------------------------------------


def test_client_module_is_chunk_free() -> None:
    import inspect

    source = inspect.getsource(client_mod)
    assert "chunkers" not in source  # no chunk-model import
    assert re.search(r"^\s*(from|import).*\bChunk\b", source, re.MULTILINE) is None
