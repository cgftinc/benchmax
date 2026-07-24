"""Slice 2: fake-backed behavioral tests for the Neon client SQL surface.

These are the fully-seeded behavior tests deferred from Slice A (which shipped
only strict xfail stubs). A faked psycopg connection records the composable SQL
the client emits and returns canned rows, so we assert the *shapes*, *ordering*,
and *failure/race paths* of the real statements without a live database:

- the B14 build lifecycle order (extensions CASCADE -> register vector -> table ->
  populate -> ANN/BM25 indexes after load -> VACUUM in autocommit -> mark ready);
- the ANN opclass/operator match and the BM25 ``to_bm25query`` + ``ASC`` polarity,
  read through the owner-rights VIEW with a schema-qualified index regclass;
- ``%s`` / named-placeholder binding everywhere + a hard reject of raw ``str`` (B4);
- the bounded scale-to-zero reconnect for single statements AND transactions (B16);
- activation/rollback one-row transition guards (abort before publishing the view);
- the version-ledger prune seam + the advisory-lock allocation/prune race
  invariant (frozen in Slice A, implemented + tested here).

Rendering: ``Composable.as_string(None)`` materializes a statement to its final
SQL text so substring assertions are meaningful (``str()`` would show the repr).
"""

from __future__ import annotations

import re

import psycopg
import pytest
from psycopg import sql

from castform.rag.corpus.neon import client as client_mod
from castform.rag.corpus.neon.client import (
    InDoubtTransactionError,
    MissingExtensionError,
    NeonClient,
    VersionStateError,
)
from castform.rag.corpus.neon.schema import (
    READ_COLUMNS,
    VIEW_COLUMNS,
    NeonTableSpec,
    NeonVersionRecord,
    ReadGrantSpec,
    RetentionPolicy,
    index_names,
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
    or ``RETURNING`` write can return seeded rows; everything else returns empty.
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


class _PruneConn(_FakeConn):
    """Ledger-aware fake: models the guarded retire ``RETURNING`` semantics.

    The retire UPDATE returns the version row ONLY for a row that is authoritatively
    non-current and in an eligible state — exactly the DB guard the client relies on
    to decide whether to drop. ``raced_current`` marks versions whose retire returns
    nothing (simulating a row that raced into ``is_current`` after planning), so the
    client must NOT drop them.
    """

    def __init__(
        self,
        ledger: list[tuple[int, str, bool]],
        raced_current: set[int] | None = None,
    ) -> None:
        super().__init__()
        self._ledger_rows = list(ledger)
        self._state = {v: (st, cur) for v, st, cur in ledger}
        self._raced = raced_current or set()

    def execute(self, query: sql.Composable, params: dict | None = None) -> _FakeCursor:
        text = render(query)
        self.executed.append(text)
        self.exec_autocommit.append(self.autocommit)
        if "SELECT version, state, is_current" in text:
            desc = [("version",), ("state",), ("is_current",)]
            return _FakeCursor(list(self._ledger_rows), desc)
        if "SET state = 'retired'" in text:
            version = params["version"]  # type: ignore[index]
            st, cur = self._state.get(version, (None, None))
            eligible = st in ("activated", "ready", "retired")
            if version in self._raced or cur or not eligible:
                return _FakeCursor([], None)  # guard blocks: no row -> no drop
            return _FakeCursor([(version,)], [("version",)])
        return _FakeCursor([], None)


def _rendered(conn: _FakeConn) -> list[str]:
    """Executed entries as flat strings (EXECUTEMANY tuples flattened to their SQL)."""
    return [e[1] if isinstance(e, tuple) else e for e in conn.executed]


def _first_index(items: list[str], needle: str) -> int:
    for i, item in enumerate(items):
        if needle in item:
            return i
    raise AssertionError(f"{needle!r} not found in {items}")


# --- B4: composable-only execute seam -----------------------------------------


def test_execute_binds_params_and_rejects_raw_str() -> None:
    conn = _FakeConn(responses={"count": ([(7,)], [("count",)])})
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    rows = c.execute(c.count_sql("mycorpus"), {"unused": 1})
    assert rows == [(7,)]
    assert conn.committed == 1  # single-statement reads commit + release snapshot
    with pytest.raises(TypeError):
        c.execute("SELECT 1")  # a raw string bypasses the composable guarantee (B4)
    with pytest.raises(TypeError):
        c.execute_in_transaction(["SELECT 1"])  # type: ignore[list-item]


def test_advisory_lock_stmt_is_shared_keyed_form() -> None:
    """Allocation, activation, rollback, and prune contend on ONE lock key form."""
    c = NeonClient(lambda: "dsn")
    assert (
        render(c._advisory_lock_stmt())
        == "SELECT pg_advisory_xact_lock(hashtext(%(logical)s))"
    )


# --- B14: build lifecycle order -----------------------------------------------


def test_build_version_lifecycle_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Extensions are provisioned out-of-band; the build only checks presence.
    conn = _FakeConn(
        responses={
            "FROM pg_extension": (
                [("lakebase_vector",), ("lakebase_text",)],
                [("extname",)],
            )
        }
    )
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    # Isolate pgvector registration into an ordered marker (no real adapters).
    monkeypatch.setattr(
        c, "_register_vector", lambda cn: cn.executed.append("REGISTER_VECTOR")
    )

    c.build_version(SPEC, rows=[("id0", "text", {}, [0.1], "a.md", 0)])

    order = _rendered(conn)
    ext_check = _first_index(order, "FROM pg_extension")
    reg = _first_index(order, "REGISTER_VECTOR")
    tbl = _first_index(order, 'CREATE TABLE "')  # the physical table, not the ledger
    load = _first_index(order, 'INSERT INTO "mycorpus__v2"')  # populate (executemany)
    ann = _first_index(order, "lakebase_ann")
    bm25 = _first_index(order, "lakebase_bm25")
    vac = _first_index(order, "VACUUM")
    ready = _first_index(order, "state = 'ready'")

    # The presence check precedes pgvector registration; no CREATE EXTENSION issued.
    assert ext_check < reg
    assert not any("CREATE EXTENSION" in s for s in order)
    # check extensions -> register vector -> table -> load -> ann/bm25 -> vacuum -> ready
    assert reg < tbl < load < ann
    assert load < bm25
    assert max(ann, bm25) < vac < ready
    # allocation (insert 'building' ledger row) happens before the table is created.
    assert _first_index(order, "INSERT INTO neon_corpus_versions") < tbl


def test_require_extensions_checks_presence_not_install() -> None:
    conn = _FakeConn(
        responses={
            "FROM pg_extension": (
                [("lakebase_vector",), ("lakebase_text",)],
                [("extname",)],
            )
        }
    )
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    c.require_extensions()  # both present => no raise
    stmts = _rendered(conn)
    assert any("SELECT extname FROM pg_extension" in s for s in stmts)
    assert not any("CREATE EXTENSION" in s for s in stmts)  # writer never installs


def test_require_extensions_raises_on_unprovisioned_db() -> None:
    conn = _FakeConn(  # only one of the two required extensions present
        responses={"FROM pg_extension": ([("lakebase_vector",)], [("extname",)])}
    )
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(MissingExtensionError, match="lakebase_text"):
        c.require_extensions()


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
    # embedding is NOT NULL: a null embedding is silently unsearchable by the ANN.
    assert "embedding vector(3072) NOT NULL" in text
    # regconfig is a bound literal cast, never interpolated (B4).
    assert "'pg_catalog.english'::regconfig" in text
    assert "content_tsv tsvector GENERATED ALWAYS AS" in text


def test_insert_uses_placeholders_only() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.insert_rows_sql(SPEC))
    assert text.count("%s") == 6  # one per physical column, content_tsv excluded
    assert "content_tsv" not in text
    assert '"mycorpus__v2"' in text


# --- B14/B1: ANN opclass + operator match, read through the view --------------


def test_ann_index_opclass() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.create_ann_index_sql(SPEC))
    assert "USING lakebase_ann" in text
    assert "embedding vector_cosine_ops" in text
    assert index_names("mycorpus", 2)["ann"] in text.replace('"', "")


def test_vector_candidate_operator_and_view() -> None:
    c = NeonClient(lambda: "dsn")
    query, params = c.vector_candidates_sql(SPEC)
    text = render(query)
    assert params == {}  # vector path carries no client-side bind params
    # cosine opclass binds <=>, NOT <-> (which is L2 and would skip the ANN index);
    # the param is cast to ::vector (a bound list binds as float8[] otherwise).
    assert "embedding <=> %(vector)s::vector" in text
    assert "<->" not in text
    # RO reads the stable owner-rights view, never the physical version table.
    assert 'FROM "mycorpus"' in text
    assert "mycorpus__v2" not in text
    assert "AS native_score" in text
    assert "ORDER BY embedding <=> %(vector)s::vector ASC" in text
    assert text.rstrip().endswith("LIMIT %(top_k)s")


# --- B13: BM25 index storage params + to_bm25query polarity + qualification ----


def test_read_projection_omits_heavy_columns_but_view_keeps_them() -> None:
    """B4: candidate/scan SELECTs omit embedding + content_tsv from the OUTPUT, while
    the reader view still exposes them so the ANN/BM25 expressions can reference them."""
    c = NeonClient(lambda: "dsn")
    for query in (
        c.vector_candidates_sql(SPEC)[0],
        c.scan_page_sql("mycorpus", after=False),
        c.top_level_sql("mycorpus"),
        c.sample_sql("mycorpus"),
    ):
        select_list = render(query).split(" FROM ")[0]
        assert '"embedding"' not in select_list  # heavy vector never projected
        assert '"content_tsv"' not in select_list
    # the vector score still references the embedding column (available via the view).
    assert "embedding <=> %(vector)s::vector" in render(c.vector_candidates_sql(SPEC)[0])
    # the reader view still projects the FULL column set (incl. embedding/content_tsv).
    view_ddl = render(c._view_ddl("mycorpus", 2))
    assert all(f'"{col}"' in view_ddl for col in VIEW_COLUMNS)


def test_bm25_index_storage_params_not_gucs() -> None:
    c = NeonClient(lambda: "dsn")
    text = render(c.create_bm25_index_sql(SPEC))
    assert "USING lakebase_bm25" in text
    assert "content_tsv tsvector_bm25_ops" in text
    # k1/b are index storage params (WITH), literal — never GUCs or placeholders.
    assert "WITH (k1 = 1.2, b = 0.75)" in text
    assert "SET " not in text
    assert "%s" not in text


def test_bm25_candidate_asc_polarity_and_schema_qualified_regclass() -> None:
    c = NeonClient(lambda: "dsn")
    query, params = c.bm25_candidates_sql(SPEC, schema="corpora")
    text = render(query)
    # scored column is the <@> LEFT operand; query text is a bound param through
    # to_tsvector with the same baked config; index is the SECOND arg (regclass).
    assert "content_tsv <@> to_bm25query(" in text
    assert "to_tsvector('pg_catalog.english'::regconfig, %(text)s)" in text
    # schema + index are BOUND params fed through quote_ident (not inlined
    # literals), so the regclass resolves under any RO search_path, injection-safe.
    assert "quote_ident(%(bm25_schema)s)" in text
    assert "quote_ident(%(bm25_index)s)" in text
    assert "::regclass)" in text
    assert "'corpora'" not in text  # schema is a param, never inlined
    assert params == {"bm25_schema": "corpora", "bm25_index": "mycorpus__v2_bm25"}
    assert 'FROM "mycorpus"' in text  # the view, not the physical table
    # negative-score polarity => best candidates come first under ASC.
    assert "ORDER BY content_tsv <@> to_bm25query(" in text
    assert text.rstrip().endswith("ASC LIMIT %(top_k)s")


def test_candidate_filter_is_spliced_not_built() -> None:
    c = NeonClient(lambda: "dsn")
    where = sql.SQL("metadata @> {}").format(sql.Placeholder("f0"))
    query, _ = c.vector_candidates_sql(SPEC, where=where)
    assert "WHERE metadata @> %(f0)s" in render(query)


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
    counters: dict = {"connects": 0, "dsn": 0, "last_kwargs": {}}

    def fake_connect(dsn: str, **kwargs: object) -> _FakeConn:
        counters["connects"] += 1
        counters["last_kwargs"] = kwargs
        return queue.pop(0)

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    return counters


def test_connect_disables_server_side_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """_connect must pass prepare_threshold=None (B13: no cached plan across a bm25
    index swap, since the query binds the index ::regclass OID)."""
    live = _FakeConn(responses={"count": ([(1,)], [("count",)])})
    counters = _patch_connect(monkeypatch, [live])
    c = NeonClient(lambda: "dsn")
    c._conn = None
    c.execute(c.count_sql("mycorpus"))
    assert "prepare_threshold" in counters["last_kwargs"]
    assert counters["last_kwargs"]["prepare_threshold"] is None


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


def test_transaction_reconnects_on_dead_first_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transaction (here activation) whose first statement finds the cached
    socket dead reconnects and retries the whole txn once (B16)."""
    live = _FakeConn(responses={"SET state = 'activated'": ([(2,)], [("version",)])})
    _patch_connect(monkeypatch, [live])

    class _DiesFirst(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            self.broken = True
            raise psycopg.OperationalError("dead on first use")

    c = NeonClient(lambda: "dsn")
    c._conn = _DiesFirst()
    c.activate(SPEC, GRANT)  # reconnects to the live conn and publishes
    assert live.committed == 1


class _CommitDies(_FakeConn):
    """A connection whose statements succeed but whose commit() dies (lost ack)."""

    def commit(self) -> None:
        self.broken = True
        raise psycopg.OperationalError("commit ack lost")


def test_txn_commit_failure_is_in_doubt_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead conn AT the commit boundary is in-doubt: raise, never retry the work
    (retrying could double-apply the possibly-committed transaction)."""
    resp = {"SET state = 'activated'": ([(2,)], [("version",)])}
    live = _FakeConn(responses=resp)
    counters = _patch_connect(monkeypatch, [live])
    conn = _CommitDies(responses=resp)
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(InDoubtTransactionError):
        c.activate(SPEC, GRANT)
    assert counters["connects"] == 0  # never reconnected / re-ran the work
    assert live.committed == 0  # the retry target was never touched


def test_execute_commit_failure_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-statement commit-boundary failure is re-raised, never retried."""
    live = _FakeConn(responses={"count": ([(1,)], [("count",)])})
    counters = _patch_connect(monkeypatch, [live])
    conn = _CommitDies(responses={"count": ([(5,)], [("count",)])})
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(psycopg.OperationalError):
        c.execute(c.count_sql("mycorpus"))
    assert counters["connects"] == 0  # commit-stage failure is not retried
    assert live.executed == []


# --- B5: activation one-row guard ---------------------------------------------


def test_activate_publishes_with_ready_guard_and_returning() -> None:
    conn = _FakeConn(responses={"SET state = 'activated'": ([(2,)], [("version",)])})
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    c.activate(SPEC, GRANT)
    order = _rendered(conn)
    assert "pg_advisory_xact_lock" in order[0]
    clear = _first_index(order, "is_current = false")
    trans = _first_index(order, "SET state = 'activated'")
    view = _first_index(order, "CREATE OR REPLACE VIEW")
    assert clear < trans < view  # clear prior current BEFORE the guarded transition
    assert "AND state = 'ready'" in order[trans]  # only a ready row may publish
    assert "RETURNING version" in order[trans]
    assert "security_invoker = false" in order[view]  # owner-rights view
    assert any("GRANT USAGE ON SCHEMA" in s for s in order)
    assert any("GRANT SELECT ON" in s for s in order)
    assert conn.committed == 1


def test_activate_aborts_before_view_when_not_ready() -> None:
    conn = _FakeConn()  # transition RETURNING yields no row -> version is not ready
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(VersionStateError):
        c.activate(SPEC, GRANT)
    order = _rendered(conn)
    assert not any("CREATE OR REPLACE VIEW" in s for s in order)  # never published
    assert conn.rolled_back == 1
    assert conn.committed == 0


def test_activate_rejects_mismatched_grant_view() -> None:
    c = NeonClient(lambda: "dsn")
    c._conn = _FakeConn()
    bad = ReadGrantSpec(schema="corpora", view="not_the_view", ro_role="ro")
    with pytest.raises(ValueError):
        c.activate(SPEC, bad)


def test_activation_rolls_back_atomically() -> None:
    """Uses the REAL activation flow: transition succeeds, view DDL fails, rollback."""

    class _FailView(_FakeConn):
        def execute(self, query, params=None):  # type: ignore[override]
            cur = super().execute(query, params)
            if "CREATE OR REPLACE VIEW" in render(query):
                raise RuntimeError("view swap failed")
            return cur

    conn = _FailView(responses={"SET state = 'activated'": ([(2,)], [("version",)])})
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(RuntimeError):
        c.activate(SPEC, GRANT)
    assert conn.rolled_back == 1
    assert conn.committed == 0


# --- B5: rollback one-row guard -----------------------------------------------


def test_rollback_validates_target_first_then_repoints() -> None:
    conn = _FakeConn(
        responses={
            "FOR UPDATE": ([(1,)], [("version",)]),
            "SET is_current = true": ([(1,)], [("version",)]),
        }
    )
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    c.rollback("mycorpus", 1)
    order = _rendered(conn)
    assert "pg_advisory_xact_lock" in order[0]
    validate = _first_index(order, "FOR UPDATE")
    clear = _first_index(order, "is_current = false")
    setcur = _first_index(order, "SET is_current = true")
    view = _first_index(order, "CREATE OR REPLACE VIEW")
    # validate+lock the activated target BEFORE clearing the current pointer.
    assert validate < clear < setcur < view
    assert "state = 'activated'" in order[validate]
    assert (
        "state = 'activated'" in order[setcur] and "RETURNING version" in order[setcur]
    )
    assert '"mycorpus__v1"' in order[view]
    assert conn.committed == 1


def test_rollback_aborts_on_missing_or_nonactivated_target() -> None:
    conn = _FakeConn()  # FOR UPDATE returns nothing -> target not activated/missing
    c = NeonClient(lambda: "dsn")
    c._conn = conn
    with pytest.raises(VersionStateError):
        c.rollback("mycorpus", 9)
    order = _rendered(conn)
    # the current pointer and the view are untouched (aborted before clearing).
    assert not any("is_current = false" in s for s in order)
    assert not any("CREATE OR REPLACE VIEW" in s for s in order)
    assert conn.rolled_back == 1
    assert conn.committed == 0


# --- B5: ledger prune seam + concurrent allocation/prune race safety ----------


def _records(*rows: tuple[int, str, bool]) -> list[NeonVersionRecord]:
    return [
        NeonVersionRecord(logical_name="mycorpus", version=v, state=st, is_current=cur)
        for v, st, cur in rows
    ]


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


def test_prune_locks_before_read_and_drops_only_returned() -> None:
    ledger = [
        (1, "activated", True),  # raced-in current: oldest, but now live
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", False),
        (5, "activated", False),
    ]
    conn = _PruneConn(ledger)
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    c._conn = conn
    dropped = c.prune("mycorpus")

    order = _rendered(conn)
    lock_i = _first_index(order, "pg_advisory_xact_lock")
    read_i = _first_index(order, "SELECT version, state, is_current")
    assert lock_i < read_i  # lock acquired BEFORE the ledger snapshot is read

    # v5,v4 kept (retention) + current-protected v1; v3,v2 retired and dropped.
    assert set(dropped) == {2, 3}
    assert not any('DROP TABLE IF EXISTS "mycorpus__v1"' in s for s in order)
    assert conn.committed == 1


def test_prune_drop_gated_on_authoritative_retire_returning() -> None:
    """A version that raced into current after planning returns no retire row, so
    its table is NOT dropped even though plan_prune selected it."""
    ledger = [
        (1, "activated", True),
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", False),
        (5, "activated", False),
    ]
    conn = _PruneConn(ledger, raced_current={3})  # v3 retire returns nothing
    c = NeonClient(
        lambda: "dsn", retention=RetentionPolicy(keep_activated=2, keep_ready=1)
    )
    c._conn = conn
    dropped = c.prune("mycorpus")
    assert dropped == [2]  # only the authoritatively-non-current row is dropped
    order = _rendered(conn)
    assert any('DROP TABLE IF EXISTS "mycorpus__v2"' in s for s in order)
    assert not any('DROP TABLE IF EXISTS "mycorpus__v3"' in s for s in order)


def test_prune_no_op_when_nothing_prunable() -> None:
    conn = _PruneConn([(1, "activated", True), (2, "activated", False)])
    c = NeonClient(lambda: "dsn")  # default retention keep_activated=2
    c._conn = conn
    assert c.prune("mycorpus") == []
    assert not any("DROP TABLE" in s for s in _rendered(conn))
    assert conn.committed == 1


def test_prune_rolls_back_on_error() -> None:
    class _FailDrop(_PruneConn):
        def execute(self, query, params=None):  # type: ignore[override]
            cur = super().execute(query, params)
            if "DROP TABLE" in render(query):
                raise RuntimeError("drop failed")
            return cur

    ledger = [
        (1, "activated", False),
        (2, "activated", False),
        (3, "activated", False),
        (4, "activated", True),
    ]
    conn = _FailDrop(ledger)
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
    id_i = READ_COLUMNS.index("id")
    file_i = READ_COLUMNS.index("source_file")
    index_i = READ_COLUMNS.index("chunk_index")

    def _row(file: str, idx: int, cid: str) -> tuple:
        row = ["x"] * len(READ_COLUMNS)
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
