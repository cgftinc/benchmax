"""Neon connection seam + versioned-replace lifecycle executor (Slice 2).

The ``NeonClient`` owns two responsibilities:

- a psycopg3 connection with a **bounded reconnect** for Neon/Lakebase
  scale-to-zero (a cached connection is *dead*, not merely slow, after an
  autosuspend, B16), and a composable-only execute seam (B4); and
- the **versioned-replace lifecycle** — build a fresh per-version physical
  table + indexes in the frozen order (B14), atomically activate/roll back the
  reader view under a per-logical advisory lock (B5), and prune retired versions
  race-safely against concurrent allocation.

Everything psycopg / pgvector is imported lazily *inside* methods (never at
module load) so this module — and the whole ``neon`` package — imports without
the ``neon`` extra installed, matching the pickle-safe, lazy-import discipline of
``turbopuffer/search.py``. The identifier helpers, allowlists, dataclasses, and
DDL skeleton constants are reused from :mod:`castform.rag.corpus.neon.schema`
(the frozen safety contract); the client only *composes and executes*.

The client is deliberately ``Chunk``-free: reads and scans return raw row tuples,
and :class:`NeonChunkSource` (Slice 1) maps them to ``Chunk`` objects. That keeps
the client cheap to pickle and decoupled from the chunk model (per the contract).

Lifecycle order (B14)
---------------------
**verify the required extensions are installed** (they are provisioned
out-of-band by ``provision.py`` under admin rights — the writer cannot ``CREATE
EXTENSION``; the build only checks presence) -> **register pgvector types on the
connection, only after the extension exists** -> ledger + current-pointer index
-> allocate a version and
build its physical table -> populate -> **build the ANN + BM25 indexes after the
load** (an empty-then-filled index is wasteful, and BM25 corpus statistics need
the rows present) -> ``VACUUM ANALYZE`` in **autocommit, outside any
transaction** (VACUUM cannot run in a transaction block, and it primes the BM25
term statistics) -> mark the version ``ready``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from castform.platform.credentials import TokenProvider
from castform.rag.corpus.neon.schema import (
    CREATE_CURRENT_POINTER_INDEX,
    CREATE_LEDGER_SKELETON,
    DEFAULT_RETENTION,
    READ_COLUMNS,
    VIEW_COLUMNS,
    NeonTableSpec,
    NeonVersionRecord,
    ReadGrantSpec,
    RetentionPolicy,
    index_names,
    physical_table_name,
    validate_logical_name,
    validate_text_search_config,
    validate_version,
    view_name,
)

if TYPE_CHECKING:
    from psycopg import sql


# --- frozen SQL surface constants (B13/B14) ------------------------------------

# B14: extensions pulled in before anything else. ``lakebase_vector`` supplies
# the ANN access method + vector type + cosine opclass (and CASCADE-installs its
# pgvector dependency); ``lakebase_text`` supplies the ``lakebase_bm25`` index
# method plus ``to_bm25query``/``<@>``. CASCADE pulls transitive dependencies.
# Ordering + CASCADE are frozen (B14); the exact names are slice-3-verified.
REQUIRED_EXTENSIONS: tuple[str, ...] = ("lakebase_vector", "lakebase_text")

# PROVEN ANN choice (Slice 3, see schema.PROVEN_ANN_DDL). Live-verified on Neon
# Lakebase (PG 18.4): native ``lakebase_ann`` indexes a full-precision
# ``vector(3072)`` column with ``vector_cosine_ops`` and the planner uses it
# (EXPLAIN: ``Index Scan using ..._ann``). lakebase_ann has no 2000-dim cap, so the
# ``halfvec`` workaround is not needed (kept documented in schema as the
# storage-saving alternative).
ANN_ACCESS_METHOD = "lakebase_ann"
ANN_VECTOR_TYPE = "vector(3072)"
ANN_OPCLASS = "vector_cosine_ops"
# Ordering operator matching the cosine opclass (B14): ``<=>`` is cosine distance;
# ``<->`` (L2) would not match ``vector_cosine_ops`` and the planner would skip the
# ANN index. Live-confirmed against ``lakebase_ann``.
ANN_DISTANCE_OPERATOR = "<=>"
# Query-param cast, frozen WITH the type/opclass/operator as one unit (Slice 3). A
# bound param (Python list/tuple) binds as ``float8[]`` and ``embedding <=> $1``
# then errors "operator does not exist: vector <=> double precision[]"; casting the
# param to ``vector`` resolves the operator and the planner still uses the
# ``lakebase_ann`` index (verified for both list- and text-form params).
ANN_QUERY_PARAM_CAST = "vector"

# B13: BM25 relevance via the ``<@>`` distance operator against ``to_bm25query``.
# The score is NEGATIVE (more-relevant is more-negative), so candidate ordering is
# ``ASC`` (best-first). The index is the ``lakebase_bm25`` access method over the
# generated ``content_tsv`` column with the ``tsvector_bm25_ops`` opclass. ``k1``/
# ``b`` are BM25 tuning knobs baked as INDEX STORAGE PARAMETERS (``WITH (...)``) —
# not GUCs/``SET`` — so they travel with the index.
BM25_ACCESS_METHOD = "lakebase_bm25"
BM25_OPCLASS = "tsvector_bm25_ops"
BM25_K1 = 1.2
BM25_B = 0.75

# B16 hardening: fail-fast socket options so a connection killed by Neon
# scale-to-zero *raises* instead of hanging on a half-open socket. During a long
# embedding-only phase the DB idles; without TCP keepalive the kernel never
# learns the peer is gone and the next read() blocks forever in poll(), so the
# bounded reconnect never fires. keepalives make the kernel probe the idle socket
# and surface a dead peer as an OperationalError (caught by _is_dead ->
# reconnect); connect_timeout bounds the reconnect's own connect; tcp_user_timeout
# (ms) bounds an in-flight send on a half-dead socket. Applied to every connect.
CONNECT_PARAMS: dict[str, int] = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    "tcp_user_timeout": 60000,
}

# Columns physically written at ingest — ``content_tsv`` is a generated column and
# is never inserted; ``embedding`` is adapted by the registered pgvector types.
INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "content",
    "metadata",
    "embedding",
    "source_file",
    "chunk_index",
)


class VersionStateError(RuntimeError):
    """A ledger state transition did not affect exactly one row (B5).

    Raised when activation/rollback cannot find a single eligible row to
    transition (wrong state, missing target, or a would-be multi-row update), so
    the caller aborts the transaction BEFORE publishing the reader view rather
    than committing an inconsistent ledger/view pair.
    """


class InDoubtTransactionError(RuntimeError):
    """A transaction's connection died AT the commit boundary (B16).

    The ``COMMIT`` was sent but its acknowledgement was lost, so the server may or
    may not have committed. Unlike a pre-commit failure, this is NOT retried —
    replaying non-idempotent work could double-apply it. The caller must reconcile
    against the ledger (re-read state) before deciding whether to re-run.

    Re-run safety after reconcile is per operation: activate (``state='ready'``
    guard), rollback (idempotent re-point), and prune (re-retire + ``DROP TABLE IF
    EXISTS``) all tolerate a blind re-run; ``allocate_version`` does NOT — its
    unguarded ``INSERT ... 'building'`` would raise a duplicate-key on a re-run if
    the in-doubt commit landed, so reconciling allocation means checking whether the
    ``building`` row already exists before retrying.
    """


class MissingExtensionError(RuntimeError):
    """A required Lakebase extension is not installed on the target database.

    Extension INSTALLATION is an admin/superuser step owned exclusively by
    ``provision.py``; the writer role that runs the build cannot ``CREATE
    EXTENSION``. The build verifies presence and raises this rather than running
    against an unprovisioned database (where a missing extension would otherwise
    surface later as an opaque missing-type / missing-access-method error).
    """


class NeonClient:
    """Thin connection + SQL-execution wrapper over a resolved Neon DSN.

    A single client instance carries one lazily-opened psycopg connection. The
    read path is handed an RO provider (SELECT only), the ingest path an RW
    provider (DDL + DML); the two never share a role (see ``credentials.py``).

    Args:
        dsn_provider: Callable resolving a DSN string per connection. Invoked on
            first connect and again on every bounded reconnect, so a rotated or
            short-lived DSN is always re-resolved.
        retention: Version-retention policy the pruner enforces. Defaults to the
            frozen ``DEFAULT_RETENTION`` (>= 2 activated, >= 1 ready).
    """

    def __init__(
        self,
        dsn_provider: TokenProvider,
        *,
        retention: RetentionPolicy = DEFAULT_RETENTION,
    ) -> None:
        self._dsn_provider = dsn_provider
        self._retention = retention
        self._conn: Any = None
        self._vector_registered = False

    # --- connection + bounded reconnect (B16) --------------------------------

    def _connect(self) -> Any:
        """Open a psycopg connection from the freshly-resolved DSN and cache it.

        Imports ``psycopg`` lazily so the module stays importable without the
        ``neon`` extra. Does NOT register pgvector types here — on a fresh build
        the ``vector`` extension may not exist yet, so registration is a separate
        step run only after ``CREATE EXTENSION`` (B14). Read connections register
        via :meth:`register_vector_types` once the corpus already exists.

        ``prepare_threshold=None`` disables psycopg's server-side auto-PREPARE
        (B13): the BM25 query binds an index ``::regclass`` OID, so a cached plan
        would reference a stale/dropped index after a versioned index swap.
        """
        import psycopg

        dsn = self._dsn_provider()
        self._conn = psycopg.connect(dsn, prepare_threshold=None, **CONNECT_PARAMS)
        # pgvector adapters are per-connection; re-register after a reconnect so a
        # search that binds a vector param still works post-autosuspend (B16). On
        # the very first build connect the extension doesn't exist yet, so the flag
        # is still False and registration is deferred to register_vector_types.
        if self._vector_registered:
            self._register_vector(self._conn)
        return self._conn

    def register_vector_types(self, conn: Any | None = None) -> None:
        """Register pgvector's psycopg adapters on *conn* (B14 ordering).

        Must run only after the ``lakebase_vector`` extension exists, otherwise the
        type OIDs it looks up are absent. Split out from :meth:`_connect` so the
        build path can order it strictly after ``CREATE EXTENSION``. Sets a flag so
        every subsequent (re)connect re-registers automatically.
        """
        self._register_vector(conn if conn is not None else self._live_conn())
        self._vector_registered = True

    def _register_vector(self, conn: Any) -> None:
        """Import + apply pgvector's adapters (isolated seam for test injection)."""
        from pgvector.psycopg import register_vector

        register_vector(conn)

    @staticmethod
    def _is_dead(conn: Any) -> bool:
        """Return True if a cached connection is closed/broken (autosuspend, B16).

        After a Neon scale-to-zero the cached socket is gone; psycopg surfaces
        that as ``closed`` (nonzero) or ``broken`` (True). Either means the next
        statement would raise, so we must reconnect rather than reuse it.
        """
        return bool(getattr(conn, "closed", 0)) or bool(getattr(conn, "broken", False))

    def _live_conn(self) -> Any:
        """Return a live connection, reconnecting if the cached one is dead (B16)."""
        conn = self._conn
        if conn is None or self._is_dead(conn):
            return self._connect()
        return conn

    @staticmethod
    def _require_composable(query: object) -> None:
        """Reject a raw ``str`` at the execute seam — composables only (B4).

        Every statement must reach the driver as ``psycopg.sql.Composable`` so
        identifiers are ``Identifier`` and values are bound params, never string
        interpolation. A raw ``str`` would bypass that guarantee, so it is a
        ``TypeError`` here, not something the driver silently accepts.
        """
        from psycopg import sql

        if not isinstance(query, sql.Composable):
            raise TypeError(
                f"query must be a psycopg sql.Composable, got {type(query).__name__}"
            )

    @staticmethod
    def _safe_rollback(conn: Any) -> None:
        """Roll back, swallowing errors (a dead connection cannot roll back)."""
        try:
            conn.rollback()
        except Exception:
            pass

    def execute(
        self, query: sql.Composable, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute one composable statement with bound params and return rows.

        Single-statement autocommit-equivalent: each successful statement is
        committed so a read releases its snapshot and a DDL persists, leaving no
        open transaction to block the ``VACUUM`` autocommit toggle. Grouped atomic
        work goes through :meth:`execute_in_transaction` (or the lifecycle mutators),
        not here. Rejects a raw ``str`` (B4).

        Bounded reconnect (B16): a cached connection killed by autosuspend raises
        an ``OperationalError``/``InterfaceError`` on first use; if the handle is
        genuinely dead we drop it, reconnect **once**, and retry. A live-connection
        error (bad SQL, a constraint or lock failure) is re-raised immediately and
        never retried, and a second dead-connection failure propagates too — the
        retry is bounded, never a reconnect storm.

        Commit-boundary failures are NOT retried (data integrity): if the statement
        may already have committed but the ack was lost, replaying it could
        double-apply a write, so the error is surfaced rather than retried (a read
        caller simply re-issues; a write caller must reconcile).
        """
        import psycopg

        self._require_composable(query)
        for attempt in range(2):  # original attempt + one bounded reconnect
            conn = self._live_conn()
            commit_attempted = False
            try:
                cur = conn.execute(query, params or {})
                rows = cur.fetchall() if cur.description is not None else []
                commit_attempted = True
                conn.commit()
                return rows
            except (psycopg.OperationalError, psycopg.InterfaceError):
                dead = self._is_dead(conn)
                self._conn = None  # drop the handle; next _live_conn reconnects
                if not commit_attempted and attempt == 0 and dead:
                    continue  # pre-commit dead conn: reconnect and retry once
                raise  # commit-stage (in-doubt), live-conn, or 2nd dead-conn failure
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _in_bounded_txn(self, work: Any) -> Any:
        """Run ``work(conn)`` as one transaction with a bounded dead-conn retry (B16).

        ``work`` receives a live connection, performs the transaction body, and its
        return value is passed through after commit.

        Retry policy hinges on WHERE the failure happens (data integrity):

        - **Pre-commit** dead connection (any ``work`` statement — including the
          first, when an apparently-live cached socket turns out dead) — the
          server-side transaction is already rolled back, nothing is durable, so we
          drop the handle and retry the WHOLE transaction ONCE on a freshly-resolved,
          pgvector-re-registered connection. ``commit_attempted`` is reset each
          iteration so the retry starts clean.
        - **Commit-boundary** failure — the ``COMMIT`` was sent but its ack was lost,
          so the server MAY have committed. Retrying could double-apply non-idempotent
          work, so we raise :class:`InDoubtTransactionError` (never retry) for the
          caller to reconcile against the ledger. No rollback is attempted (a
          committed txn cannot be undone; a dead handle cannot send one anyway).
        - A live-connection error, an application error (e.g.
          :class:`VersionStateError`), or a second pre-commit dead-conn failure
          propagates as-is.
        """
        import psycopg

        for attempt in range(2):  # original attempt + one bounded reconnect
            conn = self._live_conn()
            commit_attempted = False
            try:
                result = work(conn)
                commit_attempted = True
                conn.commit()
                return result
            except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                dead = self._is_dead(conn)
                self._conn = None  # drop the handle; next _live_conn reconnects
                if commit_attempted:
                    raise InDoubtTransactionError(
                        "connection failed at commit; the transaction may or may "
                        "not have committed — reconcile against the ledger before "
                        "retrying"
                    ) from exc
                self._safe_rollback(conn)
                if attempt == 0 and dead:
                    continue  # pre-commit dead conn: retry the whole txn once
                raise  # pre-commit live-conn error, or a second dead-conn failure
            except Exception:
                self._safe_rollback(conn)
                raise  # application error: abort, no retry
        raise AssertionError("unreachable")  # the loop always returns or raises

    def execute_in_transaction(self, statements: list[sql.Composable]) -> None:
        """Run *statements* as one all-or-nothing transaction (B5).

        Rejects raw strings (B4) and runs under :meth:`_in_bounded_txn`, so an
        autosuspend-killed cached connection discovered dead on the first statement
        reconnects and the whole transaction retries once. On any statement failure
        the transaction is rolled back and the error re-raised, leaving no partial
        state.
        """
        for statement in statements:
            self._require_composable(statement)

        def work(conn: Any) -> None:
            for statement in statements:
                conn.execute(statement)

        self._in_bounded_txn(work)

    def _advisory_lock_shared_stmt(self) -> sql.Composed:
        """Return the SHARED (reader) per-logical advisory-lock acquisition (B5).

        The reader counterpart to :meth:`_advisory_lock_stmt`: keyed identically
        (``hashtext(%(logical)s)``) so a read contends on the same key as the
        writers. Multiple readers share the lock (concurrent reads proceed), while
        activation/rollback/prune take the EXCLUSIVE lock — so a version swap waits
        for in-flight reads and blocks new reads only for the duration of its
        transaction, never mid-read. Transaction-scoped: released at commit/rollback.
        """
        from psycopg import sql

        return sql.SQL("SELECT pg_advisory_xact_lock_shared(hashtext(%(logical)s))")

    def read_in_snapshot(
        self,
        logical_name: str,
        work: Any,
        *,
        session_setup: list[sql.Composable] | None = None,
    ) -> Any:
        """Run ``work(conn)`` as ONE read transaction under the shared lock (B5/B16).

        A single query must resolve the current version AND execute every leg
        against THAT version consistently: a concurrent activation between the
        version lookup and a leg (or between the two hybrid legs) could otherwise
        pair an old BM25 index regclass with the new view, or fuse candidates from
        two versions. So the whole read runs in one transaction holding the shared
        per-logical advisory lock, which blocks a concurrent exclusive activation
        from swapping mid-read.

        ``work`` receives the live connection and issues the ledger read + leg
        SELECTs on it (never :meth:`execute`, which would commit each statement and
        drop the snapshot). Optional ``session_setup`` (e.g.
        ``SET LOCAL lakebase_bm25.prefilter = on`` for a filtered lexical/hybrid
        query, F7) runs first; ``SET LOCAL`` is transaction-scoped and auto-resets
        at commit/rollback, so it never leaks to a reused connection. Rejects raw
        strings (B4) and rides :meth:`_in_bounded_txn`: an autosuspend-killed
        connection reconnects and the whole read — lock, setup, and ``work`` —
        retries once on the fresh connection (idempotent for a read).
        """
        validate_logical_name(logical_name)
        setup = session_setup or []
        for statement in setup:
            self._require_composable(statement)
        params = {"logical": logical_name}

        def _work(conn: Any) -> Any:
            conn.execute(self._advisory_lock_shared_stmt(), params)
            for statement in setup:
                conn.execute(statement)
            return work(conn)

        return self._in_bounded_txn(_work)

    def vacuum(self, spec: NeonTableSpec) -> None:
        """``VACUUM ANALYZE`` a version's table in autocommit, outside any txn (B14).

        VACUUM cannot run inside a transaction block, so the connection is flipped
        to autocommit for the call and restored afterwards. Priming the table also
        computes the BM25 corpus statistics the lexical scorer relies on.
        """
        from psycopg import sql

        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        stmt = sql.SQL("VACUUM ANALYZE {}").format(table)
        conn = self._live_conn()
        previous = conn.autocommit
        conn.autocommit = True
        try:
            conn.execute(stmt)
        finally:
            conn.autocommit = previous

    # --- DDL assembly: extensions, ledger, table, indexes (B14) --------------

    def require_extensions(self) -> None:
        """Fail fast unless every required Lakebase extension is already installed.

        Extension INSTALLATION is an admin/superuser step owned exclusively by
        ``provision.py`` (see REQUIRED_EXTENSIONS); the writer role that runs the
        build cannot ``CREATE EXTENSION``. The build therefore only checks presence
        against ``pg_extension`` and raises :class:`MissingExtensionError` on an
        unprovisioned database, rather than issuing a privileged (and duplicative)
        ``CREATE EXTENSION`` through the writer connection.
        """
        from psycopg import sql

        present = {
            row[0]
            for row in self.execute(
                sql.SQL(
                    "SELECT extname FROM pg_extension WHERE extname = ANY(%(names)s)"
                ),
                {"names": list(REQUIRED_EXTENSIONS)},
            )
        }
        missing = [ext for ext in REQUIRED_EXTENSIONS if ext not in present]
        if missing:
            raise MissingExtensionError(
                f"required neon extensions not installed: {missing}; run "
                "python -m castform.rag.corpus.neon.provision first"
            )

    def create_ledger_sql(self) -> list[sql.Composed]:
        """Return the per-version ledger table + current-pointer index DDL (B5).

        Idempotent (``IF NOT EXISTS``): the ledger is shared across every logical
        corpus in the database. The partial unique index enforces the at-most-one
        ``is_current`` row per logical name.
        """
        from psycopg import sql

        return [sql.SQL(CREATE_LEDGER_SKELETON), sql.SQL(CREATE_CURRENT_POINTER_INDEX)]

    def create_table_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the ``CREATE TABLE`` composable for one physical corpus version.

        Identifiers are ``sql.Identifier`` and the ``regconfig`` is a validated,
        bound ``sql.Literal`` (never interpolated, B4). The embedding column is the
        slice-3-proven ``vector(3072)`` and ``NOT NULL`` — a null embedding is
        silently unsearchable by the ANN index, so ingest must always supply one.
        """
        from psycopg import sql

        validate_version(spec.version)
        tsconfig = validate_text_search_config(spec.text_search_config)
        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        return sql.SQL(
            "CREATE TABLE {table} ("
            "id text PRIMARY KEY, "
            "content text NOT NULL, "
            "metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb, "
            "embedding {vtype} NOT NULL, "
            "source_file text NOT NULL, "
            "chunk_index integer NOT NULL, "
            "content_tsv tsvector GENERATED ALWAYS AS "
            "(to_tsvector({tsconfig}::regconfig, content)) STORED"
            ")"
        ).format(
            table=table,
            vtype=sql.SQL(ANN_VECTOR_TYPE),
            tsconfig=sql.Literal(tsconfig),
        )

    def insert_rows_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the parametrized bulk ``INSERT`` for one version (executemany).

        Every value is a ``%s`` placeholder (B4); ``content_tsv`` is generated and
        never written. Callers pass one param sequence per row.
        """
        from psycopg import sql

        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        cols = sql.SQL(", ").join(sql.Identifier(c) for c in INSERT_COLUMNS)
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in INSERT_COLUMNS)
        return sql.SQL("INSERT INTO {table} ({cols}) VALUES ({vals})").format(
            table=table, cols=cols, vals=placeholders
        )

    def create_ann_index_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the ANN vector index DDL, opclass matching the query operator (B14).

        Built AFTER the load. The opclass (``vector_cosine_ops``) must match the
        ``ORDER BY embedding <=> ...`` operator used by the vector candidate query.
        """
        from psycopg import sql

        names = index_names(spec.logical_name, spec.version)
        return sql.SQL(
            "CREATE INDEX {index} ON {table} USING {method} (embedding {opclass})"
        ).format(
            index=sql.Identifier(names["ann"]),
            table=sql.Identifier(physical_table_name(spec.logical_name, spec.version)),
            method=sql.SQL(ANN_ACCESS_METHOD),
            opclass=sql.SQL(ANN_OPCLASS),
        )

    def create_bm25_index_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the BM25 index DDL with ``k1``/``b`` as index storage params (B13).

        Built AFTER the load so the corpus statistics are computed over the real
        rows. ``k1``/``b`` are bound as ``WITH`` storage parameters (frozen index
        state), never as session GUCs.
        """
        from psycopg import sql

        names = index_names(spec.logical_name, spec.version)
        return sql.SQL(
            "CREATE INDEX {index} ON {table} USING {method} "
            "(content_tsv {opclass}) WITH (k1 = {k1}, b = {b})"
        ).format(
            index=sql.Identifier(names["bm25"]),
            table=sql.Identifier(physical_table_name(spec.logical_name, spec.version)),
            method=sql.SQL(BM25_ACCESS_METHOD),
            opclass=sql.SQL(BM25_OPCLASS),
            k1=sql.Literal(BM25_K1),
            b=sql.Literal(BM25_B),
        )

    def create_aux_indexes_sql(self, spec: NeonTableSpec) -> list[sql.Composed]:
        """Return the meta-GIN, scan-btree, and tsvector-GIN index DDL (B3/B6).

        ``meta_gin`` (``jsonb_path_ops``) serves ``@>`` containment filters;
        ``scan`` backs the deterministic ``(source_file, chunk_index, id)`` order;
        ``tsv_gin`` is the native FTS fallback.
        """
        from psycopg import sql

        names = index_names(spec.logical_name, spec.version)
        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        return [
            sql.SQL(
                "CREATE INDEX {index} ON {table} USING gin (metadata jsonb_path_ops)"
            ).format(index=sql.Identifier(names["meta_gin"]), table=table),
            sql.SQL(
                "CREATE INDEX {index} ON {table} (source_file, chunk_index, id)"
            ).format(index=sql.Identifier(names["scan"]), table=table),
            sql.SQL("CREATE INDEX {index} ON {table} USING gin (content_tsv)").format(
                index=sql.Identifier(names["tsv_gin"]), table=table
            ),
        ]

    # --- ledger state machine + advisory lock (B5) ---------------------------

    def _advisory_lock_stmt(self) -> sql.Composed:
        """Return the per-logical ``pg_advisory_xact_lock`` acquisition (B5).

        Every allocation, activation, rollback, and prune acquires the SAME
        transaction-scoped lock keyed on the logical name (passed as the bound
        ``%(logical)s`` param), so concurrent build-vs-prune (and two concurrent
        builds) serialize and the lock releases automatically at commit/rollback.
        ``hashtext`` maps the name to the lock's bigint key.
        """
        from psycopg import sql

        return sql.SQL("SELECT pg_advisory_xact_lock(hashtext(%(logical)s))")

    def _view_ddl(self, logical_name: str, version: int) -> sql.Composed:
        """Return the owner-rights ``CREATE OR REPLACE VIEW`` for a version (B4/B5)."""
        from psycopg import sql

        return sql.SQL(
            "CREATE OR REPLACE VIEW {view} WITH (security_invoker = false) AS "
            "SELECT {columns} FROM {table}"
        ).format(
            view=sql.Identifier(view_name(logical_name)),
            columns=sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS),
            table=sql.Identifier(physical_table_name(logical_name, version)),
        )

    def allocate_version(self, spec: NeonTableSpec) -> None:
        """Reserve ``spec.version`` as ``building`` under the advisory lock (B5).

        Acquires the advisory lock FIRST, then inserts the ``building`` ledger row —
        holding the lock across the insert is what makes concurrent allocation
        race-safe: two builders cannot both reserve the same version. Bounded-retry
        wrapped (B16). Caller-supplied values are bound params (B4).
        """
        from psycopg import sql

        validate_logical_name(spec.logical_name)
        validate_version(spec.version)
        params = {"logical": spec.logical_name, "version": spec.version}

        def work(conn: Any) -> None:
            conn.execute(self._advisory_lock_stmt(), params)
            conn.execute(
                sql.SQL(
                    "INSERT INTO neon_corpus_versions (logical_name, version, state) "
                    "VALUES (%(logical)s, %(version)s, 'building')"
                ),
                params,
            )

        self._in_bounded_txn(work)

    def mark_ready_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the ledger update transitioning ``building`` -> ``ready`` (B5).

        Guarded by ``state = 'building'`` so the frozen transition is enforced in
        SQL alongside the DB CHECK domain; caller values are bound ``%s`` params.
        Run with params ``{"logical", "version"}``.
        """
        from psycopg import sql

        return sql.SQL(
            "UPDATE neon_corpus_versions SET state = 'ready', ready_at = now() "
            "WHERE logical_name = %(logical)s AND version = %(version)s "
            "AND state = 'building'"
        )

    def activate(self, spec: NeonTableSpec, grant: ReadGrantSpec) -> None:
        """Publish ``spec.version`` atomically, or abort before publishing (B5).

        One transaction under the advisory lock: acquire the lock; clear the prior
        ``is_current`` row; transition THIS version ``ready`` -> ``activated`` +
        ``is_current`` **guarded by ``state = 'ready'`` and RETURNING**. If that
        does not affect exactly one row (the version is not ``ready`` — e.g. still
        ``building``, already ``retired``, or missing) the transaction is aborted
        with :class:`VersionStateError` BEFORE the view is published, so a bad state
        can never publish or rewrite ``activated_at``. Only on a confirmed single
        transition do we ``CREATE OR REPLACE VIEW`` (owner-rights) and issue the RO
        grants, all committing together. Clearing the old current before setting the
        new one keeps the partial unique index satisfied at every statement
        boundary. Bounded-retry wrapped (B16).

        In-doubt-commit note: if the connection dies AFTER ``COMMIT`` is sent but
        before its ack, the wrapper raises :class:`InDoubtTransactionError` (it does
        NOT retry), since the activation may in fact have committed. Callers treat
        that as "re-check the ledger", not "the version is unpublished". A
        genuine bad-state rejection is a distinct :class:`VersionStateError`.

        ``grant.view`` must equal :func:`view_name` of the logical name, or the RO
        grant lands on a different identifier than the published view and search
        fails with a permission error — checked here.
        """
        from psycopg import sql

        validate_logical_name(spec.logical_name)
        validate_version(spec.version)
        expected_view = view_name(spec.logical_name)
        if grant.view != expected_view:
            raise ValueError(
                f"grant.view {grant.view!r} must equal view_name("
                f"{spec.logical_name!r}) = {expected_view!r}"
            )
        params = {"logical": spec.logical_name, "version": spec.version}

        def work(conn: Any) -> None:
            conn.execute(self._advisory_lock_stmt(), params)
            conn.execute(
                sql.SQL(
                    "UPDATE neon_corpus_versions SET is_current = false "
                    "WHERE logical_name = %(logical)s AND is_current"
                ),
                params,
            )
            transitioned = conn.execute(
                sql.SQL(
                    "UPDATE neon_corpus_versions "
                    "SET state = 'activated', is_current = true, activated_at = now() "
                    "WHERE logical_name = %(logical)s AND version = %(version)s "
                    "AND state = 'ready' RETURNING version"
                ),
                params,
            ).fetchall()
            if len(transitioned) != 1:
                raise VersionStateError(
                    f"cannot activate {spec.logical_name} v{spec.version}: expected "
                    f"exactly one 'ready' row to transition, got {len(transitioned)}"
                )
            conn.execute(self._view_ddl(spec.logical_name, spec.version))
            for statement in self.read_grant_sql(grant):
                conn.execute(statement)

        self._in_bounded_txn(work)

    def rollback(self, logical_name: str, target_version: int) -> None:
        """Re-point ``is_current`` to a prior ``activated`` version, or abort (B5).

        Non-destructive and O(1): prior physical tables are retained, so rollback
        only re-points the ledger + view under the advisory lock. The target is
        **validated and row-locked FIRST** (``SELECT ... WHERE state = 'activated'
        FOR UPDATE``); unless exactly one activated target exists the transaction
        aborts with :class:`VersionStateError` BEFORE the current pointer or view is
        touched — so a missing/non-activated target can never leave the view pointed
        at a table with no current ledger row. Then clear the prior ``is_current``,
        set the target ``is_current`` (RETURNING-guarded to exactly one row), and
        re-point the view. ``state`` is unchanged (rollback flips ``is_current``,
        not ``state``). Bounded-retry wrapped (B16).
        """
        from psycopg import sql

        validate_logical_name(logical_name)
        validate_version(target_version)
        params = {"logical": logical_name, "version": target_version}

        def work(conn: Any) -> None:
            conn.execute(self._advisory_lock_stmt(), params)
            target = conn.execute(
                sql.SQL(
                    "SELECT version FROM neon_corpus_versions "
                    "WHERE logical_name = %(logical)s AND version = %(version)s "
                    "AND state = 'activated' FOR UPDATE"
                ),
                params,
            ).fetchall()
            if len(target) != 1:
                raise VersionStateError(
                    f"cannot roll back {logical_name} to v{target_version}: target "
                    f"is not an activated version (found {len(target)})"
                )
            conn.execute(
                sql.SQL(
                    "UPDATE neon_corpus_versions SET is_current = false "
                    "WHERE logical_name = %(logical)s AND is_current"
                ),
                params,
            )
            pointed = conn.execute(
                sql.SQL(
                    "UPDATE neon_corpus_versions SET is_current = true "
                    "WHERE logical_name = %(logical)s AND version = %(version)s "
                    "AND state = 'activated' RETURNING version"
                ),
                params,
            ).fetchall()
            if len(pointed) != 1:
                raise VersionStateError(
                    f"rollback of {logical_name} to v{target_version} updated "
                    f"{len(pointed)} pointer rows, expected 1"
                )
            conn.execute(self._view_ddl(logical_name, target_version))

        self._in_bounded_txn(work)

    def read_grant_sql(self, grant: ReadGrantSpec) -> list[sql.Composed]:
        """Return the ``GRANT USAGE``/``GRANT SELECT`` statements for the RO role (B5).

        The RO role gets schema ``USAGE`` + ``SELECT`` on the stable owner-rights
        view only — never on the physical version tables. Issued as part of
        activation so a first-create view is immediately readable.
        """
        from psycopg import sql

        schema = sql.Identifier(grant.schema)
        view = sql.Identifier(grant.view)
        role = sql.Identifier(grant.ro_role)
        return [
            sql.SQL("GRANT USAGE ON SCHEMA {schema} TO {role}").format(
                schema=schema, role=role
            ),
            sql.SQL("GRANT SELECT ON {view} TO {role}").format(view=view, role=role),
        ]

    # --- pruning seam + concurrent-race safety (B5, deferred from Slice A) ----

    def plan_prune(
        self, records: list[NeonVersionRecord]
    ) -> tuple[list[int], list[int]]:
        """Decide which versions to retire, honoring retention (pure, race-agnostic).

        Returns ``(retire_versions, drop_table_versions)`` — identical lists here
        (a retired version's physical table is dropped), split so a caller could
        keep tombstones. The policy, newest-first:

        - never touch ``is_current`` (the live view target) or ``building`` rows
          (a concurrent build is in flight — dropping its table would corrupt it);
        - keep the ``keep_activated`` most-recent ``activated`` versions (so
          rollback always has a target) and the ``keep_ready`` most-recent
          ``ready`` versions;
        - everything older is retired and its physical table dropped.

        This is the *decision* only; :meth:`prune` executes it under the advisory
        lock against a ledger snapshot re-read UNDER that lock, and drops a table
        only after the guarded retire UPDATE authoritatively confirms the row is
        non-current — which is what makes it race-safe.
        """
        by_version = sorted(records, key=lambda r: r.version, reverse=True)
        kept_activated = 0
        kept_ready = 0
        retire: list[int] = []
        for rec in by_version:
            if rec.is_current or rec.state == "building":
                continue  # never prune the live version or an in-flight build
            if rec.state == "activated":
                if kept_activated < self._retention.keep_activated:
                    kept_activated += 1
                    continue
                retire.append(rec.version)
            elif rec.state == "ready":
                if kept_ready < self._retention.keep_ready:
                    kept_ready += 1
                    continue
                retire.append(rec.version)
            elif rec.state == "retired":
                retire.append(rec.version)  # already retired: reclaim its table
        return retire, list(retire)

    def read_ledger_sql(self) -> sql.Composed:
        """Return the ledger read for one logical corpus (version/state/is_current).

        Only the columns the prune decision needs, keyed on the bound
        ``%(logical)s`` param and ordered by version so the read is deterministic.
        Run UNDER the advisory lock so the snapshot it returns is race-consistent
        with the prune that follows.
        """
        from psycopg import sql

        return sql.SQL(
            "SELECT version, state, is_current FROM neon_corpus_versions "
            "WHERE logical_name = %(logical)s ORDER BY version"
        )

    def prune(self, logical_name: str) -> list[int]:
        """Race-safely prune retired versions; return the versions dropped (B5).

        There is NO path that decides drops from a pre-lock snapshot. In one
        bounded-retry transaction (B16): acquire the advisory lock FIRST; re-read
        the ledger UNDER the lock (so any activation that raced ahead is reflected —
        its version is now ``is_current`` and excluded by :meth:`plan_prune`); then
        for each candidate run the guarded retire ``UPDATE ... WHERE is_current =
        false AND state IN (...) RETURNING version`` and **drop the physical table
        only if that UPDATE returned a row** — i.e. only after the ledger
        authoritatively confirms the version is non-current. So a version that is
        (or became) current is never dropped. Concurrent allocation contends on the
        same lock, so build-vs-prune fully serializes; the lock releases at commit.
        """
        from psycopg import sql

        validate_logical_name(logical_name)

        def work(conn: Any) -> list[int]:
            conn.execute(self._advisory_lock_stmt(), {"logical": logical_name})
            rows = conn.execute(
                self.read_ledger_sql(), {"logical": logical_name}
            ).fetchall()
            records = [
                NeonVersionRecord(
                    logical_name=logical_name,
                    version=version,
                    state=state,
                    is_current=is_current,
                )
                for version, state, is_current in rows
            ]
            retire, _ = self.plan_prune(records)
            dropped: list[int] = []
            for version in retire:
                params = {"logical": logical_name, "version": version}
                confirmed = conn.execute(
                    sql.SQL(
                        "UPDATE neon_corpus_versions "
                        "SET state = 'retired', retired_at = now() "
                        "WHERE logical_name = %(logical)s AND version = %(version)s "
                        "AND is_current = false "
                        "AND state IN ('activated', 'ready', 'retired') "
                        "RETURNING version"
                    ),
                    params,
                ).fetchall()
                if confirmed:  # authoritative: this row is non-current, safe to drop
                    conn.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}").format(
                            sql.Identifier(physical_table_name(logical_name, version))
                        )
                    )
                    dropped.append(version)
            return dropped

        return self._in_bounded_txn(work)

    # --- candidate queries (B13): vector / bm25 / hybrid ---------------------

    def vector_candidates_sql(
        self, spec: NeonTableSpec, where: sql.Composable | None = None
    ) -> tuple[sql.Composed, dict[str, Any]]:
        """Return the vector candidate ``(query, params)`` (``ORDER BY <=>``, B14).

        Selects the view columns plus the raw cosine distance as ``native_score``,
        ordered best-first by the ANN operator matching the index opclass. An
        optional pre-composed ``where`` (a metadata filter fragment built by the
        filter layer) is spliced in; the client never builds the filter itself,
        keeping it decoupled from ``filter_mapper``. The vector path carries no
        client-side bind params (the caller supplies ``vector``/``top_k`` at
        execution), so ``params`` is empty — returned for a uniform candidate API.
        """
        from psycopg import sql

        query = self._candidate_query(
            spec,
            score_expr=sql.SQL("embedding {op} %(vector)s::{cast}").format(
                op=sql.SQL(ANN_DISTANCE_OPERATOR),
                cast=sql.SQL(ANN_QUERY_PARAM_CAST),
            ),
            order=sql.SQL("ASC"),
            where=where,
        )
        return query, {}

    def bm25_candidates_sql(
        self,
        spec: NeonTableSpec,
        where: sql.Composable | None = None,
        *,
        schema: str,
    ) -> tuple[sql.Composed, dict[str, Any]]:
        """Return the BM25 candidate ``(query, params)`` (``<@> to_bm25query`` ASC).

        The scored column ``content_tsv`` is the LEFT operand of ``<@>``; the
        ``<@>`` score is negative (more-relevant is more-negative), so best
        candidates are ordered ``ASC``. ``to_bm25query(query, index)`` takes the
        query FIRST as a ``tsvector`` (the ``%(text)s`` placeholder run through
        ``to_tsvector`` with the SAME baked ``regconfig`` as the column, or scores
        would be tokenized inconsistently) and the BM25 index regclass SECOND.

        The index regclass is **schema-qualified** (``schema`` = the schema the
        corpus objects live in). Unlike the ``FROM`` view (read under the view
        owner's rights), ``to_bm25query`` and the ``::regclass`` name lookup run in
        the RO *invoker's* ``search_path`` — which need not include the corpus
        schema — so an unqualified name would fail to resolve for the very RO caller
        this enables. The schema and index are **bound params** (``%(bm25_schema)s``
        / ``%(bm25_index)s``) fed through ``quote_ident`` (not inlined literals), so
        they are injection-safe and correctly quoted; the returned ``params`` carry
        the client-known values for the caller to merge with ``text``/``top_k``.
        """
        from psycopg import sql

        bm25_index = index_names(spec.logical_name, spec.version)["bm25"]
        tsconfig = validate_text_search_config(spec.text_search_config)
        score_expr = sql.SQL(
            "content_tsv <@> to_bm25query("
            "to_tsvector({tsconfig}::regconfig, %(text)s), "
            "(quote_ident(%(bm25_schema)s) || '.' || quote_ident(%(bm25_index)s))"
            "::regclass)"
        ).format(tsconfig=sql.Literal(tsconfig))
        query = self._candidate_query(
            spec, score_expr=score_expr, order=sql.SQL("ASC"), where=where
        )
        return query, {"bm25_schema": schema, "bm25_index": bm25_index}

    def _candidate_query(
        self,
        spec: NeonTableSpec,
        *,
        score_expr: sql.Composable,
        order: sql.Composable,
        where: sql.Composable | None,
    ) -> sql.Composed:
        """Assemble a candidate ``SELECT`` from a score expression + ordering.

        Shared by the vector and BM25 paths: projects the read columns (the heavy
        ``embedding`` and ``content_tsv`` are omitted from the OUTPUT, B4) plus the
        score as ``native_score``, applies the optional filter, orders by the score,
        and bounds the row count with a ``%(top_k)s`` placeholder.

        Reads the stable owner-rights **view**, but RO privilege differs by path:
        the vector and filter paths run entirely through the ``security_invoker =
        false`` view (the scan, including the ANN/GIN index, executes with owner
        rights), so they need no physical-table grant. The BM25 path additionally
        calls ``to_bm25query`` with the version's index regclass — that call runs
        with the **RO invoker's** rights (not the view owner's) and reads the bm25
        index's base-table stats, so RO also needs ``SELECT`` on the version tables
        (issued via the writer's default privileges, see credentials/provision).
        """
        from psycopg import sql

        view = sql.Identifier(view_name(spec.logical_name))
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in READ_COLUMNS)
        where_clause = (
            sql.SQL(" WHERE {}").format(where) if where is not None else sql.SQL("")
        )
        return sql.SQL(
            "SELECT {columns}, {score} AS native_score FROM {view}{where} "
            "ORDER BY {score} {order} LIMIT %(top_k)s"
        ).format(
            columns=columns,
            score=score_expr,
            view=view,
            where=where_clause,
            order=order,
        )

    # --- reads: count / sample / neighbors / top-level / scan (B6/B9) --------

    def count_sql(self, logical_name: str) -> sql.Composed:
        """Return ``SELECT count(*)`` against the active reader view."""
        from psycopg import sql

        return sql.SQL("SELECT count(*) FROM {}").format(
            sql.Identifier(view_name(logical_name))
        )

    def sample_sql(self, logical_name: str) -> sql.Composed:
        """Return a random-sample query (``ORDER BY random() LIMIT %(n)s``).

        Filters out chunks shorter than ``%(min_chars)s`` so callers can request a
        minimum length; the bound placeholders keep it injection-safe.
        """
        from psycopg import sql

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in READ_COLUMNS)
        return sql.SQL(
            "SELECT {columns} FROM {view} WHERE length(content) >= %(min_chars)s "
            "ORDER BY random() LIMIT %(n)s"
        ).format(columns=columns, view=sql.Identifier(view_name(logical_name)))

    def neighbors_sql(self, logical_name: str) -> sql.Composed:
        """Return the adjacent-chunk query for context previews (same file).

        Fetches the chunks at ``%(prev_index)s`` and ``%(next_index)s`` within
        ``%(source_file)s`` so the source can build a neighboring-context preview.
        """
        from psycopg import sql

        return sql.SQL(
            "SELECT chunk_index, content FROM {view} "
            "WHERE source_file = %(source_file)s "
            "AND chunk_index IN (%(prev_index)s, %(next_index)s) "
            "ORDER BY chunk_index"
        ).format(view=sql.Identifier(view_name(logical_name)))

    def top_level_sql(self, logical_name: str) -> sql.Composed:
        """Return the first-chunk-per-file query (the top-level chunks).

        ``DISTINCT ON (source_file)`` with the scan order yields chunk index 0 of
        each source file — the natural top-level entry points into a corpus.
        """
        from psycopg import sql

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in READ_COLUMNS)
        return sql.SQL(
            "SELECT DISTINCT ON (source_file) {columns} FROM {view} "
            "ORDER BY source_file, chunk_index, id"
        ).format(columns=columns, view=sql.Identifier(view_name(logical_name)))

    def scan_page_sql(self, logical_name: str, *, after: bool) -> sql.Composed:
        """Return one keyset page of the deterministic full-corpus scan (B9).

        Total order ``(source_file, chunk_index, id)`` over the typed NOT NULL
        columns (never JSONB extraction), backed by the ``scan`` btree. When
        *after* is True the row-tuple cursor ``(source_file, chunk_index, id) >
        (%s, %s, %s)`` pages past the previous batch; the first page omits it.
        """
        from psycopg import sql

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in READ_COLUMNS)
        cursor = (
            sql.SQL(
                "WHERE (source_file, chunk_index, id) > "
                "(%(after_file)s, %(after_index)s, %(after_id)s) "
            )
            if after
            else sql.SQL("")
        )
        return sql.SQL(
            "SELECT {columns} FROM {view} {cursor}"
            "ORDER BY source_file, chunk_index, id LIMIT %(batch_size)s"
        ).format(
            columns=columns,
            view=sql.Identifier(view_name(logical_name)),
            cursor=cursor,
        )

    def scan_chunks(self, logical_name: str, batch_size: int = 1000) -> Any:
        """Yield every row of the active corpus in the frozen scan order (B9).

        A generator over keyset pages: deterministic and pageable. Yields raw row
        tuples (``READ_COLUMNS`` order) — the ``Chunk`` mapping lives in the
        ChunkSource, keeping the client ``Chunk``-free. ``id``/``source_file``/
        ``chunk_index`` drive the next cursor.

        Low-level primitive: each page runs through :meth:`execute`, which COMMITS
        per statement, so a concurrent activation CAN swap the reader view between
        pages. For a snapshot-consistent full-corpus scan (qa-gen), use
        :meth:`scan_in_snapshot`, which holds one transaction + the shared advisory
        lock for the whole iterator.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        id_i = READ_COLUMNS.index("id")
        file_i = READ_COLUMNS.index("source_file")
        index_i = READ_COLUMNS.index("chunk_index")
        after: tuple[Any, Any, Any] | None = None
        while True:
            query = self.scan_page_sql(logical_name, after=after is not None)
            params: dict[str, Any] = {"batch_size": batch_size}
            if after is not None:
                params["after_file"], params["after_index"], params["after_id"] = after
            rows = self.execute(query, params)
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < batch_size:
                return
            last = rows[-1]
            after = (last[file_i], last[index_i], last[id_i])

    def _resolve_current_version(self, conn: Any, logical_name: str) -> int:
        """Return the current published version (ledger read), or raise if none.

        Read on *conn* (inside the caller's locked transaction), mirroring
        :func:`query._resolve_current_spec`, so a scan can fail fast with a clear
        error rather than hitting a missing view.
        """
        cur = conn.execute(self.read_ledger_sql(), {"logical": logical_name})
        for version, _state, is_current in cur.fetchall():
            if is_current:
                return version
        raise LookupError(
            f"neon corpus {logical_name!r} has no current published version"
        )

    def scan_in_snapshot(self, logical_name: str, batch_size: int = 1000) -> Any:
        """Snapshot-consistent full-corpus scan in the frozen order (B6/B9).

        Streams every row of the active version in ``(source_file, chunk_index,
        id)`` order within ONE read transaction that holds the shared per-logical
        advisory lock for the iterator's whole lifetime. Because activation,
        rollback, and prune each take the EXCLUSIVE advisory lock on the same key
        as their first step (before any ``CREATE OR REPLACE VIEW`` or ``DROP
        TABLE``), they block while this scan runs — so the reader view cannot be
        re-pointed and no scanned version table can be dropped mid-scan, and every
        page resolves to one consistent version. Versions are build-once-immutable,
        so default (READ COMMITTED) isolation is sufficient; the lock, not the
        isolation level, provides the stable snapshot.

        Runs on a DEDICATED connection (not the client's cached one): the generator
        suspends between pages with its transaction open, so sharing the cached
        connection would corrupt any other read issued while the scan is paused.
        The connection is always closed (``try/finally``), releasing the
        xact-scoped lock even if the caller abandons the iterator.

        Latency note: a long full-corpus scan holds the shared lock for its whole
        duration, so a concurrent activation waits until the scan ends (and, by
        fair lock queueing, briefly delays reads behind it). Acceptable for the
        offline qa-gen materialization this serves.
        """
        import psycopg

        validate_logical_name(logical_name)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        id_i = READ_COLUMNS.index("id")
        file_i = READ_COLUMNS.index("source_file")
        index_i = READ_COLUMNS.index("chunk_index")

        conn = psycopg.connect(
            self._dsn_provider(), prepare_threshold=None, **CONNECT_PARAMS
        )
        try:
            with conn.transaction():
                conn.execute(
                    self._advisory_lock_shared_stmt(), {"logical": logical_name}
                )
                self._resolve_current_version(conn, logical_name)  # fail-fast
                after: tuple[Any, Any, Any] | None = None
                while True:
                    query = self.scan_page_sql(logical_name, after=after is not None)
                    params: dict[str, Any] = {"batch_size": batch_size}
                    if after is not None:
                        (
                            params["after_file"],
                            params["after_index"],
                            params["after_id"],
                        ) = after
                    rows = conn.execute(query, params).fetchall()
                    if not rows:
                        return
                    for row in rows:
                        yield row
                    if len(rows) < batch_size:
                        return
                    last = rows[-1]
                    after = (last[file_i], last[index_i], last[id_i])
        finally:
            conn.close()

    # --- end-to-end build lifecycle (B14) ------------------------------------

    def build_version(
        self,
        spec: NeonTableSpec,
        rows: list[tuple[Any, ...]],
    ) -> None:
        """Build one physical corpus version in the frozen lifecycle order (B14).

        Verify extensions present (installed out-of-band by ``provision.py``) ->
        register pgvector types (the extension exists) -> ledger + current-pointer
        index -> allocate the version (``building``, under the advisory lock) ->
        create the table -> populate -> build the ANN + BM25 + auxiliary indexes
        (after the load) -> ``VACUUM ANALYZE`` in autocommit outside any
        transaction -> mark ``ready``.

        Activation is a separate, explicitly-triggered step (:meth:`activate`) so a
        freshly-built ``ready`` version is staged, then published atomically with
        its RO grants.
        """
        self.require_extensions()
        self.register_vector_types()
        for statement in self.create_ledger_sql():
            self.execute(statement)
        self.allocate_version(spec)
        self.execute(self.create_table_sql(spec))
        if rows:
            self._insert_many(spec, rows)
        self.execute(self.create_ann_index_sql(spec))
        self.execute(self.create_bm25_index_sql(spec))
        for statement in self.create_aux_indexes_sql(spec):
            self.execute(statement)
        self.vacuum(spec)
        self.execute(
            self.mark_ready_sql(spec),
            {"logical": spec.logical_name, "version": spec.version},
        )

    def _insert_many(self, spec: NeonTableSpec, rows: list[tuple[Any, ...]]) -> None:
        """Bulk-insert *rows* via a single ``executemany``, committed as a unit."""
        conn = self._live_conn()
        with conn.cursor() as cur:
            cur.executemany(self.insert_rows_sql(spec), rows)
        conn.commit()
