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
``CREATE EXTENSION ... CASCADE`` (so the vector type + Lakebase BM25 access
method exist) -> **register pgvector types on the connection, only after the
extension exists** -> ledger + current-pointer index -> allocate a version and
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

# PROVISIONAL ANN choice (see schema.PROVISIONAL_ANN_OPTIONS). Provisional primary
# is native ``lakebase_ann`` on ``vector(3072)`` with the cosine opclass. Slice 3
# must confirm the access method / vector type / opclass / operator on the live DB
# before it is frozen — the client emits the primary form and slice 3 swaps the
# proven one (the ``hnsw``/``halfvec_cosine_ops`` fallback uses the SAME operator).
ANN_ACCESS_METHOD = "lakebase_ann"
ANN_VECTOR_TYPE = "vector(3072)"
ANN_OPCLASS = "vector_cosine_ops"
# Ordering operator, matching the cosine opclass (B14). DEVIATION: the slice-2
# brief wrote ``<->`` for the vector ORDER BY, but ``<->`` is L2 (Euclidean) — it
# does NOT match ``vector_cosine_ops`` and would make the planner skip the ANN
# index. The operator matching a cosine opclass is ``<=>`` (both for
# ``lakebase_ann`` and the vanilla ``hnsw``/``halfvec_cosine_ops`` fallback), so
# we emit the opclass-matching operator the brief actually asked for. Slice 3
# confirms against the live DB.
ANN_DISTANCE_OPERATOR = "<=>"

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
        """
        import psycopg

        dsn = self._dsn_provider()
        self._conn = psycopg.connect(dsn)
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

    def execute(
        self, query: sql.Composable, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute one composable statement with bound params and return rows.

        Single-statement autocommit-equivalent: each successful statement is
        committed so a read releases its snapshot and a DDL persists, leaving no
        open transaction to block the ``VACUUM`` autocommit toggle. Grouped atomic
        work goes through :meth:`execute_in_transaction`, not here.

        Bounded reconnect (B16): a cached connection killed by autosuspend raises
        an ``OperationalError``/``InterfaceError`` on first use; if the handle is
        genuinely dead we drop it, reconnect **once**, and retry. A live-connection
        error (bad SQL, a constraint or lock failure) is re-raised immediately and
        never retried, and a second dead-connection failure propagates too — the
        retry is bounded, never a reconnect storm.
        """
        import psycopg

        for attempt in range(2):  # original attempt + one bounded reconnect
            conn = self._live_conn()
            try:
                cur = conn.execute(query, params or {})
                rows = cur.fetchall() if cur.description is not None else []
                conn.commit()
                return rows
            except (psycopg.OperationalError, psycopg.InterfaceError):
                dead = self._is_dead(conn)
                self._conn = None  # drop the handle; next _live_conn reconnects
                if attempt == 0 and dead:
                    continue  # autosuspend-killed conn: reconnect and retry once
                raise  # live-conn error, or a second dead-conn failure
        raise AssertionError("unreachable")  # the loop always returns or raises

    def execute_in_transaction(self, statements: list[sql.Composable]) -> None:
        """Run *statements* as one all-or-nothing transaction (B5).

        Used to publish (or roll back) a version so the ledger update, the ``CREATE
        OR REPLACE VIEW``, and the RO grants commit or roll back together. The
        connection is refreshed via :meth:`_live_conn` first so an autosuspend-
        killed cached connection reconnects *before* the transaction opens (a
        mid-transaction reconnect would silently drop the advisory lock). On the
        first failing statement the whole transaction is rolled back and the error
        re-raised, leaving no partial state.
        """
        conn = self._live_conn()
        try:
            for statement in statements:
                conn.execute(statement)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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

    def create_extensions_sql(self) -> list[sql.Composed]:
        """Return the ``CREATE EXTENSION IF NOT EXISTS <e> CASCADE`` statements (B14).

        CASCADE so a not-yet-installed dependency of the BM25/ANN surface is
        pulled in automatically. Emitted first in the build so the vector type and
        Lakebase access methods exist before anything references them.
        """
        from psycopg import sql

        return [
            sql.SQL("CREATE EXTENSION IF NOT EXISTS {} CASCADE").format(
                sql.Identifier(ext)
            )
            for ext in REQUIRED_EXTENSIONS
        ]

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
        bound ``sql.Literal`` (never interpolated, B4). The embedding column uses
        the PROVISIONAL ANN vector type; slice 3 freezes the proven type.
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
            "embedding {vtype}, "
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

    def _advisory_lock_sql(self, logical_name: str) -> sql.Composed:
        """Return the per-logical ``pg_advisory_xact_lock`` acquisition (B5).

        Every allocation, activation, rollback, and prune acquires the SAME
        transaction-scoped lock keyed on the logical name, so concurrent
        build-vs-prune (and two concurrent builds) serialize and the lock releases
        automatically at commit/rollback. ``hashtext`` maps the name to the lock's
        bigint key.
        """
        from psycopg import sql

        validate_logical_name(logical_name)
        return sql.SQL("SELECT pg_advisory_xact_lock(hashtext({}))").format(
            sql.Literal(logical_name)
        )

    def allocate_version_sql(
        self, logical_name: str, version: int
    ) -> list[sql.Composed]:
        """Return the txn that reserves *version* as ``building`` under the lock (B5).

        Acquires the advisory lock FIRST, then inserts the ``building`` ledger row.
        Holding the lock across the insert is what makes concurrent allocation
        race-safe: two builders cannot both reserve the same next version.
        """
        from psycopg import sql

        validate_version(version)
        return [
            self._advisory_lock_sql(logical_name),
            sql.SQL(
                "INSERT INTO neon_corpus_versions (logical_name, version, state) "
                "VALUES ({logical}, {version}, 'building')"
            ).format(logical=sql.Literal(logical_name), version=sql.Literal(version)),
        ]

    def mark_ready_sql(self, spec: NeonTableSpec) -> sql.Composed:
        """Return the ledger update transitioning ``building`` -> ``ready`` (B5).

        Guarded by ``state = 'building'`` so the frozen transition is enforced in
        SQL alongside the DB CHECK domain.
        """
        from psycopg import sql

        return sql.SQL(
            "UPDATE neon_corpus_versions SET state = 'ready', ready_at = now() "
            "WHERE logical_name = {logical} AND version = {version} "
            "AND state = 'building'"
        ).format(
            logical=sql.Literal(spec.logical_name),
            version=sql.Literal(spec.version),
        )

    def activate_version_sql(
        self, spec: NeonTableSpec, grant: ReadGrantSpec
    ) -> list[sql.Composed]:
        """Return the single-transaction statements that publish a version (B5).

        Ordered, all under one ``pg_advisory_xact_lock`` on the logical name:
        acquire the lock; clear the prior ``is_current`` row; set this version
        ``activated``/``is_current``; ``CREATE OR REPLACE VIEW`` (owner-rights,
        explicit columns); and issue the RO grants so a first-create view is
        readable atomically with publication. The partial unique index enforces
        the single-current invariant. Runs via :meth:`execute_in_transaction`.
        """
        from psycopg import sql

        validate_version(spec.version)
        view = sql.Identifier(view_name(spec.logical_name))
        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
        return [
            self._advisory_lock_sql(spec.logical_name),
            sql.SQL(
                "UPDATE neon_corpus_versions SET is_current = false "
                "WHERE logical_name = {logical} AND is_current"
            ).format(logical=sql.Literal(spec.logical_name)),
            sql.SQL(
                "UPDATE neon_corpus_versions "
                "SET state = 'activated', is_current = true, activated_at = now() "
                "WHERE logical_name = {logical} AND version = {version}"
            ).format(
                logical=sql.Literal(spec.logical_name),
                version=sql.Literal(spec.version),
            ),
            sql.SQL(
                "CREATE OR REPLACE VIEW {view} WITH (security_invoker = false) AS "
                "SELECT {columns} FROM {table}"
            ).format(view=view, columns=columns, table=table),
            *self.read_grant_sql(grant),
        ]

    def rollback_version_sql(
        self, logical_name: str, target_version: int
    ) -> list[sql.Composed]:
        """Return the statements re-pointing ``is_current`` to a prior version (B5).

        Non-destructive and O(1): prior physical tables are retained, so rollback
        only re-points the ledger + view under the advisory lock. The target must
        be an already-``activated`` version (``WHERE state = 'activated'`` guards
        it — you cannot roll forward onto something never published), and its state
        is left unchanged (rollback flips ``is_current``, not ``state``).
        """
        from psycopg import sql

        validate_version(target_version)
        view = sql.Identifier(view_name(logical_name))
        table = sql.Identifier(physical_table_name(logical_name, target_version))
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
        return [
            self._advisory_lock_sql(logical_name),
            sql.SQL(
                "UPDATE neon_corpus_versions SET is_current = false "
                "WHERE logical_name = {logical} AND is_current"
            ).format(logical=sql.Literal(logical_name)),
            sql.SQL(
                "UPDATE neon_corpus_versions SET is_current = true "
                "WHERE logical_name = {logical} AND version = {version} "
                "AND state = 'activated'"
            ).format(
                logical=sql.Literal(logical_name),
                version=sql.Literal(target_version),
            ),
            sql.SQL(
                "CREATE OR REPLACE VIEW {view} WITH (security_invoker = false) AS "
                "SELECT {columns} FROM {table}"
            ).format(view=view, columns=columns, table=table),
        ]

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

        This is the *decision*; :meth:`prune_versions_sql` emits the guarded DDL
        that executes it under the advisory lock, which is what makes it race-safe.
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

    def read_ledger_sql(self, logical_name: str) -> sql.Composed:
        """Return the ledger read for one logical corpus (version/state/is_current).

        Only the columns the prune decision needs. Ordered by version so the read
        is deterministic; meant to be run UNDER the advisory lock so the snapshot
        it returns is race-consistent with the prune that follows.
        """
        from psycopg import sql

        validate_logical_name(logical_name)
        return sql.SQL(
            "SELECT version, state, is_current FROM neon_corpus_versions "
            "WHERE logical_name = {logical} ORDER BY version"
        ).format(logical=sql.Literal(logical_name))

    def _prune_body_sql(
        self, logical_name: str, retire: list[int], drop_tables: list[int]
    ) -> list[sql.Composed]:
        """Return the DROP + guarded-retire statements (no lock; caller holds it).

        Each retire is guarded by ``is_current = false`` — belt-and-suspenders so
        that even if a snapshot were stale, the live version's ledger row is never
        retired.
        """
        from psycopg import sql

        statements: list[sql.Composed] = []
        for version in drop_tables:
            table = sql.Identifier(physical_table_name(logical_name, version))
            statements.append(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
        for version in retire:
            statements.append(
                sql.SQL(
                    "UPDATE neon_corpus_versions "
                    "SET state = 'retired', retired_at = now() "
                    "WHERE logical_name = {logical} AND version = {version} "
                    "AND is_current = false"
                ).format(
                    logical=sql.Literal(logical_name),
                    version=sql.Literal(version),
                )
            )
        return statements

    def prune_versions_sql(
        self, logical_name: str, records: list[NeonVersionRecord]
    ) -> list[sql.Composed]:
        """Return the guarded prune transaction for a PRE-READ snapshot (B5).

        Structure, all under ONE advisory lock (same key as allocation/activation,
        so a concurrent build cannot interleave): acquire ``pg_advisory_xact_lock``,
        then ``DROP TABLE IF EXISTS`` + mark ``retired`` (guarded by ``is_current =
        false``) for each version :meth:`plan_prune` selected. Empty when nothing is
        prunable.

        This is the static form for a snapshot the caller already holds. For full
        race-safety prefer :meth:`prune`, which re-reads the ledger UNDER the lock
        so a version that races into ``is_current`` after the snapshot can never
        have its table dropped.
        """
        retire, drop_tables = self.plan_prune(records)
        if not retire:
            return []
        return [
            self._advisory_lock_sql(logical_name),
            *self._prune_body_sql(logical_name, retire, drop_tables),
        ]

    def prune(self, logical_name: str) -> list[int]:
        """Race-safely prune retired versions; return the versions dropped (B5).

        The load-bearing ordering: **acquire the advisory lock FIRST, then re-read
        the ledger under it**, so the prune decision sees any activation that raced
        ahead of us — the freshly-published version is now ``is_current`` in the
        snapshot and is excluded, so its physical table is never dropped. The lock,
        read, drops, and retires all commit as one transaction (rolled back on any
        error); the lock releases at commit. Concurrent allocation contends on the
        same lock, so build-vs-prune fully serializes.
        """
        conn = self._live_conn()
        try:
            conn.execute(self._advisory_lock_sql(logical_name))
            rows = conn.execute(self.read_ledger_sql(logical_name)).fetchall()
            records = [
                NeonVersionRecord(
                    logical_name=logical_name,
                    version=version,
                    state=state,
                    is_current=is_current,
                )
                for version, state, is_current in rows
            ]
            retire, drop_tables = self.plan_prune(records)
            for statement in self._prune_body_sql(logical_name, retire, drop_tables):
                conn.execute(statement)
            conn.commit()
            return drop_tables
        except Exception:
            conn.rollback()
            raise

    # --- candidate queries (B13): vector / bm25 / hybrid ---------------------

    def vector_candidates_sql(
        self, spec: NeonTableSpec, where: sql.Composable | None = None
    ) -> sql.Composed:
        """Return the vector candidate query (``ORDER BY embedding <=> %s``, B14).

        Selects the view columns plus the raw cosine distance as ``native_score``,
        ordered best-first by the ANN operator matching the index opclass. An
        optional pre-composed ``where`` (a metadata filter fragment built by the
        filter layer) is spliced in; the client never builds the filter itself,
        keeping it decoupled from ``filter_mapper``.
        """
        from psycopg import sql

        return self._candidate_query(
            spec,
            score_expr=sql.SQL("embedding {op} %(vector)s").format(
                op=sql.SQL(ANN_DISTANCE_OPERATOR)
            ),
            order=sql.SQL("ASC"),
            where=where,
        )

    def bm25_candidates_sql(
        self, spec: NeonTableSpec, where: sql.Composable | None = None
    ) -> sql.Composed:
        """Return the BM25 candidate query (``<@> to_bm25query(...)`` ASC, B13).

        The scored column ``content_tsv`` is the LEFT operand of ``<@>``; the
        ``<@>`` score is negative (more-relevant is more-negative), so best
        candidates are ordered ``ASC``. ``to_bm25query(query, index)`` takes the
        query FIRST as a ``tsvector`` (the ``%(text)s`` placeholder run through
        ``to_tsvector`` with the SAME baked ``regconfig`` as the column, or scores
        would be tokenized inconsistently) and the BM25 index regclass SECOND. The
        same expression drives both the projected ``native_score`` and the ordering.
        """
        from psycopg import sql

        bm25_index = index_names(spec.logical_name, spec.version)["bm25"]
        tsconfig = validate_text_search_config(spec.text_search_config)
        score_expr = sql.SQL(
            "content_tsv <@> to_bm25query("
            "to_tsvector({tsconfig}::regconfig, %(text)s), {index}::regclass)"
        ).format(tsconfig=sql.Literal(tsconfig), index=sql.Literal(bm25_index))
        return self._candidate_query(
            spec, score_expr=score_expr, order=sql.SQL("ASC"), where=where
        )

    def hybrid_candidates_sql(
        self, spec: NeonTableSpec, where: sql.Composable | None = None
    ) -> tuple[sql.Composed, sql.Composed]:
        """Return the ``(vector, bm25)`` candidate queries for hybrid fusion (B13).

        Two independent ranked candidate lists; RRF fusion over them is owned by
        the query layer (Slice 1), not the client. The same optional ``where`` is
        applied to both so the filtered candidate sets are consistent.
        """
        return (
            self.vector_candidates_sql(spec, where=where),
            self.bm25_candidates_sql(spec, where=where),
        )

    def _candidate_query(
        self,
        spec: NeonTableSpec,
        *,
        score_expr: sql.Composable,
        order: sql.Composable,
        where: sql.Composable | None,
    ) -> sql.Composed:
        """Assemble a candidate ``SELECT`` from a score expression + ordering.

        Shared by the vector and BM25 paths: projects the view columns plus the
        score as ``native_score``, applies the optional filter, orders by the score,
        and bounds the row count with a ``%(top_k)s`` placeholder.
        """
        from psycopg import sql

        table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
        where_clause = (
            sql.SQL(" WHERE {}").format(where) if where is not None else sql.SQL("")
        )
        return sql.SQL(
            "SELECT {columns}, {score} AS native_score FROM {table}{where} "
            "ORDER BY {score} {order} LIMIT %(top_k)s"
        ).format(
            columns=columns,
            score=score_expr,
            table=table,
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

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
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

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
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

        columns = sql.SQL(", ").join(sql.Identifier(c) for c in VIEW_COLUMNS)
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

        A generator over keyset pages: deterministic and pageable so full-corpus
        materialization (qa-gen) is reproducible run to run. Yields raw row tuples
        (``VIEW_COLUMNS`` order) — the ``Chunk`` mapping lives in the ChunkSource,
        keeping the client ``Chunk``-free. ``id``/``source_file``/``chunk_index``
        drive the next cursor.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        id_i = VIEW_COLUMNS.index("id")
        file_i = VIEW_COLUMNS.index("source_file")
        index_i = VIEW_COLUMNS.index("chunk_index")
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

    # --- end-to-end build lifecycle (B14) ------------------------------------

    def build_version(
        self,
        spec: NeonTableSpec,
        rows: list[tuple[Any, ...]],
    ) -> None:
        """Build one physical corpus version in the frozen lifecycle order (B14).

        Extensions (CASCADE) -> register pgvector types (only now the extension
        exists) -> ledger + current-pointer index -> allocate the version
        (``building``, under the advisory lock) -> create the table -> populate ->
        build the ANN + BM25 + auxiliary indexes (after the load) -> ``VACUUM
        ANALYZE`` in autocommit outside any transaction -> mark ``ready``.

        Activation is a separate, explicitly-triggered step (:meth:`activate_version_sql`
        via :meth:`execute_in_transaction`) so a freshly-built ``ready`` version is
        staged, then published atomically with its RO grants.
        """
        for statement in self.create_extensions_sql():
            self.execute(statement)
        self.register_vector_types()
        for statement in self.create_ledger_sql():
            self.execute(statement)
        self.execute_in_transaction(
            self.allocate_version_sql(spec.logical_name, spec.version)
        )
        self.execute(self.create_table_sql(spec))
        if rows:
            self._insert_many(spec, rows)
        self.execute(self.create_ann_index_sql(spec))
        self.execute(self.create_bm25_index_sql(spec))
        for statement in self.create_aux_indexes_sql(spec):
            self.execute(statement)
        self.vacuum(spec)
        self.execute(self.mark_ready_sql(spec))

    def _insert_many(self, spec: NeonTableSpec, rows: list[tuple[Any, ...]]) -> None:
        """Bulk-insert *rows* via a single ``executemany``, committed as a unit."""
        conn = self._live_conn()
        with conn.cursor() as cur:
            cur.executemany(self.insert_rows_sql(spec), rows)
        conn.commit()
