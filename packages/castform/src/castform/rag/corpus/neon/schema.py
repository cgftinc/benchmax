"""Physical table schema and versioned-replace lifecycle for the Neon corpus.

Contract-freeze artifact (Slice A). The identifier helpers, allowlists, and
validation here are implemented (they are the frozen safety contract); the DDL
*assembly* and execution are Slice 2 stubs that return ``psycopg.sql`` composables
built from these helpers — never interpolated strings.

Versioned-replace lifecycle
---------------------------
A corpus is addressed by a stable *logical name*. Each ingest builds a fresh
*physical* table ``<logical>__v<version>`` with its own indexes; readers query a
stable owner-rights view under the logical name. A per-version *ledger*
(``neon_corpus_versions``) tracks each version's state (building -> ready ->
activated -> retired) with timestamps, so concurrent ingest, enumerate, prune,
and build-vs-ready are all well-defined. Activation flips the ledger's active
row AND re-points the view in one transaction under a per-logical advisory lock;
rollback re-points to any prior ``ready``/``activated`` version (old physical
tables are retained until pruned, so rollback is O(1) and non-destructive).

Chunk identity is content+metadata derived (``rag/chunkers/models.py``:
``hash = sha256(metadata_str + "\\n" + content)``), so re-ingesting changed
content yields new ids — hence versioned *replace*, not in-place upsert.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from psycopg import sql

# --- Contract #7: embedding dim / metric (frozen) ------------------------------

EMBEDDING_DIM = 3072
"""Embedding vector dimensionality (text-embedding-3-large)."""

DISTANCE_METRIC = "cosine"
"""Vector distance metric. Fixed for the managed table; index uses cosine ops."""

DEFAULT_TEXT_SEARCH_CONFIG = "pg_catalog.english"
"""Default Postgres text-search config for the tsvector generated column."""

# B4: regconfig is never interpolated raw — only a value from this allowlist is
# accepted, then emitted as a bound ``sql.Literal`` cast to ``regconfig``.
ALLOWED_TEXT_SEARCH_CONFIGS: frozenset[str] = frozenset(
    {"pg_catalog.english", "pg_catalog.simple"}
)

MAX_IDENTIFIER_BYTES = 63
"""Postgres identifier length limit; longer names are hash-suffixed."""

# Explicit column list for the stable view — never ``SELECT *`` (B4). ``id`` is
# the chunk hash; ``source_file``/``chunk_index`` are typed non-null columns that
# back deterministic scan ordering (B6).
TABLE_COLUMNS: tuple[str, ...] = (
    "id",
    "content",
    "metadata",
    "embedding",
    "source_file",
    "chunk_index",
    "content_tsv",
)
VIEW_COLUMNS: tuple[str, ...] = TABLE_COLUMNS


# --- B1 (Group 2): ANN access method / vector type / opclass — PROVEN (Slice 3) --
# Verified live on Neon Lakebase (PG 18.4, lakebase_vector 1.0.0-dev): the native
# ``lakebase_ann`` access method indexes a full-precision ``vector(3072)`` column
# with ``vector_cosine_ops`` and is used by the planner (EXPLAIN: ``Index Scan
# using ..._ann``) for a cosine ``ORDER BY``. Unlike pgvector's ``hnsw`` — which
# rejects >2000 dims — ``lakebase_ann`` has no dimension cap at 3072, so the
# ``halfvec`` workaround is NOT required for correctness (it also builds and is
# kept documented below as the storage-saving alternative). The query param MUST be
# cast to ``vector`` (a bound Python list binds as ``float8[]`` and the cast-less
# ``<=>`` errors "operator does not exist: vector <=> double precision[]"). The
# type, opclass, distance operator, and query-param cast are frozen together as one
# coherent unit — changing any one alone breaks index use. See CONTRACT.md §1.
PROVEN_ANN_DDL: dict[str, str] = {
    "access_method": "lakebase_ann",
    "vector_type": "vector(3072)",
    "opclass": "vector_cosine_ops",
    "operator": "<=>",
    "query_param_cast": "vector",
}

# Storage-saving alternative (also live-verified to build + be planner-used): halves
# the bytes per vector at a small recall cost. NOT the frozen primary — documented
# so a future cost-driven switch is a one-line, already-proven change.
ANN_HALFVEC_ALTERNATIVE: dict[str, str] = {
    "access_method": "lakebase_ann",
    "vector_type": "halfvec(3072)",
    "opclass": "halfvec_cosine_ops",
    "operator": "<=>",
    "query_param_cast": "halfvec",
}


VersionState = Literal["building", "ready", "activated", "retired"]
"""Lifecycle states for one physical corpus version (B5).

``activated`` is *historical* — the version has been published at least once.
Which single version is *currently* published is tracked separately by the
``is_current`` flag (see the current-pointer invariant below), so a rollback flips
``is_current`` between two ``activated`` versions without changing their state.
"""

# Frozen legal state transitions (enforced in Slice 2 alongside the DB CHECK).
VERSION_STATE_TRANSITIONS: dict[VersionState, tuple[VersionState, ...]] = {
    "building": ("ready",),
    "ready": ("activated", "retired"),
    "activated": ("retired",),
    "retired": (),
}


@dataclass(frozen=True)
class NeonTableSpec:
    """Frozen description of one physical corpus-version table.

    Args:
        logical_name: Stable logical corpus name (readers address this).
        version: Monotonic physical version number (>= 1).
        embedding_dim: Vector column dimensionality. Defaults to EMBEDDING_DIM.
        distance_metric: Vector distance metric. Defaults to DISTANCE_METRIC.
        text_search_config: Postgres ``regconfig`` for the tsvector column; must
            be in ALLOWED_TEXT_SEARCH_CONFIGS. Baked per physical version (a
            config change requires a new version, not an ALTER).
    """

    logical_name: str
    version: int
    embedding_dim: int = EMBEDDING_DIM
    distance_metric: str = DISTANCE_METRIC
    text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG


@dataclass(frozen=True)
class NeonVersionRecord:
    """One row of the per-version ledger (B5).

    Args:
        logical_name: Logical corpus this version belongs to.
        version: Physical version number.
        state: Lifecycle state (historical).
        is_current: Whether this is the single currently-published version the
            reader view points to (at most one true per logical name).
        created_at: When the physical table build started (epoch seconds).
        ready_at: When the build finished and indexes were valid, else None.
        activated_at: When this version was first published, else None.
        retired_at: When this version was retired, else None.
    """

    logical_name: str
    version: int
    state: VersionState
    is_current: bool = False
    created_at: float = 0.0
    ready_at: float | None = None
    activated_at: float | None = None
    retired_at: float | None = None


@dataclass(frozen=True)
class RetentionPolicy:
    """Version retention / pruning policy (B5).

    Args:
        keep_activated: Number of most-recent activated versions to retain
            (>= 2 so rollback always has a target).
        keep_ready: Number of most-recent non-activated ``ready`` versions to
            retain for fast promotion.
    """

    keep_activated: int = 2
    keep_ready: int = 1

    def __post_init__(self) -> None:
        # Enforce the documented minimum, not just describe it: rollback needs a
        # prior activated version to fall back to.
        if self.keep_activated < 2:
            raise ValueError("keep_activated must be >= 2 so rollback has a target")
        if self.keep_ready < 1:
            raise ValueError("keep_ready must be >= 1")


DEFAULT_RETENTION = RetentionPolicy()


@dataclass(frozen=True)
class ReadGrantSpec:
    """RO grant the ingest role must issue so the search role can read (B5).

    The stable view is owner-rights (``security_invoker = false``), so the RO
    role needs only schema ``USAGE`` and ``SELECT`` on the view — never on the
    physical version tables. These grants must be (re)issued on FIRST view
    creation (``CREATE OR REPLACE VIEW`` preserves an existing ACL but the
    first create has none).

    Args:
        schema: Schema holding the corpus objects.
        view: Stable logical view name.
        ro_role: The read-only role receiving USAGE + SELECT.
    """

    schema: str
    view: str
    ro_role: str


# --- identifier + value validation (B4, implemented — the safety contract) -----


def validate_version(version: int) -> int:
    """Return *version* if it is a positive int, else raise ValueError."""
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"version must be a positive int, got {version!r}")
    return version


def validate_text_search_config(config: str) -> str:
    """Return *config* if allowlisted, else raise ValueError (B4)."""
    if config not in ALLOWED_TEXT_SEARCH_CONFIGS:
        raise ValueError(
            f"text_search_config {config!r} not in allowlist "
            f"{sorted(ALLOWED_TEXT_SEARCH_CONFIGS)}"
        )
    return config


_HASH_WIDTH = 16  # 64-bit content-hash suffix


def validate_logical_name(logical_name: str) -> str:
    """Return *logical_name* if it is a non-empty printable-ASCII string.

    Restricting to ASCII keeps byte-length == char-length for the stable
    reader-facing view identifier and avoids surprising multibyte truncation
    (B4). Raises ValueError otherwise.
    """
    if not logical_name or not logical_name.isascii() or not logical_name.isprintable():
        raise ValueError(
            f"logical_name must be non-empty printable ASCII, got {logical_name!r}"
        )
    return logical_name


def _fit_identifier(base: str, reserved: int) -> str:
    """Fit *base* into MAX_IDENTIFIER_BYTES leaving *reserved* bytes for a suffix.

    Byte-safe: the budget and truncation are measured in UTF-8 bytes (not
    characters) and a partial trailing codepoint is dropped, so a multibyte name
    can never exceed the 63-byte limit. Over-budget names keep a 16-hex (64-bit)
    content hash — post-truncation collisions are cryptographically improbable,
    not impossible (this is collision-resistance, not a uniqueness guarantee).
    """
    budget = MAX_IDENTIFIER_BYTES - reserved
    min_suffix = _HASH_WIDTH + 1  # hash chars + '_' separator
    if budget < min_suffix + 1:
        raise ValueError(f"reserved={reserved} leaves no room for an identifier")
    raw = base.encode()
    if len(raw) <= budget:
        return base
    digest = hashlib.sha256(raw).hexdigest()[:_HASH_WIDTH]
    keep = budget - min_suffix
    truncated = raw[:keep].decode("utf-8", errors="ignore")
    return f"{truncated}_{digest}"


def view_name(logical_name: str) -> str:
    """Return the stable reader-facing VIEW identifier for *logical_name* (B4).

    The view carries no per-version suffix, so it uses the full 63-byte budget.
    Readers MUST resolve the view through this function rather than assume it
    equals ``logical_name`` (a long name is hash-fitted, same as physical names).
    """
    validate_logical_name(logical_name)
    return _fit_identifier(logical_name, reserved=0)


def physical_table_name(logical_name: str, version: int) -> str:
    """Return the physical table name for a corpus version: ``<logical>__v<N>``.

    Length-safe: the logical portion is hashed if the full name would exceed the
    63-byte identifier limit. The same fitted base backs all per-version indexes.
    """
    validate_logical_name(logical_name)
    validate_version(version)
    suffix = f"__v{version}"
    # Reserve room for both the version suffix and the longest index suffix.
    reserved = len(suffix) + len(_LONGEST_INDEX_SUFFIX)
    return f"{_fit_identifier(logical_name, reserved)}{suffix}"


_INDEX_SUFFIXES: dict[str, str] = {
    "ann": "_ann",
    "bm25": "_bm25",
    "meta_gin": "_meta_gin",
    "scan": "_scan",
    "tsv_gin": "_tsv_gin",
}
_LONGEST_INDEX_SUFFIX = max(_INDEX_SUFFIXES.values(), key=len)


def index_names(logical_name: str, version: int) -> dict[str, str]:
    """Return the per-version index identifier set for a corpus version.

    Keys: ``ann`` (vector index, see PROVEN_ANN_DDL — ``lakebase_ann``),
    ``bm25`` (lexical), ``meta_gin`` (``jsonb_path_ops`` for ``@>`` containment,
    B3), ``scan`` (btree on ``(source_file, chunk_index, id)``, B6), ``tsv_gin``
    (native FTS fallback). All are length-safe.
    """
    base = physical_table_name(logical_name, version)
    return {key: f"{base}{suffix}" for key, suffix in _INDEX_SUFFIXES.items()}


# --- DDL assembly seams (Slice 2 stubs; return psycopg.sql composables) --------
# Reference skeletons documenting the FROZEN physical shape. These are NOT
# executed by ``str`` interpolation — Slice 2 composes them with
# ``sql.SQL(...).format(sql.Identifier(...), ...)`` and bound ``sql.Literal`` for
# the regconfig, using only validated inputs from the helpers above.

CREATE_TABLE_SKELETON = """
CREATE TABLE {table} (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding {vector_type},
    source_file text NOT NULL,
    chunk_index integer NOT NULL,
    content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector({tsconfig}::regconfig, content)) STORED
)
""".strip()

# B3: metadata predicates are emitted as ``@>`` containment (see filter_mapper),
# which ``jsonb_path_ops`` GIN accelerates. ``?|``/``?&`` are intentionally NOT
# used — a whole-doc GIN cannot serve them.
CREATE_META_GIN_INDEX_SKELETON = (
    "CREATE INDEX {index} ON {table} USING gin (metadata jsonb_path_ops)"
)

# B6: deterministic scan order is backed by a typed btree, not JSONB extraction.
CREATE_SCAN_INDEX_SKELETON = (
    "CREATE INDEX {index} ON {table} (source_file, chunk_index, id)"
)

CREATE_TSV_GIN_INDEX_SKELETON = (
    "CREATE INDEX {index} ON {table} USING gin (content_tsv)"
)

CREATE_LEDGER_SKELETON = """
CREATE TABLE IF NOT EXISTS neon_corpus_versions (
    logical_name text NOT NULL,
    version integer NOT NULL,
    state text NOT NULL
        CHECK (state IN ('building', 'ready', 'activated', 'retired')),
    is_current boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    ready_at timestamptz,
    activated_at timestamptz,
    retired_at timestamptz,
    PRIMARY KEY (logical_name, version)
)
""".strip()

# Current-pointer invariant (B5): at most one published version per logical name.
CREATE_CURRENT_POINTER_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS neon_corpus_current "
    "ON neon_corpus_versions (logical_name) WHERE is_current"
)

# Owner-rights view (security_invoker = false) so the RO role never touches
# physical tables. Explicit column list, never SELECT *.
ACTIVATE_VIEW_SKELETON = (
    "CREATE OR REPLACE VIEW {view} WITH (security_invoker = false) AS "
    "SELECT {columns} FROM {table}"
)


def create_table_ddl(spec: NeonTableSpec) -> sql.Composed:
    """Return the ``CREATE TABLE`` composable for a physical corpus version.

    Composed from ``sql.Identifier`` (table) + a bound ``sql.Literal`` regconfig
    drawn from ALLOWED_TEXT_SEARCH_CONFIGS, using the PROVISIONAL vector type
    resolved in Slice 3. Design-lock stub: assembly lands in Slice 2.
    """
    raise NotImplementedError("schema DDL assembly is built in Slice 2")


def activate_version_sql(
    spec: NeonTableSpec, grant: ReadGrantSpec
) -> list[sql.Composed]:
    """Return the single-transaction composables that publish a version (B5).

    Ordered, all under one ``pg_advisory_xact_lock`` on the logical name:
    acquire the lock; clear the prior ``is_current`` row; set this version
    ``state='activated'``, ``is_current=true``; ``CREATE OR REPLACE VIEW``
    (owner-rights, explicit columns via ``view_name``/``VIEW_COLUMNS``); and issue
    the RO grants from *grant* so a first-create view is readable atomically with
    publication. The current-pointer unique index enforces the single-current
    invariant. Design-lock stub: assembly lands in Slice 2.
    """
    raise NotImplementedError("version activation is built in Slice 2")


def rollback_version_sql(logical_name: str, target_version: int) -> list[sql.Composed]:
    """Return the composables that re-point the active version to a prior one.

    Non-destructive: prior physical tables are retained, so rollback re-points
    the ledger + view under the advisory lock. Design-lock stub: built in Slice 2.
    """
    validate_version(target_version)
    raise NotImplementedError("version rollback is built in Slice 2")


def read_grant_sql(grant: ReadGrantSpec) -> list[sql.Composed]:
    """Return the ``GRANT USAGE``/``GRANT SELECT`` composables for the RO role.

    Issued on first view creation (see ReadGrantSpec). Design-lock stub: built
    in Slice 2.
    """
    raise NotImplementedError("RO grant assembly is built in Slice 2")
