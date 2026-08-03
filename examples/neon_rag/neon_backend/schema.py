"""Physical schema and versioned-replace lifecycle for the Neon example.

This module defines identifier helpers, allowlists, value validation, version
dataclasses, and shared ledger DDL constants. The versioned-replace lifecycle SQL (per-version
CREATE TABLE, index builds, activate/rollback, RO grants) is composed and executed
in :mod:`neon_backend.client` (the lifecycle-SQL owner), which reuses
the helpers and constants here — never interpolated strings.

Versioned-replace lifecycle
---------------------------
A corpus is addressed by a stable *logical name*. Each ingest builds a fresh
*physical* table ``<logical>__v<version>`` with its own indexes; readers query a
stable owner-rights view under the logical name. A per-version *ledger*
(``neon_corpus_versions``) tracks each version's state (building -> ready ->
activated -> retired) with timestamps, so concurrent ingest, enumerate, prune,
and build-vs-ready are all well-defined. Activation flips the ledger's active
row AND re-points the view in one transaction under a per-logical advisory lock;
rollback re-points to any prior ``activated`` version (a ``ready``-only version
was never published and cannot be a rollback target — the impl row-locks
``state = 'activated'``; old physical tables are retained until pruned, so
rollback is O(1) and non-destructive).

Chunk identity is content+metadata derived (``rag/chunkers/models.py``:
``hash = sha256(metadata_str + "\\n" + content)``), so re-ingesting changed
content yields new ids — hence versioned *replace*, not in-place upsert.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

# --- embedding dimension and metric -------------------------------------------

EMBEDDING_DIM = 3072
"""Embedding vector dimensionality (text-embedding-3-large)."""

DISTANCE_METRIC = "cosine"
"""Vector distance metric. Fixed for the managed table; index uses cosine ops."""

DEFAULT_TEXT_SEARCH_CONFIG = "pg_catalog.english"
"""Default Postgres text-search config for the tsvector generated column."""

# Text-search configs are never interpolated raw. Only values from this allowlist are
# accepted, then emitted as a bound ``sql.Literal`` cast to ``regconfig``.
ALLOWED_TEXT_SEARCH_CONFIGS: frozenset[str] = frozenset({"pg_catalog.english", "pg_catalog.simple"})

MAX_IDENTIFIER_BYTES = 63
"""Postgres identifier length limit; longer names are hash-suffixed."""

# Explicit column list for the stable view; never ``SELECT *``. ``id`` is
# the chunk hash; ``source_file``/``chunk_index`` are typed non-null columns that
# provide deterministic scan ordering.
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

# Client-facing read projection. The view exposes all VIEW_COLUMNS so the
# ANN/BM25 expressions can reference ``embedding`` / ``content_tsv``, but the
# candidate/sample/scan SELECTs project only these columns — omitting the ~12KB
# ``embedding`` and the ``content_tsv`` from the OUTPUT cuts serialization/egress
# per returned row for data no search consumer needs (the score is returned
# separately). Ordering/scoring still reference the omitted columns via the view.
READ_COLUMNS: tuple[str, ...] = (
    "id",
    "content",
    "metadata",
    "source_file",
    "chunk_index",
)


# --- ANN access method, vector type, and opclass -------------------------------
# Verified live on Neon Lakebase (PG 18.4, lakebase_vector 1.0.0-dev): the native
# ``lakebase_ann`` access method indexes a full-precision ``vector(3072)`` column
# with ``vector_cosine_ops`` and is used by the planner (EXPLAIN: ``Index Scan
# using ..._ann``) for a cosine ``ORDER BY``. Unlike pgvector's ``hnsw`` — which
# rejects >2000 dims — ``lakebase_ann`` has no dimension cap at 3072, so the
# ``halfvec`` workaround is NOT required for correctness (it also builds and is
# kept documented below as the storage-saving alternative). The query param MUST be
# cast to ``vector`` (a bound Python list binds as ``float8[]`` and the cast-less
# ``<=>`` errors "operator does not exist: vector <=> double precision[]"). The
# type, opclass, distance operator, and query-param cast must change together as one
# coherent unit — changing any one alone breaks index use. See CONTRACT.md §1.
PROVEN_ANN_DDL: dict[str, str] = {
    "access_method": "lakebase_ann",
    "vector_type": "vector(3072)",
    "opclass": "vector_cosine_ops",
    "operator": "<=>",
    "query_param_cast": "vector",
}

# Storage-saving alternative, verified to build and be used by the planner. It
# halves bytes per vector at a small recall cost.
ANN_HALFVEC_ALTERNATIVE: dict[str, str] = {
    "access_method": "lakebase_ann",
    "vector_type": "halfvec(3072)",
    "opclass": "halfvec_cosine_ops",
    "operator": "<=>",
    "query_param_cast": "halfvec",
}


VersionState = Literal["building", "ready", "activated", "retired"]
"""Lifecycle states for one physical corpus version.

``activated`` is *historical* — the version has been published at least once.
Which single version is *currently* published is tracked separately by the
``is_current`` flag (see the current-pointer invariant below), so a rollback flips
``is_current`` between two ``activated`` versions without changing their state.
"""

# Legal state transitions, also enforced by the database check constraint.
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
    """One row of the per-version ledger.

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
    """Version retention and pruning policy.

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
    """Read-only grant the ingest role issues for the search role.

    The stable view is owner-rights (``security_invoker = false``), so the vector
    + filter read paths need only schema ``USAGE`` and ``SELECT`` on the view.
    These grants must be (re)issued on FIRST view creation (``CREATE OR REPLACE
    VIEW`` preserves an existing ACL but the first create has none).

    BM25 exception: ``to_bm25query`` runs with the read-only invoker's
    rights and reads the bm25 index's base-table stats, so the RO role ALSO needs
    ``SELECT`` on the physical version tables — granted narrowly by the writer's
    ``ALTER DEFAULT PRIVILEGES`` + ``GRANT SELECT ON ALL TABLES`` (see
    ``provision.py``), never any write/DDL privilege. So "RO never touches physical
    tables" holds for the view reads; bm25 is the one read that needs the
    base-table SELECT. See CONTRACT.md §1.

    Args:
        schema: Schema holding the corpus objects.
        view: Stable logical view name.
        ro_role: The read-only role receiving USAGE + SELECT.
    """

    schema: str
    view: str
    ro_role: str


# --- identifier and value validation ------------------------------------------


def validate_version(version: int) -> int:
    """Return *version* if it is a positive int, else raise ValueError."""
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"version must be a positive int, got {version!r}")
    return version


def validate_text_search_config(config: str) -> str:
    """Return *config* if allowlisted, else raise ValueError."""
    if config not in ALLOWED_TEXT_SEARCH_CONFIGS:
        raise ValueError(
            f"text_search_config {config!r} not in allowlist {sorted(ALLOWED_TEXT_SEARCH_CONFIGS)}"
        )
    return config


_HASH_WIDTH = 16  # 64-bit content-hash suffix


def validate_logical_name(logical_name: str) -> str:
    """Return *logical_name* if it is a non-empty printable-ASCII string.

    Restricting to ASCII keeps byte-length == char-length for the stable
    reader-facing view identifier and avoids surprising multibyte truncation
    Raises ValueError otherwise.
    """
    if not logical_name or not logical_name.isascii() or not logical_name.isprintable():
        raise ValueError(f"logical_name must be non-empty printable ASCII, got {logical_name!r}")
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
    """Return the stable reader-facing view identifier for *logical_name*.

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
    ``bm25`` (lexical), ``meta_gin`` (``jsonb_path_ops`` for ``@>`` containment),
    ``scan`` (btree on ``(source_file, chunk_index, id)``), ``tsv_gin``
    (native FTS fallback). All are length-safe.
    """
    base = physical_table_name(logical_name, version)
    return {key: f"{base}{suffix}" for key, suffix in _INDEX_SUFFIXES.items()}


# --- shared ledger DDL (consumed by the client.py lifecycle executor) ---------
# The per-version ledger + current-pointer index are shared across every logical
# corpus, so their DDL is defined here as constants; NeonClient wraps them in
# ``sql.SQL(...)`` and executes them. The per-version table/index/view/activation
# SQL is composed from the validated helpers above inside client.py (the
# lifecycle-SQL owner) — see CONTRACT.md §1.

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

# Current-pointer invariant: at most one published version per logical name.
CREATE_CURRENT_POINTER_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS neon_corpus_current "
    "ON neon_corpus_versions (logical_name) WHERE is_current"
)
