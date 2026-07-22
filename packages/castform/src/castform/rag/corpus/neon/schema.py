"""Physical table schema and versioned-replace model for the Neon corpus.

Contract-freeze artifact (Slice A). The DDL string templates and identifier
helpers here are the *frozen* physical model; the executor that runs them is
built in Slice 2. Nothing in this module opens a connection or emits real SQL
against a database.

Versioned-replace model
-----------------------
A corpus is addressed by a stable *logical name*. Each ingest builds a fresh
*physical* table ``<logical>__v<version>`` with its own indexes, so an in-flight
rebuild never mutates the table readers are querying. An *active-version pointer*
(a registry row plus a ``CREATE OR REPLACE VIEW`` under the logical name) is
flipped in a single transaction to publish a new version, and flipped back to
roll one back. Old physical tables are retained until explicitly pruned, which is
what makes rollback O(1) and non-destructive.

Chunk identity is content+metadata derived (see ``rag/chunkers/models.py``:
``hash = sha256(metadata_str + "\\n" + content)``), so re-ingesting changed
content yields new ids — the reason a whole-table versioned replace, not an
in-place upsert, is the correct shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Contract #7: embedding dim / metric (frozen) ------------------------------

EMBEDDING_DIM = 3072
"""Embedding vector dimensionality (text-embedding-3-large)."""

DISTANCE_METRIC = "cosine"
"""Vector distance metric. Fixed for the managed table; index uses cosine ops."""

DEFAULT_TEXT_SEARCH_CONFIG = "pg_catalog.english"
"""Default Postgres text-search config. Configurable per corpus, never hardcoded
downstream — the tsvector generated column bakes this per physical version."""


@dataclass(frozen=True)
class NeonTableSpec:
    """Frozen description of one physical corpus-version table.

    Args:
        logical_name: Stable logical corpus name (readers address this).
        version: Monotonic physical version number for this build.
        embedding_dim: Vector column dimensionality. Defaults to EMBEDDING_DIM.
        distance_metric: Vector distance metric. Defaults to DISTANCE_METRIC.
        text_search_config: Postgres ``regconfig`` for the tsvector column and
            the lexical tokenizer. Baked into the generated column of this
            version (a config change requires a new version, not an ALTER).
    """

    logical_name: str
    version: int
    embedding_dim: int = EMBEDDING_DIM
    distance_metric: str = DISTANCE_METRIC
    text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG
    _reserved: dict[str, object] = field(default_factory=dict)


def physical_table_name(logical_name: str, version: int) -> str:
    """Return the physical table name for a corpus version: ``<logical>__v<N>``."""
    return f"{logical_name}__v{version}"


def index_names(logical_name: str, version: int) -> dict[str, str]:
    """Return the per-version index identifier set for a corpus version.

    Keys: ``hnsw`` (ANN over the embedding), ``bm25`` (lexical), ``meta_gin``
    (JSONB metadata containment), ``tsv_gin`` (native FTS fallback).
    """
    base = physical_table_name(logical_name, version)
    return {
        "hnsw": f"{base}_hnsw",
        "bm25": f"{base}_bm25",
        "meta_gin": f"{base}_meta_gin",
        "tsv_gin": f"{base}_tsv_gin",
    }


# --- Contract #1: DDL constant templates (frozen shapes, not executed) ---------

# The embedding is stored as ``vector(3072)`` but pgvector HNSW/IVFFlat index the
# ``vector`` type only up to 2000 dims, so the ANN index is built on a
# ``halfvec(3072)`` cast expression (HNSW supports halfvec up to 4000 dims).
CREATE_TABLE_TEMPLATE = """
CREATE TABLE {table} (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    embedding vector({dim}),
    content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector({tsconfig}::regconfig, content)) STORED
)
""".strip()

CREATE_HNSW_INDEX_TEMPLATE = (
    "CREATE INDEX {index} ON {table} "
    "USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)"
)

# Lexical ranking uses the bm25 ``<@>`` operator over a bm25 index tokenized with
# the same configurable text-search config; the tsvector GIN index below is the
# portable native-FTS fallback (ts_rank_cd).
CREATE_BM25_INDEX_TEMPLATE = (
    "CREATE INDEX {index} ON {table} USING bm25 (id, content) "
    "WITH (key_field='id', text_config={tsconfig})"
)

CREATE_META_GIN_INDEX_TEMPLATE = (
    "CREATE INDEX {index} ON {table} USING gin (metadata jsonb_path_ops)"
)

CREATE_TSV_GIN_INDEX_TEMPLATE = (
    "CREATE INDEX {index} ON {table} USING gin (content_tsv)"
)

# Active-version pointer: a registry table plus a view under the logical name.
CREATE_REGISTRY_TABLE = """
CREATE TABLE IF NOT EXISTS neon_corpus_versions (
    logical_name text PRIMARY KEY,
    active_version int NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
)
""".strip()

ACTIVATE_VIEW_TEMPLATE = "CREATE OR REPLACE VIEW {logical} AS SELECT * FROM {table}"

UPSERT_ACTIVE_VERSION_TEMPLATE = (
    "INSERT INTO neon_corpus_versions (logical_name, active_version) "
    "VALUES (%(logical)s, %(version)s) "
    "ON CONFLICT (logical_name) DO UPDATE SET "
    "active_version = EXCLUDED.active_version, updated_at = now()"
)


def create_table_ddl(spec: NeonTableSpec) -> str:
    """Return the ``CREATE TABLE`` DDL for a physical corpus version.

    Design-lock stub: the executor and DDL assembly land in Slice 2.
    """
    raise NotImplementedError("schema DDL assembly is built in Slice 2")


def activate_version_sql(spec: NeonTableSpec) -> list[str]:
    """Return the ordered, single-transaction statements that publish a version.

    Atomic activate = upsert the registry pointer + ``CREATE OR REPLACE VIEW``
    under the logical name, run in one transaction. Design-lock stub.
    """
    raise NotImplementedError("version activation is built in Slice 2")


def rollback_version_sql(logical_name: str, target_version: int) -> list[str]:
    """Return the statements that roll the active pointer back to a prior version.

    Non-destructive: prior physical tables are retained, so rollback re-points
    the registry + view. Design-lock stub.
    """
    raise NotImplementedError("version rollback is built in Slice 2")
