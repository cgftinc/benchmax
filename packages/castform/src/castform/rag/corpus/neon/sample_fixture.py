"""Tiny deterministic smoke fixture for the Neon Lakebase corpus (Slice 3).

A handful of rows with KNOWN relevance ordering for one vector query, one BM25
query, and one metadata-filtered query — enough to prove the frozen ANN / BM25 /
GIN DDL end to end under the read-only role. This is NOT the full corpus (that
needs the ChunkSource + real embeddings from Slices 5/7); embeddings here are
deterministic synthetic vectors laid out on two basis axes so nearest-neighbour
order is known without an embedding model.

Vector layout: every document's embedding is zero except two leading components
``(w0, w1)``. The vector query is the unit ``w0`` axis, so cosine similarity to the
query is ``w0 / sqrt(w0^2 + w1^2)`` — higher ``w0`` share ⇒ nearer. The weights are
chosen so the top-3 vector order is exactly ``[smoke-6, smoke-2, smoke-5]``.

BM25 layout: for the query ``"quick brown fox"``, ``smoke-1`` repeats the query
terms most, so it is the unambiguous best hit; ``<@>`` scores are negative and
sort ascending (best-first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from castform.rag.corpus.neon.schema import EMBEDDING_DIM, NeonTableSpec, ReadGrantSpec, view_name

if TYPE_CHECKING:
    from castform.rag.corpus.neon.client import NeonClient

SMOKE_LOGICAL_NAME = "benchmax_smoke"
SMOKE_VERSION = 1
CORPUS_SCHEMA = "benchmax_corpus"
RO_ROLE = "benchmax_ro"

BM25_QUERY = "quick brown fox"
BM25_EXPECTED_TOP_ID = "smoke-1"

# Vector query = the unit w0 axis; expected nearest-first ids (top-3).
VECTOR_EXPECTED_ORDER = ("smoke-6", "smoke-2", "smoke-5")

FILTER_LANG = "fr"
FILTER_EXPECTED_IDS = frozenset({"smoke-4", "smoke-7"})


@dataclass(frozen=True)
class _Doc:
    id: str
    content: str
    lang: str
    w0: float  # weight on the query (v0) axis
    w1: float  # weight on the orthogonal (v1) axis


# Ordering intent, per the query definitions above:
#   BM25 "quick brown fox": smoke-1 (terms x2+) >> smoke-2 ~ smoke-5 > rest.
#   Vector (w0 axis):       smoke-6 (1.0,0) < smoke-2 (0.8,0.6) < smoke-5 (0.6,0.8) < rest (0,1).
#   Filter lang='fr':       only smoke-4, smoke-7.
SMOKE_DOCS: tuple[_Doc, ...] = (
    _Doc("smoke-1", "quick brown fox quick brown fox quick fox", "en", 0.0, 1.0),
    _Doc("smoke-2", "the quick brown fox jumps over the lazy dog", "en", 0.8, 0.6),
    _Doc("smoke-3", "a quiet meadow with wildflowers and honeybees", "en", 0.0, 1.0),
    _Doc("smoke-4", "le renard brun rapide saute le chien", "fr", 0.0, 1.0),
    _Doc("smoke-5", "brown foxes roam beside the quick river", "en", 0.6, 0.8),
    _Doc("smoke-6", "postgres lakebase bm25 vector ranking search", "en", 1.0, 0.0),
    _Doc("smoke-7", "montagnes rivieres et forets ensoleillees", "fr", 0.0, 1.0),
    _Doc("smoke-8", "the lazy dog sleeps all afternoon", "en", 0.0, 1.0),
)


def _embedding(doc: _Doc) -> list[float]:
    """Return the doc's 3072-dim synthetic embedding: zeros except the two axes."""
    vec = [0.0] * EMBEDDING_DIM
    vec[0], vec[1] = doc.w0, doc.w1
    return vec


def query_vector() -> list[float]:
    """Return the vector query — the unit w0 axis (nearest is the highest-w0 doc)."""
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = 1.0
    return vec


def smoke_rows() -> list[tuple[Any, ...]]:
    """Return the fixture rows in the client's INSERT_COLUMNS order.

    ``(id, content, metadata, embedding, source_file, chunk_index)`` — matching
    :data:`castform.rag.corpus.neon.client.INSERT_COLUMNS`. The metadata dict is
    wrapped in ``psycopg.types.json.Jsonb`` so it adapts to the ``jsonb`` column.
    """
    from psycopg.types.json import Jsonb

    return [
        (
            doc.id,
            doc.content,
            Jsonb({"lang": doc.lang}),
            _embedding(doc),
            f"{doc.id}.txt",
            0,
        )
        for doc in SMOKE_DOCS
    ]


def smoke_spec() -> NeonTableSpec:
    """Return the fixture's table spec (logical name + version)."""
    return NeonTableSpec(logical_name=SMOKE_LOGICAL_NAME, version=SMOKE_VERSION)


def smoke_grant() -> ReadGrantSpec:
    """Return the RO grant published with the fixture's reader view."""
    return ReadGrantSpec(
        schema=CORPUS_SCHEMA, view=view_name(SMOKE_LOGICAL_NAME), ro_role=RO_ROLE
    )


def load_smoke_corpus(writer: NeonClient) -> None:
    """Build + activate the smoke corpus with *writer* (idempotent per version).

    Drops any prior physical table for this version first so a re-run rebuilds
    cleanly (the fixture is a fixed single version, not the production versioned
    lifecycle). Builds the table + ANN/BM25/GIN indexes, VACUUMs (BM25 corpus
    stats need it), then activates the owner-rights reader view and issues the RO
    grants — leaving the corpus queryable by the read-only role.
    """
    from psycopg import sql

    from castform.rag.corpus.neon.schema import physical_table_name

    spec = smoke_spec()
    table = physical_table_name(spec.logical_name, spec.version)
    # Idempotent reset: ensure the ledger exists, then clear this version's table +
    # ledger row so build_version starts fresh (allocate_version inserts a fresh
    # 'building' row that would otherwise duplicate-key on a re-run).
    for stmt in writer.create_ledger_sql():
        writer.execute(stmt)
    writer.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
    writer.execute(
        sql.SQL(
            "DELETE FROM neon_corpus_versions "
            "WHERE logical_name = %(logical)s AND version = %(version)s"
        ),
        {"logical": spec.logical_name, "version": spec.version},
    )
    writer.build_version(spec, smoke_rows())
    writer.activate(spec, smoke_grant())
