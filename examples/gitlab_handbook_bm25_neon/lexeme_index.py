"""Authoritative Postgres-lexeme postings for the corpus, pulled at freeze time.

The curated FILTER rows need a query token whose real BM25 match set is small and
known, so that every cross-section decoy provably surfaces in an unfiltered top-k
and the gold is uniquely correct under the section predicate. Naive substring
counting misses this because Postgres BM25 STEMS and drops stopwords (``accessibly``
stems to ``access`` and matches 4605 chunks). This module reads the corpus's own
``content_tsv`` — the exact tokenizer output — so a lexeme's document frequency ==
``plainto_tsquery(lexeme)`` match count, giving the curated selection a teeth
guarantee that holds against the live gate.

Pulled from Neon (RO) once per freeze; not committed (the full posting list is
~160MB). No re-embed and no ranked retrieval are involved — this is the corpus's
static token structure, so it does not compromise blind authoring.
"""

from __future__ import annotations

import re
from collections import defaultdict

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.schema import NeonTableSpec, physical_table_name

# Lexemes in a ``content_tsv::text`` render: ``'lexeme':pos,pos 'next':pos``. A
# literal quote inside a lexeme is doubled per Postgres text output.
_LEX_RE = re.compile(r"'((?:[^']|'')+)':")


def build_lexeme_postings(client: NeonClient, spec: NeonTableSpec) -> dict[str, list[str]]:
    """Map each Postgres lexeme to the sorted chunk hashes whose ``content_tsv``
    contains it (the authoritative BM25 posting list).

    Args:
        client: A read client over the corpus (RO).
        spec: The current corpus version (physical table resolves from it).
    """
    from psycopg import sql

    table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
    rows = client.execute(
        sql.SQL("SELECT id, content_tsv::text FROM {}").format(table), {}
    )
    postings: dict[str, list[str]] = defaultdict(list)
    for cid, tsv in rows:
        for lex in set(_LEX_RE.findall(tsv or "")):
            postings[lex.replace("''", "'")].append(cid)
    return {k: sorted(v) for k, v in postings.items()}
