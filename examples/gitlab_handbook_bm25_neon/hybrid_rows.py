"""HYBRID golden rows — a deferred SMOKE capability (no teeth threshold).

Status: fusion-necessity gate DEFERRED to the Path X follow-up.

A genuine hybrid gate would require FUSION-NECESSITY rows: a query where neither
lexical-only NOR vector-only ranks the gold in the top-k, but RRF fusion (the
Slice-4 query layer) does. We attempted to author such rows blind against this
corpus and could not build a single one (28 blind candidates across 8 sections,
0 met the both-legs-miss precondition). The reason is structural, not effort:

* the corpus is LEXICAL-strong — a distinctive term ranks its gold #1 under BM25;
* it is ALSO VECTOR-strong — ``text-embedding-3-large`` @ 3072-dim ranks the gold
  at position 1-4 for any FAITHFUL paraphrase (vector-only hit@5 ~0.96), and blind
  authoring REQUIRES a faithful paraphrase, so the vector leg alone almost always
  already succeeds;
* when lexical does miss, the vector leg rescues the gold, so fusion adds nothing
  over the stronger single leg (and in the one lexical-absent case, RRF actively
  HURT by diluting with wrong-doc lexical candidates).

Isolating fusion on this corpus would require weakening a retrieval leg (a
retrieval-CONFIG change — a weaker/lower-dim embedder or a starving top-k
truncation), which is out of scope for dataset authoring. The RRF mechanism itself
is real and unit-tested in Slice 4; this eval simply cannot isolate it here.

So these rows are a SMOKE check only: they exercise the ``hybrid`` mode end to end
(text + vector legs + section filter) and confirm the fused query returns the gold.
They carry NO "beats both single legs" threshold — the measured hybrid threshold is
a loose smoke floor, and the live gate asserts fusion runs and finds gold, not that
fusion is necessary. Authored blind from chunk content, same as the FILTER rows.
"""

from __future__ import annotations

import re
from collections import defaultdict

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

from curated_rows import _by_hash, _rare_lexemes, _section_eq_candidate

_WORD_RE = re.compile(r"[a-z]{5,}")
_ALPHA_WORD_RE = re.compile(r"^[a-z][a-z'-]+$")
_STOP = frozenset(
    {"handbook", "gitlab", "which", "these", "their", "there", "where", "about",
     "would", "should", "could", "other", "under", "using", "based"}
)


def _looks_prose(content: str) -> bool:
    """Cheap guard that a chunk is readable prose, not a diagram / code / config
    block (mermaid ``accdescr``/``fontfamily`` etc.), so a smoke query reads cleanly.
    """
    toks = content.lower().split()
    if len(toks) < 40:
        return False
    alpha = sum(1 for t in toks if _ALPHA_WORD_RE.match(t))
    return alpha / len(toks) >= 0.7


def _smoke_query(content: str, lex: str, k: int = 6) -> str:
    """A short natural phrase seeded on the rare lexeme plus the gold chunk's first
    few contentful words, so BOTH retrieval legs have real signal (a bare token is a
    lexical-only probe). Deterministic in document order."""
    terms = [lex]
    for w in _WORD_RE.findall(content.lower()):
        if w != lex and w not in _STOP and w not in terms:
            terms.append(w)
        if len(terms) >= k:
            break
    return " ".join(terms)


def build_hybrid_rows(
    collection: ChunkCollection,
    lexeme_postings: dict[str, list[str]],
    *,
    n: int = 6,
    exclude_lexemes: frozenset[str] = frozenset(),
) -> list[NeonEvalRecord]:
    """Build the deferred hybrid SMOKE rows (deterministic, blind).

    Reuses the ``filter_section_eq`` construction (a rare in-section gold + a section
    filter) but runs in ``hybrid`` mode with a natural multi-term query, spread
    across sections and disjoint from each other by section.

    Args:
        collection: The re-chunked handbook collection (in-memory, with hashes).
        lexeme_postings: Authoritative lexeme -> chunk-hash postings.
        n: Number of hybrid smoke rows.
        exclude_lexemes: Lexemes already spent on other capabilities (e.g. the FILTER
            rows) so hybrid rows address distinct chunks.
    """
    by_hash = _by_hash(collection)
    rows: list[NeonEvalRecord] = []
    seen_sections: set[str] = set()
    for lex in _rare_lexemes(lexeme_postings):
        if lex in exclude_lexemes:
            continue
        cand = _section_eq_candidate(lex, lexeme_postings[lex], by_hash)
        if not cand:
            continue
        gold, decoys, section = cand
        if section in seen_sections or not _looks_prose(by_hash[gold].content):
            continue
        seen_sections.add(section)
        rows.append(
            NeonEvalRecord(
                capability="hybrid_smoke",
                search_mode="hybrid",
                query=_smoke_query(by_hash[gold].content, lex),
                filter_dsl={"field": "handbook_section", "op": "eq", "value": section},
                gold_chunk_hashes=[gold],
                decoy_chunk_hashes=decoys,
            )
        )
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"insufficient hybrid smoke candidates: {len(rows)}/{n}")
    return rows
