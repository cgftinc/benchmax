"""Hand-curated FILTER golden rows (Path Y), authored BLIND with real teeth.

qa-gen produces the lexical / vector rows; the FILTER capability is authored here
from CHUNK CONTENT (never from a Neon retrieval result), so gold ids stay
independent of what the provider returns (F12/B8 non-circularity). The HYBRID rows
live in :mod:`hybrid_rows`.

Teeth guarantee (why these rows are a non-vacuous gate)
------------------------------------------------------
Selection keys on the corpus's own Postgres lexeme postings
(:mod:`lexeme_index`) — the exact BM25 tokenizer output, so a lexeme's document
frequency IS its ``plainto_tsquery`` match count. A query token is admitted only
when its lexeme matches a small, fully-known set of chunks (``2..MAX_MATCHES``).
Because that whole set fits in an unfiltered top-k, EVERY decoy provably surfaces
unfiltered and then vanishes once the predicate excludes its section — the live
gate checks exactly this. Naive substring counting cannot give this guarantee
(``accessibly`` stems to ``access``, matching thousands of chunks), which is why
the earlier bag-of-words selection left most filter rows toothless.

Two constructions:

* ``filter_section_eq`` — the lexeme has EXACTLY ONE chunk in the target section
  (the gold) and its other chunks live in DIFFERENT sections (the decoys); the
  ``handbook_section`` equality alone must exclude them.
* ``filter_section_depth`` — the lexeme appears MULTIPLE times within ONE section
  at DIFFERENT ``path_depth`` values; the gold is its unique chunk at one depth and
  the decoys are same-section chunks at OTHER depths. The section clause alone
  cannot remove them (same section), so the ``path_depth`` predicate isolates depth
  — a genuine test of the ``and`` / numeric operators, not a no-op AND.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from castform.rag.chunkers.models import ChunkCollection
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

# A query token reads as handbook vocabulary and its lexeme is globally rare: a real
# lowercase word, 6-20 chars, whose Postgres lexeme matches 2..MAX_MATCHES chunks so
# the whole match set fits in an unfiltered top-k (the teeth guarantee).
_TOKEN_RE = re.compile(r"[a-z]{6,20}")
_MIN_MATCHES = 2
_MAX_MATCHES = 5

_STOPWORDS = frozenset(
    {
        "handbook",
        "gitlab",
        "should",
        "process",
        "section",
        "content",
        "please",
        "https",
        "about",
        "which",
        "these",
        "their",
        "there",
        "where",
        "example",
        "following",
    }
)


@dataclass(frozen=True)
class _Chunk:
    hash: str
    section: str
    path_depth: int
    content: str


def _by_hash(collection: ChunkCollection) -> dict[str, _Chunk]:
    out: dict[str, _Chunk] = {}
    for c in collection.chunks:
        md = dict(c.metadata)
        out[c.hash] = _Chunk(
            hash=c.hash,
            section=str(md.get("handbook_section", "")),
            path_depth=int(md.get("path_depth", 0) or 0),
            content=c.content,
        )
    return out


def _rare_lexemes(lexeme_postings: dict[str, list[str]]) -> list[str]:
    """Clean, globally rare lexemes in deterministic (sorted) order."""
    out = []
    for lex in sorted(lexeme_postings):
        if lex in _STOPWORDS or not _TOKEN_RE.fullmatch(lex):
            continue
        if _MIN_MATCHES <= len(lexeme_postings[lex]) <= _MAX_MATCHES:
            out.append(lex)
    return out


def _contains_word(content: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", content.lower()) is not None


def _section_eq_candidate(
    lex: str, hashes: list[str], by_hash: dict[str, _Chunk]
) -> tuple[str, list[str], str] | None:
    """(gold, decoys, section) where ``lex`` is unique in the gold's section and its
    other chunks are in different sections. The query lexeme must appear literally in
    the gold so the token reads as text and the gold is guaranteed to match."""
    chunks = [by_hash[h] for h in hashes if h in by_hash]
    by_section: dict[str, list[_Chunk]] = defaultdict(list)
    for c in chunks:
        by_section[c.section].append(c)
    for section in sorted(s for s, cs in by_section.items() if len(cs) == 1):
        gold = by_section[section][0]
        if not _contains_word(gold.content, lex):
            continue
        decoys = [c.hash for c in chunks if c.section != section]
        if decoys:
            return gold.hash, decoys, section
    return None


def _section_depth_candidate(
    lex: str, hashes: list[str], by_hash: dict[str, _Chunk]
) -> tuple[str, list[str], str, int] | None:
    """(gold, decoys, section, depth) where ``lex`` recurs within ONE section across
    depths; gold is its unique chunk at ``depth`` and decoys are same-section chunks
    at other depths (so the section clause alone cannot remove them)."""
    chunks = [by_hash[h] for h in hashes if h in by_hash]
    by_section: dict[str, list[_Chunk]] = defaultdict(list)
    for c in chunks:
        by_section[c.section].append(c)
    for section in sorted(by_section):
        cs = by_section[section]
        by_depth: dict[int, list[_Chunk]] = defaultdict(list)
        for c in cs:
            by_depth[c.path_depth].append(c)
        if len(by_depth) < 2:
            continue
        for depth in sorted(d for d, dc in by_depth.items() if len(dc) == 1):
            gold = by_depth[depth][0]
            if not _contains_word(gold.content, lex):
                continue
            decoys = [c.hash for c in cs if c.path_depth != depth]
            if decoys:
                return gold.hash, decoys, section, depth
    return None


def _section_filter(section: str) -> dict:
    return {"field": "handbook_section", "op": "eq", "value": section}


def _section_and_depth_filter(section: str, depth: int) -> dict:
    return {
        "and": [
            {"field": "handbook_section", "op": "eq", "value": section},
            {"field": "path_depth", "op": "eq", "value": depth},
        ]
    }


def _diverse(picked: list, key, cands: list, n: int) -> None:
    """Append up to ``n`` candidates into ``picked``, spreading across ``key`` groups
    (stable, deterministic)."""
    by_group: dict[str, list] = defaultdict(list)
    for c in cands:
        by_group[key(c)].append(c)
    groups = sorted(by_group)
    idx = 0
    while len(picked) < n and any(by_group[g] for g in groups):
        g = groups[idx % len(groups)]
        if by_group[g]:
            picked.append(by_group[g].pop(0))
        idx += 1


def build_filter_rows(
    collection: ChunkCollection,
    lexeme_postings: dict[str, list[str]],
    *,
    n_section_eq: int = 6,
    n_section_depth: int = 4,
) -> list[NeonEvalRecord]:
    """Build the FILTER golden rows (deterministic, teeth-guaranteed, blind).

    Args:
        collection: The re-chunked handbook collection (in-memory, with hashes).
        lexeme_postings: Authoritative lexeme -> chunk-hash postings from
            :func:`lexeme_index.build_lexeme_postings`.
        n_section_eq: Number of ``filter_section_eq`` rows.
        n_section_depth: Number of ``filter_section_depth`` rows (depth-isolation).
    """
    by_hash = _by_hash(collection)
    lexemes = _rare_lexemes(lexeme_postings)

    eq_cands: list[tuple[str, str, list[str], str]] = []
    depth_cands: list[tuple[str, str, list[str], str, int]] = []
    for lex in lexemes:
        hashes = lexeme_postings[lex]
        eq = _section_eq_candidate(lex, hashes, by_hash)
        if eq:
            eq_cands.append((lex, *eq))
        dep = _section_depth_candidate(lex, hashes, by_hash)
        if dep:
            depth_cands.append((lex, *dep))

    # Depth rows pick first, then eq rows avoid reusing those lexemes so each query
    # token addresses exactly one row.
    picked_depth: list = []
    _diverse(picked_depth, lambda c: c[3], depth_cands, n_section_depth)
    used = {c[0] for c in picked_depth}
    eq_cands = [c for c in eq_cands if c[0] not in used]
    picked_eq: list = []
    _diverse(picked_eq, lambda c: c[3], eq_cands, n_section_eq)
    if len(picked_eq) < n_section_eq or len(picked_depth) < n_section_depth:
        raise ValueError(
            f"insufficient filter candidates: eq={len(picked_eq)}/{n_section_eq} "
            f"depth={len(picked_depth)}/{n_section_depth}"
        )

    rows: list[NeonEvalRecord] = []
    for lex, gold, decoys, section in picked_eq:
        rows.append(
            NeonEvalRecord(
                capability="filter_section_eq",
                search_mode="lexical",
                query=lex,
                filter_dsl=_section_filter(section),
                gold_chunk_hashes=[gold],
                decoy_chunk_hashes=decoys,
            )
        )
    for lex, gold, decoys, section, depth in picked_depth:
        rows.append(
            NeonEvalRecord(
                capability="filter_section_depth",
                search_mode="lexical",
                query=lex,
                filter_dsl=_section_and_depth_filter(section, depth),
                gold_chunk_hashes=[gold],
                decoy_chunk_hashes=decoys,
            )
        )
    return rows
