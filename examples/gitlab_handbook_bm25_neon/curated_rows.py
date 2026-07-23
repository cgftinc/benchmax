"""Hand-curated FILTER + HYBRID golden rows (Path Y), authored BLIND.

qa-gen produces the lexical / vector rows; the FILTER and HYBRID capabilities are
authored here instead (Path X — extending qa-gen with a filter predicate — is an
explicit followup, not built). "Hand-curated" means authored from CHUNK CONTENT,
never from a Neon retrieval result, so the gold ids stay independent of what the
provider happens to return (F12/B8 non-circularity).

Construction is programmatic but principled, so it is reproducible and every gold
/ decoy id is an exact chunk hash rather than a hand-typed guess:

* pick a distinctive token that occurs in EXACTLY ONE chunk of a target
  ``handbook_section`` but ALSO in chunks of other sections;
* the unique in-section chunk is the GOLD (a section-filtered search for the
  token must surface it), and the same-token chunks in OTHER sections are the
  DECOYS (the section filter must EXCLUDE them — if one is surfaced the filter
  failed);
* FILTER rows exercise ``handbook_section`` equality; a subset AND a numeric
  ``path_depth`` predicate so both derived metadata keys and the DSL's ``and`` /
  range operators are covered; HYBRID rows exercise the same filter through RRF
  fusion.

Because the token is unique within the target section, the correct answer ranks
first under the filter, so these rows carry tight thresholds honestly (not a
"non-empty" assertion).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

# Token shape for a "distinctive" term: a real lowercase word (no digits/codes),
# 6-20 chars, so the curated queries read like handbook vocabulary rather than
# part numbers.
_TOKEN_RE = re.compile(r"\b[a-z]{6,20}\b")

# A term is usable only if it is globally rare (a handful of chunks) so the single
# in-section occurrence ranks at the top and the cross-section occurrences are a
# small, checkable decoy set.
_MIN_GLOBAL_CHUNKS = 2
_MAX_GLOBAL_CHUNKS = 8
_MAX_DECOYS = 3

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
        "https",
        "example",
        "following",
    }
)


@dataclass(frozen=True)
class _TokenChunk:
    token: str
    hash: str
    section: str
    path_depth: int


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _index(collection: ChunkCollection) -> dict[str, list[_TokenChunk]]:
    """Map each distinctive token to the chunks that contain it (stable order)."""
    postings: dict[str, list[_TokenChunk]] = defaultdict(list)
    for chunk in _sorted_chunks(collection):
        md = dict(chunk.metadata)
        section = str(md.get("handbook_section", ""))
        depth = int(md.get("path_depth", 0) or 0)
        for tok in _tokens(chunk.content):
            postings[tok].append(_TokenChunk(tok, chunk.hash, section, depth))
    return postings


def _sorted_chunks(collection: ChunkCollection) -> list[Chunk]:
    # Deterministic order independent of collection construction: sort by hash.
    return sorted(collection.chunks, key=lambda c: c.hash)


def _candidates(
    postings: dict[str, list[_TokenChunk]],
) -> list[tuple[str, _TokenChunk, list[_TokenChunk]]]:
    """Yield (token, gold, decoys) where the token is unique in the gold's section.

    Deterministic: tokens are visited in sorted order and the first section with a
    single occurrence wins.
    """
    out: list[tuple[str, _TokenChunk, list[_TokenChunk]]] = []
    for token in sorted(postings):
        entries = postings[token]
        if not (_MIN_GLOBAL_CHUNKS <= len(entries) <= _MAX_GLOBAL_CHUNKS):
            continue
        by_section: dict[str, list[_TokenChunk]] = defaultdict(list)
        for e in entries:
            by_section[e.section].append(e)
        # a target section with exactly one occurrence + at least one other section
        unique_sections = sorted(s for s, es in by_section.items() if len(es) == 1)
        for section in unique_sections:
            gold = by_section[section][0]
            decoys = [e for e in entries if e.section != section][:_MAX_DECOYS]
            if decoys:
                out.append((token, gold, decoys))
                break
    return out


def _diverse_select(
    cands: list[tuple[str, _TokenChunk, list[_TokenChunk]]], n: int
) -> list[tuple[str, _TokenChunk, list[_TokenChunk]]]:
    """Pick ``n`` candidates spread across as many sections as possible (stable)."""
    by_section: dict[str, list[tuple[str, _TokenChunk, list[_TokenChunk]]]] = (
        defaultdict(list)
    )
    for cand in cands:
        by_section[cand[1].section].append(cand)
    picked: list[tuple[str, _TokenChunk, list[_TokenChunk]]] = []
    sections = sorted(by_section)
    idx = 0
    while len(picked) < n and any(by_section[s] for s in sections):
        section = sections[idx % len(sections)]
        if by_section[section]:
            picked.append(by_section[section].pop(0))
        idx += 1
    return picked


def _section_filter(section: str) -> dict:
    return {"field": "handbook_section", "op": "eq", "value": section}


def _section_and_depth_filter(section: str, depth: int) -> dict:
    return {
        "and": [
            {"field": "handbook_section", "op": "eq", "value": section},
            {"field": "path_depth", "op": "eq", "value": depth},
        ]
    }


def _multi_term_query(content: str, seed: str, freq: dict[str, int], k: int = 5) -> str:
    """Build a multi-term query: the seed token plus the chunk's rarest other words.

    A single-token query is a lexical strength that the vector leg only dilutes; a
    handful of the gold chunk's own distinctive words instead gives BOTH legs a
    real signal (the lexical leg matches several terms, the vector leg embeds a
    contentful phrase), so fusion helps rather than hurts.
    """
    seen = {seed}
    terms = [seed]
    for tok in sorted(_tokens(content), key=lambda t: (freq.get(t, 0), t)):
        if tok not in seen and freq.get(tok, 0) >= _MIN_GLOBAL_CHUNKS:
            terms.append(tok)
            seen.add(tok)
        if len(terms) >= k:
            break
    return " ".join(terms)


def build_curated_rows(
    collection: ChunkCollection,
    *,
    n_filter: int = 10,
    n_hybrid: int = 8,
) -> list[NeonEvalRecord]:
    """Build the FILTER + HYBRID golden rows from the collection (deterministic).

    The first ``n_filter`` candidates become lexical+section-filter rows (a third
    of them additionally AND a ``path_depth`` predicate) whose query is a single
    distinctive token — a filter-precision probe. The next ``n_hybrid`` become
    hybrid+section-filter rows whose query is a MULTI-TERM phrase built from the
    gold chunk's own rare words, so lexical and vector legs reinforce rather than
    fight. Every row's gold is the unique in-section chunk; decoys are the
    same-token chunks in other sections that the filter must exclude.

    Args:
        collection: The re-chunked handbook collection (in-memory, with hashes).
        n_filter: Number of lexical filter rows to emit.
        n_hybrid: Number of hybrid filter rows to emit.
    """
    postings = _index(collection)
    freq = {tok: len(entries) for tok, entries in postings.items()}
    content_by_hash = {c.hash: c.content for c in collection.chunks}
    cands = _candidates(postings)
    if len(cands) < n_filter + n_hybrid:
        raise ValueError(
            f"only {len(cands)} curated candidates found; need {n_filter + n_hybrid}"
        )
    selected = _diverse_select(cands, n_filter + n_hybrid)

    rows: list[NeonEvalRecord] = []
    for i, (token, gold, decoys) in enumerate(selected[:n_filter]):
        use_depth = i % 3 == 0  # every third row also pins path_depth
        rows.append(
            NeonEvalRecord(
                capability="filter_section_depth" if use_depth else "filter_section_eq",
                search_mode="lexical",
                query=token,
                filter_dsl=(
                    _section_and_depth_filter(gold.section, gold.path_depth)
                    if use_depth
                    else _section_filter(gold.section)
                ),
                gold_chunk_hashes=[gold.hash],
                decoy_chunk_hashes=[d.hash for d in decoys],
            )
        )
    for token, gold, decoys in selected[n_filter : n_filter + n_hybrid]:
        query = _multi_term_query(content_by_hash.get(gold.hash, token), token, freq)
        rows.append(
            NeonEvalRecord(
                capability="hybrid_section_filter",
                search_mode="hybrid",
                query=query,
                filter_dsl=_section_filter(gold.section),
                gold_chunk_hashes=[gold.hash],
                decoy_chunk_hashes=[d.hash for d in decoys],
            )
        )
    return rows
