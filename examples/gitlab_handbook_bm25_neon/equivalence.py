"""Blind equivalence-set expansion of gold via near-duplicate corpus chunks.

Exact-hash scoring unfairly penalizes pairs whose answer is *templated* — text
that recurs near-identically across many chunks (e.g. the career-competency triad
repeated on ~42 role pages, the confidentiality/integrity/availability triad on
~17 security pages). Retrieving any of those chunks answers the query equally, so
they form an equivalence set and a hit on any member counts.

The set is computed blind of retrieval: for each gold chunk it takes the corpus's
own nearest neighbours by cosine over the ALREADY-STORED embeddings (no re-embed),
then keeps only those that are BOTH highly similar (cosine >= threshold) AND
near-duplicate in content (token Jaccard >= ``min_jaccard``) AND in the same
handbook section. The content + section guards keep the union to genuine templated
duplicates rather than merely-topical neighbours, so the graded metric never
credits a similar-but-distinct chunk. This is a property of the corpus, not of any
query or retrieval result, so it does not compromise non-circularity.
"""

from __future__ import annotations

import re

from castform.rag.corpus.neon.client import NeonClient
from castform.rag.corpus.neon.schema import NeonTableSpec, physical_table_name

_WORD_RE = re.compile(r"[a-z0-9]+")

DEFAULT_COSINE_THRESHOLD = 0.85
DEFAULT_MIN_JACCARD = 0.5
DEFAULT_TOP_K = 128
DEFAULT_CAP = 64


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_equivalence_sets(
    client: NeonClient,
    spec: NeonTableSpec,
    gold_hashes: list[str],
    *,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    min_jaccard: float = DEFAULT_MIN_JACCARD,
    top_k: int = DEFAULT_TOP_K,
    cap: int = DEFAULT_CAP,
) -> dict[str, list[str]]:
    """Map each gold hash to its templated-equivalence set (itself included).

    Args:
        client: A read client over the corpus (RO).
        spec: The current corpus version.
        gold_hashes: Distinct gold chunk hashes to expand.
        cosine_threshold: Minimum cosine similarity for a candidate duplicate.
        min_jaccard: Minimum content-token Jaccard for a candidate duplicate.
        top_k: ANN candidate depth scanned per gold chunk.
        cap: Maximum size of any one equivalence set.
    """
    from psycopg import sql

    table = sql.Identifier(physical_table_name(spec.logical_name, spec.version))
    ids = sorted(set(gold_hashes))
    seed = client.execute(
        sql.SQL(
            "SELECT id, embedding::text, content, "
            "metadata->>'handbook_section' FROM {} WHERE id = ANY(%(ids)s)"
        ).format(table),
        {"ids": ids},
    )
    by_id = {r[0]: (r[1], r[2], r[3]) for r in seed}
    max_dist = 1.0 - cosine_threshold
    out: dict[str, list[str]] = {}
    for h in ids:
        if h not in by_id:
            out[h] = [h]
            continue
        vtext, content, section = by_id[h]
        gold_toks = _tokens(content)
        neighbours = client.execute(
            sql.SQL(
                "SELECT id, (embedding <=> %(v)s::vector) AS dist, content, "
                "metadata->>'handbook_section' AS section "
                "FROM {} ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s"
            ).format(table),
            {"v": vtext, "k": top_k},
        )
        equiv = [h]
        for nid, dist, ncontent, nsection in neighbours:
            if nid == h or dist is None or float(dist) > max_dist:
                continue
            if nsection != section:
                continue
            if _jaccard(gold_toks, _tokens(ncontent)) < min_jaccard:
                continue
            equiv.append(nid)
            if len(equiv) >= cap:
                break
        out[h] = list(dict.fromkeys(equiv))
    return out


def params() -> dict[str, float | int]:
    """The equivalence parameters, for the provenance manifest."""
    return {
        "cosine_threshold": DEFAULT_COSINE_THRESHOLD,
        "min_content_jaccard": DEFAULT_MIN_JACCARD,
        "ann_top_k": DEFAULT_TOP_K,
        "cap": DEFAULT_CAP,
        "same_section_required": True,
    }
