"""Deterministic per-row overrides applied at the END of the freeze.

Two hand-audited corrections that the generic expansion pipeline gets wrong on a
specific row. Each is keyed by a stable chunk-hash prefix (not by line number) and
records an audit entry, so the change is reproducible and review-addressable.

* GOLD_RESTRICTIONS — a row whose equivalence/multi-gold expansion admitted chunks
  that are NOT truly equivalent. We intersect its gold with an explicit allowlist of
  the genuinely-equivalent chunks. VT:45 (anchor ``b0e04cd6``, a Testing Agreement
  "FEEDBACK" clause) had its gold expanded across all five testing-agreement
  versions, but only the ``v5`` copy (``1ba40196``) is BYTE-IDENTICAL to the anchor;
  ``v1/v2/v3/v4`` are different clauses or reworded, so they are pruned.

* QUERY_ANCHORS — a templated/non-unique natural row re-anchored so its gold is
  uniquely correct, with gold pinned to the single answering chunk. LT:8's query
  ("optional async daily updates channel") matched a boilerplate stand-up block
  repeated on many team pages; it is re-anchored on the distinctive team + channel
  so it points at the one Software-Supply-Chain-Security:Compliance chunk.

* QUERY_REWRITES — a query that LEAKS its own answer, rewritten to remove the leak
  (gold unchanged). AL-01 (VEC-de3f14) named "Belief Bias" in the question that asks
  for that bias; AL-02 (VEC-3b0544) appended the answer "Team Member Relations?".

* ROW_DROPS — an under-specified/templated natural row dropped outright. TMPL-01
  (VEC-25ac78) asks for a competency triad that recurs verbatim across ~42 career
  matrices, so no single gold is uniquely correct.
"""

from __future__ import annotations

from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

# (row_ref, anchor_prefix, keep_prefixes, prune_prefixes, reason)
GOLD_RESTRICTIONS = [
    (
        "VT:45",
        "b0e04cd6",
        ["b0e04cd6", "1ba40196"],
        ["44a52648", "ea25e033", "88dac88e", "5eba960c"],
        "byte-identical FEEDBACK clause only (anchor + v5); v1/v2/v3/v4 are reworded "
        "or different clauses, not truly equivalent",
    ),
]

# (row_ref, match_query, new_query, gold_prefixes, reason)
QUERY_ANCHORS = [
    (
        "LT:8",
        "optional async daily updates channel",
        "software supply chain security compliance asynchronous stand-up channel",
        ["355e0f51"],
        "templated stand-up boilerplate recurs across team pages; anchored on the "
        "SSCS:Compliance team + channel so the gold is uniquely correct (drops the "
        "generic devops copy 46842214)",
    ),
]

# (row_ref, match_query, new_query, reason) — remove an answer leak; gold unchanged.
QUERY_REWRITES = [
    (
        "AL-01",
        "At GitLab, under Belief Bias, what is the mistake called where someone rates "
        "an idea based more on whether they already agree with it than on how well it "
        "actually supports the outcome?",
        "At GitLab, in explaining why playing politics runs counter to our values, "
        "what is the name of the cognitive bias where someone judges an argument's "
        "strength by whether they already agree with its conclusion rather than by "
        "how well it actually supports that conclusion?",
        "query named the answer 'Belief Bias'; re-anchored on the non-answer "
        "'playing politics counter to our values' context (gold acfc5fd2 unchanged)",
    ),
    (
        "AL-02",
        "If a manager is seeing repeated signs that someone isn’t meeting role "
        "expectations, which group should they bring in early to help with coaching, "
        "legal compliance, and making sure feedback is handled correctly: Team Member "
        "Relations?",
        "If a manager is seeing repeated signs that someone isn’t meeting role "
        "expectations, which group should they bring in early to help with coaching, "
        "legal compliance, and making sure feedback is handled correctly?",
        "query supplied the answer 'Team Member Relations?'; trailing answer removed",
    ),
]

# (row_ref, match_query, reason) — drop an under-specified/templated row outright.
ROW_DROPS = [
    (
        "TMPL-01",
        "At GitLab, for the experienced individual-contributor stage in engineering, "
        "what are the three expectation groupings listed on that role framework page?",
        "competency triad recurs verbatim across ~42 career matrices; no single gold "
        "is uniquely correct (under-specified templated query)",
    ),
]


def _has_prefix(hashes: list[str], prefix: str) -> bool:
    return any(h.startswith(prefix) for h in hashes)


def apply_overrides(records: list[NeonEvalRecord]) -> tuple[list[NeonEvalRecord], list[dict]]:
    """Apply the gold restrictions and query anchors; return (records, audit).

    Each override asserts its target row is present and still exhibits the defect it
    corrects (a byte-identical prefix to keep, or the prune targets present), so a
    corpus or pipeline change that removes the target surfaces loudly instead of
    silently no-opping.
    """
    audit: list[dict] = []
    out = list(records)

    for row_ref, anchor, keep, prune, reason in GOLD_RESTRICTIONS:
        hit = [r for r in out if _has_prefix(r.gold_chunk_hashes, anchor)]
        if not hit:
            raise AssertionError(f"{row_ref}: anchor {anchor} not found in any row gold")
        for r in hit:
            before = list(r.gold_chunk_hashes)
            # The row must still exhibit the defect: every prune target present so a
            # corpus/pipeline change that already removed one surfaces loudly.
            missing = [p for p in prune if not _has_prefix(before, p)]
            if missing:
                raise AssertionError(
                    f"{row_ref}: expected prune targets absent (already gone?): {missing}"
                )
            kept = [h for h in before if any(h.startswith(p) for p in keep)]
            pruned = [h for h in before if h not in kept]
            if not kept:
                raise AssertionError(f"{row_ref}: restriction left no gold")
            r.gold_chunk_hashes = kept
            audit.append({
                "row_ref": row_ref, "action": "gold_restriction",
                "kept": [h[:8] for h in kept], "pruned": [h[:8] for h in pruned],
                "reason": reason,
            })

    for row_ref, match_q, new_q, gold_prefixes, reason in QUERY_ANCHORS:
        hit = [r for r in out if r.query == match_q]
        if not hit:
            raise AssertionError(f"{row_ref}: query {match_q!r} not found")
        for r in hit:
            before_gold = list(r.gold_chunk_hashes)
            kept = [h for h in before_gold if any(h.startswith(p) for p in gold_prefixes)]
            if not kept:
                raise AssertionError(f"{row_ref}: no gold matches {gold_prefixes}")
            r.query = new_q
            r.gold_chunk_hashes = kept
            audit.append({
                "row_ref": row_ref, "action": "query_anchor",
                "new_query": new_q, "gold": [h[:8] for h in kept],
                "dropped_gold": [h[:8] for h in before_gold if h not in kept],
                "reason": reason,
            })

    for row_ref, match_q, new_q, reason in QUERY_REWRITES:
        hit = [r for r in out if r.query == match_q]
        if not hit:
            raise AssertionError(f"{row_ref}: query {match_q!r} not found")
        for r in hit:
            r.query = new_q
            audit.append({
                "row_ref": row_ref, "action": "query_rewrite",
                "new_query": new_q, "gold": [h[:8] for h in r.gold_chunk_hashes],
                "reason": reason,
            })

    dropped_queries = {match_q for _ref, match_q, _reason in ROW_DROPS}
    for row_ref, match_q, reason in ROW_DROPS:
        if not any(r.query == match_q for r in out):
            raise AssertionError(f"{row_ref}: drop target {match_q!r} not found")
        audit.append({"row_ref": row_ref, "action": "row_drop", "reason": reason})
    out = [r for r in out if r.query not in dropped_queries]

    return out, audit
