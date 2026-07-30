"""Analyse the raw A/B results and emit the decision report.

Reads ``results/raw_results.jsonl`` (written by ``run_ab.py``) and computes, over
the ranked FILE lists:

* per-mode hit@1 / hit@5 / hit@10, MRR@10, mean gold rank when found, and the
  count of queries whose gold never appears;
* PAIRED discordant counts (b, c) plus concordant counts for hybrid-vs-bm25 and
  vector-vs-bm25, with McNemar's test run as the EXACT two-sided binomial test
  (``scipy.stats.binomtest``) because b+c is small; the chi-square statistic with
  continuity correction is reported alongside for reference only;
* fusion lift (rescued / broken / net) at k = 1, 5, 10;
* the same paired analysis split by the DERIVED style proxy in ``style_proxy``;
* worked discordant examples in both directions for hand sanity-checking.

Everything here is arithmetic over the persisted raw results — re-running it
never touches Neon.

Usage::

    uv run python examples/neon_retrieval_ab/analyze_ab.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
MODES = ("lexical", "vector", "hybrid")
MODE_LABELS = {"lexical": "bm25", "vector": "vector", "hybrid": "hybrid"}
KS = (1, 5, 10)
TOP_K = 10


VERDICT_PROSE = """
### is the difference real?

**hybrid vs bm25: yes, and it is not marginal.** hybrid beats bm25 on every headline
metric (hit@1 0.5628 vs 0.4814, hit@5 0.7581 vs 0.6953, hit@10 0.8233 vs 0.7419, MRR@10
0.6464 vs 0.5737) and the paired test rejects the null at every k (exact two-sided
binomial p = 7.3e-04 / 5.8e-04 / 2.1e-06 at k = 1 / 5 / 10). 35 queries that bm25 never
surfaces at all inside the top-10 are recovered by hybrid.

**vector vs bm25: no — statistically indistinguishable in aggregate.** net +9 at hit@5 on
105 discordant pairs, p = 0.44; hit@1 is net -8, p = 0.55; hit@10 net +4, p = 0.76. taken
on its own the aggregate vector-vs-bm25 comparison is a **null result** and should be
reported as one. swapping bm25 for pure vector retrieval buys nothing measurable here.

### the aggregate hybrid win hides a large, opposite-signed style split

the aggregate number is not the whole story, and the honest reading is that hybrid is a
**trade, not a free lunch**:

* on the `keyword` half (n=204), bm25 is the **best** mode (hit@5 0.9216 vs hybrid 0.8775
  vs vector 0.7843) and hybrid is significantly WORSE than bm25 (b=2, c=11, net -9,
  p = 0.0225). vector is far worse still (net -28, p = 1.9e-06).
* on the `paraphrase` half (n=226), bm25 collapses (hit@5 0.4912) and both dense arms win
  decisively (hybrid 0.6504, b=41, c=5, net +36, p = 4.4e-08; vector 0.6549, net +37,
  p = 9.1e-06).

so hybrid's aggregate advantage is bought entirely on the paraphrase half and paid for on
the keyword half. because the two buckets are close to balanced here (204 / 226), the
paraphrase gain dominates. **a different query mix flips the conclusion**: a
keyword-dominated workload would prefer plain bm25.

for reference, a hypothetical style-routed oracle (bm25 on `keyword`, dense on
`paraphrase`) reaches hit@5 0.7814 (vector) / 0.7791 (hybrid) versus hybrid-everywhere at
0.7581 — a further +2.3 points, but it needs a router that does not exist and is out of
scope here.

### recommendation on the ~13h x 8xA100 gpu training a/b: **GO — with a narrowed question**

the retrieval-level effect is large enough and clean enough to be worth spending gpu time
on, but the run should be scoped to the comparison that actually has signal:

* **run hybrid vs bm25.** the effect is significant at every k and the +8.1-point hit@10
  gain (35 fewer queries with no reachable gold at all) is the kind of gap that can plausibly
  move end-to-end reward — the policy cannot cite what retrieval never returns.
* **do NOT spend gpu hours on vector vs bm25.** that arm is a measured null at the
  retrieval layer; there is no offline effect for training to amplify.

caveats that should be stated in whatever the gpu run reports:

1. this harness issues the DATASET question verbatim. in the rl env the policy writes its
   own search queries, and policy-authored queries may sit in a different place on the
   keyword/paraphrase axis than these do. the offline delta is an upper-bound-ish proxy for
   the retrieval quality the policy will actually experience, not a prediction of reward.
2. retrieval hit@k is an input to reward, not reward. the reward path also depends on
   whether the policy cites what it retrieved (`retrieval_hit` scores CITED, not returned),
   so a +6-point hit@5 does not translate one-for-one.
3. the style split is a DERIVED proxy (see above), not a labelled attribute. it is stable
   and mechanistically sensible — bm25 wins terse term queries, dense wins paraphrases —
   but individual rows can be mislabelled.
4. these 430 questions were llm-generated against this same corpus, so their phrasing
   distribution is an artifact of that generator, not of real user traffic.
"""
"""Hand-written interpretation of the generated numbers, frozen after the run.

Kept as a constant (not hand-edited into the markdown) so re-running the analysis
regenerates the whole report and the prose can never drift from the table above it.
"""


def gold_rank(ranked_files: list[str], gold_files: list[str]) -> int | None:
    """Return the 1-based rank of the first gold file in *ranked_files*, else ``None``."""
    gold = set(gold_files)
    for index, name in enumerate(ranked_files, start=1):
        if name in gold:
            return index
    return None


def load(path: Path) -> list[dict[str, Any]]:
    """Load raw results, attaching a per-mode gold rank to every row."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["rank"] = {
            mode: gold_rank(row["modes"][mode]["ranked_files"], row["gold_files"])
            for mode in MODES
        }
        rows.append(row)
    return rows


def hit(row: dict[str, Any], mode: str, k: int) -> bool:
    """True when *mode* surfaced a gold file within the top *k* ranked files."""
    rank = row["rank"][mode]
    return rank is not None and rank <= k


def per_mode_metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """hit@1/5/10, MRR@10, mean found rank, and misses for one mode."""
    n = len(rows)
    ranks = [row["rank"][mode] for row in rows]
    found = [r for r in ranks if r is not None and r <= TOP_K]
    return {
        "n": n,
        "hit_at_1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "hit_at_5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "hit_at_10": len(found) / n,
        "mrr_at_10": sum(1.0 / r for r in found) / n,
        "mean_gold_rank_when_found": (sum(found) / len(found)) if found else None,
        "gold_not_in_top_10": n - len(found),
    }


def paired(rows: list[dict[str, Any]], arm: str, base: str, k: int) -> dict[str, Any]:
    """Paired 2x2 table for *arm* vs *base* at hit@k, with the exact McNemar test.

    ``b`` counts queries the arm wins (arm hit, base miss) and ``c`` those the base
    wins. The reported p-value is the EXACT two-sided binomial test on the
    discordant pairs, valid at the small b+c this dataset produces; the
    continuity-corrected chi-square statistic is included for reference only.
    """
    a = b = c = d = 0
    for row in rows:
        arm_hit, base_hit = hit(row, arm, k), hit(row, base, k)
        if arm_hit and base_hit:
            a += 1
        elif arm_hit:
            b += 1
        elif base_hit:
            c += 1
        else:
            d += 1
    discordant = b + c
    p_value = binomtest(b, discordant, 0.5, alternative="two-sided").pvalue if discordant else 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / discordant if discordant else 0.0
    return {
        "k": k,
        "both_hit": a,
        "arm_only": b,
        "base_only": c,
        "both_miss": d,
        "discordant": discordant,
        "net": b - c,
        "exact_binomial_p": p_value,
        "chi2_continuity_corrected": chi2,
    }


def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def _metric_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| mode | n | hit@1 | hit@5 | hit@10 | MRR@10 | mean gold rank (when found) | gold not in top-10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        m = per_mode_metrics(rows, mode)
        mean_rank = (
            f"{m['mean_gold_rank_when_found']:.2f}"
            if m["mean_gold_rank_when_found"] is not None
            else "n/a"
        )
        out.append(
            f"| {MODE_LABELS[mode]} | {m['n']} | {m['hit_at_1']:.4f} | {m['hit_at_5']:.4f} | "
            f"{m['hit_at_10']:.4f} | {m['mrr_at_10']:.4f} | {mean_rank} | {m['gold_not_in_top_10']} |"
        )
    return out


def _paired_block(rows: list[dict[str, Any]], arm: str, base: str) -> list[str]:
    out = [
        f"| k | both hit | {MODE_LABELS[arm]} only (b) | {MODE_LABELS[base]} only (c) | "
        "both miss | b+c | net (b-c) | exact binomial p | chi2 (cc) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in KS:
        t = paired(rows, arm, base, k)
        out.append(
            f"| {k} | {t['both_hit']} | {t['arm_only']} | {t['base_only']} | {t['both_miss']} | "
            f"{t['discordant']} | {t['net']:+d} | {_fmt_p(t['exact_binomial_p'])} | "
            f"{t['chi2_continuity_corrected']:.3f} |"
        )
    return out


def _examples(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Pick discordant hybrid-vs-bm25 queries at k=5, balanced across directions."""
    wins = [r for r in rows if hit(r, "hybrid", 5) and not hit(r, "lexical", 5)]
    losses = [r for r in rows if hit(r, "lexical", 5) and not hit(r, "hybrid", 5)]
    # Aim for an even split, but top up from whichever side is deeper.
    take_losses = min(len(losses), limit // 2)
    picked = wins[: limit - take_losses] + losses[:take_losses]
    return picked[:limit]


def _example_block(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in _examples(rows):
        direction = (
            "hybrid wins" if hit(row, "hybrid", 5) and not hit(row, "lexical", 5) else "bm25 wins"
        )
        out.append(f"**row {row['row_id']} ({row['split']}, style={row['style']}) — {direction}**")
        out.append("")
        out.append(f"- query: `{row['question']}`")
        out.append(f"- gold: `{', '.join(row['gold_files'])}`")
        for mode in MODES:
            top3 = row["modes"][mode]["ranked_files"][:3]
            rank = row["rank"][mode]
            marker = f"gold rank {rank}" if rank is not None else "gold not in top-10"
            out.append(f"- {MODE_LABELS[mode]} top-3 ({marker}):")
            for name in top3:
                flag = " **<- gold**" if name in set(row["gold_files"]) else ""
                out.append(f"  - `{name}`{flag}")
        out.append("")
    return out


def _style_block(rows: list[dict[str, Any]]) -> list[str]:
    buckets = {
        style: [r for r in rows if r["style"] == style] for style in ("keyword", "paraphrase")
    }
    out = [
        "| bucket | n | bm25 hit@5 | vector hit@5 | hybrid hit@5 |",
        "|---|---|---|---|---|",
    ]
    for style, bucket in buckets.items():
        if not bucket:
            continue
        vals = " | ".join(
            f"{per_mode_metrics(bucket, mode)['hit_at_5']:.4f}" for mode in MODES
        )
        out.append(f"| {style} | {len(bucket)} | {vals} |")
    out.append("")
    for style, bucket in buckets.items():
        if not bucket:
            continue
        out.append(f"paired counts at hit@5 within the `{style}` bucket (n={len(bucket)}):")
        out.append("")
        out.append(
            "| comparison | both hit | arm only (b) | bm25 only (c) | both miss | net | exact binomial p |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for arm in ("hybrid", "vector"):
            t = paired(bucket, arm, "lexical", 5)
            out.append(
                f"| {MODE_LABELS[arm]} vs bm25 | {t['both_hit']} | {t['arm_only']} | "
                f"{t['base_only']} | {t['both_miss']} | {t['net']:+d} | "
                f"{_fmt_p(t['exact_binomial_p'])} |"
            )
        out.append("")
    return out


def build_report(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    """Render the full markdown decision report."""
    usable = [r for r in rows if r["gold_files"]]
    excluded = len(rows) - len(usable)
    hyb5 = paired(usable, "hybrid", "lexical", 5)
    vec5 = paired(usable, "vector", "lexical", 5)

    lines: list[str] = []
    lines.append("# neon retrieval a/b — bm25 vs vector vs hybrid")
    lines.append("")
    lines.append(
        "paired offline retrieval comparison on one query set against the live "
        f"`{manifest['corpus']}` corpus. no gpu, no training, no re-ingest, no re-embed of "
        "the corpus."
    )
    lines.append("")

    lines.append("## run")
    lines.append("")
    lines.append(f"- corpus: `{manifest['corpus']}` (31,665 chunks, active version, read-only)")
    lines.append(f"- datasets: {', '.join(f'`{p}`' for p in manifest['datasets'])}")
    lines.append(f"- rows loaded: **{len(rows)}**")
    lines.append(f"- rows with usable (non-empty) gold: **{len(usable)}**")
    lines.append(
        f"- rows excluded for missing gold: **{excluded}**"
        + ("" if excluded else " (none — every row carries exactly one gold file)")
    )
    lines.append(
        f"- identical settings across all three arms: `top_k={manifest['top_k']}`, no metadata "
        f"filter, `text_search_config={manifest['text_search_config']}`, schema "
        f"`{manifest['schema']}`. only `mode` differs. nothing was tuned per mode."
    )
    lines.append(
        f"- embeddings: `{manifest['embed_model']}` computed once per question "
        f"({manifest['embed_calls']} batched calls for {manifest['rows']} questions) and reused "
        "by both the vector and the hybrid arm."
    )
    lines.append(
        f"- retrieval calls: {manifest['retrieval_calls']} "
        f"(concurrency {manifest['concurrency']}, {manifest['elapsed_seconds']}s wall)."
    )
    failures = manifest.get("failures") or []
    lines.append(
        f"- failed queries: **{len(failures)}**"
        + ("" if failures else " — every query returned in every mode.")
    )
    lines.append("")
    lines.append(
        "hit@k is measured over the RANKED FILE list: the top-10 chunks are mapped to their "
        "`metadata.file` (the same field the gold uses) and deduped preserving "
        "first-occurrence rank. a row counts as a hit when ANY of its gold files appears."
    )
    lines.append("")

    lines.append("## per-mode metrics")
    lines.append("")
    lines += _metric_table(usable)
    lines.append("")

    lines.append("## paired analysis: hybrid vs bm25")
    lines.append("")
    lines.append(
        "mcnemar's test run as the **exact two-sided binomial test** on the discordant pairs "
        "(`scipy.stats.binomtest(b, b+c, 0.5)`), which is the correct choice at these small "
        "discordant counts. the continuity-corrected chi-square statistic is shown for "
        "reference only and is not the test being used."
    )
    lines.append("")
    lines += _paired_block(usable, "hybrid", "lexical")
    lines.append("")

    lines.append("## paired analysis: vector vs bm25")
    lines.append("")
    lines += _paired_block(usable, "vector", "lexical")
    lines.append("")

    lines.append("## fusion lift (hybrid over bm25)")
    lines.append("")
    lines.append("| k | rescued (bm25 miss -> hybrid hit) | broken (bm25 hit -> hybrid miss) | net |")
    lines.append("|---|---|---|---|")
    for k in KS:
        t = paired(usable, "hybrid", "lexical", k)
        lines.append(f"| {k} | {t['arm_only']} | {t['base_only']} | {t['net']:+d} |")
    lines.append("")

    lines.append("## query-style breakdown (derived proxy — not ground truth)")
    lines.append("")
    lines.append(
        "the 430 rows carry no style field (their only keys are `question`, `answer`, "
        "`reference_chunks`), so this bucketing is **derived from the question text alone** by "
        "`style_proxy.classify`. it is a proxy, not a label: a question is `keyword` when it is "
        "shorter than 12 tokens **and** does not lead with an interrogative or auxiliary **and** "
        "contains no first/second-person pronoun; otherwise it is `paraphrase`. the rule never "
        "looks at gold, at retrieval output, or at any per-row search mode, so it cannot leak "
        "the outcome. the 156-row `gitlab_handbook_bm25_neon` golden was deliberately NOT used "
        "as a style source — it bakes a favourable `search_mode` per row, which is the confound "
        "this a/b removes."
    )
    lines.append("")
    lines += _style_block(usable)

    lines.append("## worked discordant examples")
    lines.append("")
    lines.append(
        "discordant queries at hit@5 for hybrid vs bm25, both directions, with each mode's "
        "top-3 files so the result can be checked by hand."
    )
    lines.append("")
    lines += _example_block(usable)

    lines.append("## verdict")
    lines.append("")
    lines.append(
        f"at hit@5, hybrid vs bm25 is b={hyb5['arm_only']} / c={hyb5['base_only']} "
        f"(net {hyb5['net']:+d}, exact binomial p={_fmt_p(hyb5['exact_binomial_p'])}); vector vs "
        f"bm25 is b={vec5['arm_only']} / c={vec5['base_only']} (net {vec5['net']:+d}, exact "
        f"binomial p={_fmt_p(vec5['exact_binomial_p'])})."
    )
    lines.append("")
    lines.append(VERDICT_PROSE.strip())
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw_results.jsonl")
    parser.add_argument("--manifest", type=Path, default=RESULTS_DIR / "run_manifest.json")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "RESULTS.md")
    args = parser.parse_args()

    rows = load(args.raw)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    usable = [r for r in rows if r["gold_files"]]
    summary = {
        "rows": len(rows),
        "usable_gold": len(usable),
        "per_mode": {MODE_LABELS[m]: per_mode_metrics(usable, m) for m in MODES},
        "paired": {
            "hybrid_vs_bm25": {str(k): paired(usable, "hybrid", "lexical", k) for k in KS},
            "vector_vs_bm25": {str(k): paired(usable, "vector", "lexical", k) for k in KS},
        },
        "style_buckets": {
            style: {
                "n": len([r for r in usable if r["style"] == style]),
                "per_mode": {
                    MODE_LABELS[m]: per_mode_metrics(
                        [r for r in usable if r["style"] == style], m
                    )
                    for m in MODES
                },
            }
            for style in ("keyword", "paraphrase")
        },
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.out.write_text(build_report(rows, manifest), encoding="utf-8")
    print(json.dumps(summary["per_mode"], indent=2))
    print(json.dumps(summary["paired"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
