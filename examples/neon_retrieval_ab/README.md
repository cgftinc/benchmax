# neon retrieval a/b

paired offline a/b of the three neon retrieval modes (`lexical` / bm25, `vector`,
`hybrid`) on ONE shared query set against the live `gitlab_handbook_neon` corpus.
answers a single question: is a gpu training a/b between retrieval modes worth
running?

no gpu, no training, no re-ingest, no re-embed of the corpus. read-only.

## what it does

`run_ab.py` sends every query through the production query layer
(`castform.rag.corpus.neon.query.run_query`) once per mode with **identical**
settings — same corpus version, `top_k=10`, no metadata filter, same
`text_search_config`. `mode` is the only difference between arms; there is no
per-mode knob and nothing was tuned. the query embedding is computed once per
question and reused by the vector and hybrid arms.

returned chunks are mapped to `metadata.file` — the same field the dataset gold
uses — and deduped to a ranked FILE list preserving first-occurrence rank; all
metrics are computed over that list.

`analyze_ab.py` reads the persisted raw results (never re-queries neon) and
computes per-mode hit@1/5/10 + MRR@10, paired mcnemar counts with the **exact**
two-sided binomial test, fusion lift, a derived query-style breakdown, and worked
discordant examples.

`style_proxy.py` owns the derived keyword/paraphrase rule. the rows carry no
style field, so the label is a **proxy derived from question text alone** — never
from gold, retrieval output, or any per-row search mode.

## query set

430 rows = 400 train + 30 eval, from
`examples/neon_rag_smoke/datasets/{train,eval}_large.jsonl` (materialized from
commit `fa724a2`). every row carries exactly one gold file; none were excluded.

## running

```sh
source ~/.config/neon-benchmax.env   # provides NEON_CORPUS_DSN_RO (read-only)
export CASTFORM_BASE_DOMAIN=castform.dev
uv run python examples/neon_retrieval_ab/run_ab.py --concurrency 4
uv run python examples/neon_retrieval_ab/analyze_ab.py
```

cost: one embedding pass over 430 short questions (7 batched calls) plus 1290
retrieval calls. ~93s wall at concurrency 4.

## outputs

- `results/raw_results.jsonl` — per query, per mode: chunk ids, per-rank files,
  deduped ranked files. auditable and re-analyzable without re-querying.
- `results/run_manifest.json` — settings, counts, failures, wall time.
- `results/summary.json` — machine-readable metrics.
- `results/RESULTS.md` — the decision report, including the go / no-go call.
