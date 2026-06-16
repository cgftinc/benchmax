---
name: generate-data
description: Create the train/eval datasets for a castform run (prompt/ground_truth jsonl) and upload local files. Use when building or uploading training data.
---

# Generate the data

## Fast path (to a green baseline)

A run needs `train_dataset.jsonl` and `eval_dataset.jsonl` — one JSON object per
line. Each row needs at least a `prompt`; most rewards also want a `ground_truth`
(or whatever fields your `compute_reward` reads off `task`).

```jsonl
{"prompt": "Translate 'hello' to French.", "ground_truth": "bonjour"}
{"prompt": "Capital of Japan?", "ground_truth": "Tokyo"}
```

`castform setup` already shipped a small starter dataset — **replace its rows**
with your task's. Keep it tiny at first (tens of rows) so you iterate cheaply
through `castform validate`, then grow.

- Keep train and eval **disjoint**. Eval is what you watch for generalisation.
- The row is passed to `compute_reward` as `task`, so put any per-example scoring
  data (answers, rubrics, configs) right in the row.
- Store large integers as **strings**, not JSON numbers. A numeric value above the
  JS safe-integer limit (~`2^53`) fails the rollout (a Python↔TypeScript
  number-divergence guard), so write `"ground_truth": "24024198…"`, not a bare int.

`castform launch` uploads your datasets automatically; manual upload is only for
sharing/inspecting a file out of band:

```bash
castform data upload train_dataset.jsonl   # → blobPath: datasets/cli/train_dataset.jsonl
```

When the data's in place, go to **verify-environment** and run `castform validate`.

## Going deeper

### Difficulty-filter for real training signal

A dataset the model already gets right (or wrong) *every* time gives **no
gradient** — the reward never varies, so there's nothing to learn. Pick rows by
what the model *currently* gets wrong: roll candidates through the cheap model and
keep the misses. From the project dir (so your `run.py` reward is applied):

```bash
# candidates.jsonl = prompt + ground_truth rows you're considering
castform validate --train candidates.jsonl --examples 20 --json \
  | jq -c '.examples[] | select(.ok and .rewards.correct == 0) | .index'
```

Those indices are the rows worth keeping — the model misses them, so there's
signal. (Swap `correct` for your reward component.) Mix in some it gets right so
the reward **varies** across rollouts — a green baseline needs both. The shipped
starter does exactly this: easy rows plus large-multiplication rows the cheap
model reliably misses.

### First-party generators (RAG / traces) — not wired as CLI verbs yet

For **search/RAG** (generate QA pairs from a corpus) or **traces** (build data
from collected agent traces), castform generates the data via the
`benchmax.rag.qa_generation` / `benchmax.traces` library pipelines — call those
**directly from Python**, not via the CLI. They need the corpus/traces and the
relevant extras — see `castform.com/docs/rag` and `.../traces`.

Dedicated `castform data qa-gen` / `traces` verbs (and corpus-connect) are
**coming, not in today's CLI**. Only the generic `prompt`/`ground_truth` flow
above completes end-to-end on the CLI today.
