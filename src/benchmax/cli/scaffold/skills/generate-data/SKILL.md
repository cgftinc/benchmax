---
name: generate-data
description: Create the train/eval datasets for a castform run — generic prompt/ground_truth jsonl, or RAG QA pairs from a corpus (`corpus ingest` + `data qa-gen`). Use when building or uploading training data.
---

# Generate the data

This skill is the **data** step of the path every run follows:

```bash
castform setup        # 1. scaffold a working env + starter data
castform data …       # 2. data — keep the starter, upload your own, or generate (rag)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

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

### RAG — generate QA pairs from a corpus (first-party CLI verbs)

For **search/RAG** (post-training a model to search a corpus and cite sources),
the whole data path is CLI verbs. **Fast path — a local doc folder, no provider
key** (needs the `[rag]` extra: `pip install castform[rag]`):

```bash
castform corpus ingest ./docs --name my-corpus       # chunk + upload → BM25 corpus
castform data qa-gen --corpus-name my-corpus --fast  # → train_dataset.jsonl + eval_dataset.jsonl
castform setup --template rag --force                # SearchEnv run.py (edit the CORPUS_NAME constant)
castform validate
```

qa-gen rows are `{question, answer, reference_chunks}` — exactly what the rag
`SearchEnv` reads (no remap). `--fast` skips the LLM-judge filters for a quick
small set; drop it for the full filtered pipeline. (`--force` on `setup` is only
needed to replace an existing `run.py`.)

> **`validate` green ≠ working retrieval.** The search tool swallows errors into a
> string, so a rag env can validate green against an empty/unreachable corpus.
> Confirm the search tool returns real chunks (not an `Error:` / `No results`
> string) before trusting the baseline.

Confirm retrieval directly — this hits the corpus the same way the rollout does
(resolve by name), so real hits here mean the env can actually search:

```bash
python -c "
from benchmax import config
from benchmax.rag.corpus.postgres.search import PostgresSearch
hits = PostgresSearch('my-corpus', base_url=config.platform_url()).search('a question about your docs', top_k=3)
print(f'{len(hits)} hits'); [print(round(h['score'],2), h['source'], h['content'][:80]) for h in hits]
"
```

Non-empty, sensibly-scored hits = retrieval works. Empty or an exception = fix the
corpus name / ingest before reading anything into a green `validate`.

#### Choosing a data source

| Source | When | Setup |
|---|---|---|
| **Local folder → CGFT corpus** | docs on disk; simplest | `castform corpus ingest` — **no key** (your `castform login` session). The gated fast path. |
| **Remote provider** (turbopuffer / pinecone / chroma) | your corpus already lives in a vector DB | set the `DATA_*` env vars below, point `run.py`'s search client at the provider. *Documented; not gated by this CLI yet — needs an account + a provider chunk source for qa-gen.* |

#### Provider credentials — env vars, NEVER in `run.py`

Keys live in **environment variables**, read at build/bundle time. Never write a
key into `run.py` — it gets bundled and uploaded. The secret is `DATA_api_key` for
all three providers; resource identifiers also take `DATA_*` overrides:

| Provider | Secret (required) | Resource fields |
|---|---|---|
| **turbopuffer** | `DATA_api_key` | `DATA_namespace` (req), `DATA_region` |
| **pinecone** | `DATA_api_key` | `DATA_index_name` (req), `DATA_index_host`, `DATA_namespace` |
| **chroma** | `DATA_api_key` | `DATA_collection_name` (req), `DATA_tenant`, `DATA_database` |

The local-folder path needs **no key** — it authenticates with your `castform
login` session.

### Traces — not in today's CLI

For **traces** (build data from collected agent traces), call the
`benchmax.traces` library pipeline directly from Python for now — a `castform data
traces` verb is coming. See `castform.com/docs/traces`.
