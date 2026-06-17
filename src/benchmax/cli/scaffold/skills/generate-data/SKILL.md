---
name: generate-data
description: Create the train/eval datasets for a castform run — generic prompt/ground_truth jsonl, or RAG QA pairs from a corpus (`corpus ingest` + `data qa-gen`). Use when building or uploading training data.
---

# Generate the data

This skill is the **data** step of the path every run follows:

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
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

`castform setup` ships **no** dataset — write `train_dataset.jsonl` /
`eval_dataset.jsonl` for your task. Keep it tiny at first (tens of rows) so you
iterate cheaply through `castform validate`, then grow.

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
the reward **varies** across rollouts — a green baseline needs both. A good mix is
easy rows plus genuinely hard ones the cheap model reliably misses.

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
castform corpus search my-corpus "a question about your docs" --top-k 3
```

Non-empty, sensibly-scored hits = retrieval works. Empty or an error = fix the
corpus name / ingest before reading anything into a green `validate`.

#### Choosing a data source

| Source | When | Setup |
|---|---|---|
| **Local folder → CGFT corpus** | docs on disk; simplest | `castform corpus ingest` — **no key** (your `castform login` session). The gated fast path. |
| **Remote provider** (turbopuffer / pinecone / chroma) | your corpus already lives in a vector DB | set the `DATA_*` env vars below, then `castform data qa-gen --provider <name>` reads the corpus directly; point `run.py`'s search client at the same provider (see **design-environment**). |

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
login` session. With the `DATA_*` vars set, generate QA pairs straight from the
provider corpus (no `corpus ingest` — it reads stored chunks directly; needs that
provider's extra, e.g. `pip install castform[turbopuffer]`):

```bash
castform data qa-gen --provider turbopuffer --fast   # → train_dataset.jsonl + eval_dataset.jsonl
```

> That `pip install castform[<provider>]` only sets up **your machine** for qa-gen. The
> validate/launch **sandbox** needs the same SDK independently — pass `--pip <provider>`
> to `castform validate` (see design-environment), or the env hollow-greens.

### Traces — build data from recorded agent traces

For **traces** (post-training on what your agent already did), pull and shape them
with `castform data traces` (Braintrust; needs the `[traces]` extra and a
`BT_API_KEY`):

```bash
export BT_API_KEY=...                         # Braintrust key
castform data traces --project my-agent       # → train_dataset.jsonl + eval_dataset.jsonl
```

It fetches the project's traces, detects the **system prompt + tools**, and writes
`{prompt_messages, ground_truth, init_rollout_args}` rows. Pass `--project-id` (or
`BT_PROJECT_ID`) instead of `--project` if you have the id, and `--limit N` to cap
the fetch. Then author the env's `dataset_preprocess` to match those rows — see
**design-environment**'s traces note.
