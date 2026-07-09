---
name: generate-data
description: Create the train/eval datasets for a castform run — generic prompt/ground_truth jsonl, or RAG QA pairs from a corpus (`corpus ingest` + `data qa-gen`). Use when building or uploading training data.
---

# Generate the data

This is stage 2 of the castform loop:

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

## Fast path (to a green baseline)

A run needs `train_dataset.jsonl` and `eval_dataset.jsonl`, one JSON object per
line. Each row needs at least a `prompt`; most rewards also need `ground_truth` or
the fields your `compute_reward` reads from `task`.

```jsonl
{"prompt": "Translate 'hello' to French.", "ground_truth": "bonjour"}
{"prompt": "Capital of Japan?", "ground_truth": "Tokyo"}
```

`castform setup` ships tiny seed datasets. Replace the seed rows with task rows,
keep the first pass small, then grow after validate shows signal.

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

When the data is in place, invoke **verify-environment** before running
`castform validate`.

## Going deeper

### Difficulty-filter for real training signal

A dataset the model already gets right (or wrong) *every* time gives **no
gradient** — the reward never varies, so there's nothing to learn. Pick rows by
what the model *currently* gets wrong: roll candidates through the cheap model and
keep the misses. From the project dir (so your `main.py` reward is applied):

```bash
# candidates.jsonl = prompt + ground_truth rows you're considering
castform validate --train candidates.jsonl --examples 20 --json \
  | jq -c '.examples[] | select(.ok and .rewards.correct == 0) | .index'
```

Those indices are the rows worth keeping — the model misses them, so there's
signal. (Swap `correct` for your reward component.) Mix in some it gets right so
the reward **varies** across rollouts — a green baseline needs both. A good mix is
easy rows plus genuinely hard ones the cheap model reliably misses.

<!-- rag:start -->
### RAG — generate QA pairs from a corpus

Use the user's real documents. Never generate fake/demo source docs; for RAG,
"synthetic data" means QA pairs from the real corpus.

First run `castform corpus list`. If any corpora exist, show their names and offer
reuse (`castform data qa-gen --corpus-name <n>`) before asking about a new source.
For a new corpus, ask for either a local folder (`castform corpus ingest <folder>`)
or a vector-DB provider (`castform data qa-gen --provider <name>`). If they have no
documents yet, scaffold the env and pause at the ingest step.

Fast path for a local docs folder, no provider key (needs `castform[rag]`):

```bash
castform corpus ingest <your-docs-folder> --name my-corpus   # the user's real docs → BM25 corpus
castform data qa-gen --corpus-name my-corpus --fast          # → train_dataset.jsonl + eval_dataset.jsonl
castform setup --template rag --force                        # SearchEnv main.py (edit the CORPUS_NAME constant)
castform validate
```

qa-gen rows are `{question, answer, reference_chunks}` — exactly what the rag
`SearchEnv` reads (no remap). `--fast` skips the LLM-judge filters for a quick
small set; drop it for the full filtered pipeline. (`--force` on `setup` is only
needed to replace an existing `main.py`.) For **small docs** that error with "No
eligible chunks", lower the eligibility floor:
`castform data qa-gen … --min-chunk-chars 120` (default 400).

`reference_chunks[].metadata` is corpus-dependent. A Notion/handbook export may
carry page-id hashes and title paths; a flat docs folder may only have filenames;
provider corpora may expose different source fields. Do not hard-code citation
reward logic to one exact `metadata.file` string without checking the generated rows.
For RAG rewards, design source matching around the identifiers your corpus actually
emits: id/hash, title/path, and file path when available.

Treat generated metadata as hints, not universal labels. In particular,
`linking_hints.confidence == 0` can be a default sentinel from the metadata linker,
not proof the row is mislabeled; spot-check rows before using it as a prune rule.
Likewise, fields such as `query_style_observed` or `retrieval_difficulty` are useful
for sampling and analysis only after you verify their semantics on your corpus.

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

If the request declares the source, use it. Otherwise confirm before ingesting.

| Source | When | Setup |
|---|---|---|
| **Local folder → CGFT corpus** | docs on disk, not uploaded yet | `castform corpus ingest <folder> --name <n>` — **no key** (your `castform login` session). The gated fast path. |
| **Existing Castform corpus** | already `corpus ingest`-ed before (or a corpus already on Castform) | **skip ingest** — `castform corpus list` to find the name, then `castform data qa-gen --corpus-name <n>`. The rag scaffold's `PostgresSearch(CORPUS_NAME)` reads it directly; just set `CORPUS_NAME`. |
| **Remote provider** (turbopuffer / pinecone / chroma) | your corpus already lives in an external vector DB | set the `DATA_*` env vars below, then `castform data qa-gen --provider <name>` reads the corpus directly; point `main.py`'s search client at the same provider (see **design-environment**). |

#### Provider credentials — env vars, NEVER in `main.py`

Keys live in **environment variables**, read at build/bundle time. **Export them in
your shell** (`export DATA_api_key=…`); never write a key into `main.py` (it gets bundled
and uploaded) and never paste one into the chat with your agent. Have the user set them
and **verify presence** (`echo ${DATA_api_key:+set}`) rather than handling the value
yourself. The secret is `DATA_api_key` for all three providers; resource identifiers
also take `DATA_*` overrides:

| Provider | Secret (required) | Resource fields |
|---|---|---|
| **turbopuffer** | `DATA_api_key` | `DATA_namespace` (req), `DATA_region` |
| **pinecone** | `DATA_api_key` | `DATA_index_name` (req), `DATA_index_host`, `DATA_namespace` |
| **chroma** | `DATA_api_key` | `DATA_collection_name` (req), `DATA_tenant`, `DATA_database` |

> **pinecone default namespace:** pinecone serves the default namespace as
> `__default__`, not `""`. If your index uses it, set `DATA_namespace=__default__`
> explicitly — an unset namespace queries the wrong (empty) one → hollow green.

The local-folder path needs **no key** — it authenticates with your `castform
login` session. With the `DATA_*` vars set, generate QA pairs straight from the
provider corpus (no `corpus ingest` — it reads stored chunks directly; needs that
provider's extra, e.g. `pip install castform[turbopuffer]`):

```bash
castform data qa-gen --provider turbopuffer --fast   # → train_dataset.jsonl + eval_dataset.jsonl
```

> That `pip install castform[<provider>]` only sets up **your machine** for qa-gen. The
> validate/launch **sandbox** needs the same SDK independently — pass `--provider <name>`
> to `castform validate`/`launch` (it injects the SDK, incl. chroma's `snowballstemmer`;
> see design-environment), or the env hollow-greens. No package names to memorize.

> **Confirm provider retrieval** with `castform validate --full-messages` (read the
> search tool's output — real chunks vs a swallowed `Error:`). `castform corpus search`
> only targets CGFT corpora, not a provider namespace.
<!-- rag:end -->

### Traces — build data from recorded agent traces

For **traces** (post-training on what your agent already did), pull and shape them
with `castform data traces` (Braintrust; needs the `[traces]` extra and a
`BT_API_KEY`):

```bash
export BT_API_KEY=...                              # Braintrust key (never paste in chat)
castform data traces --project my-agent --dry-run  # detect-only: print prompt + tools, write nothing
castform data traces --project my-agent            # → train_dataset.jsonl + eval_dataset.jsonl
```

It fetches the project's traces, detects the **system prompt + tools**, and writes
`{prompt_messages, ground_truth, init_rollout_args}` rows. **Confirm the detection
first:** `--dry-run` prints the full detected system prompt + tools and writes nothing —
check they match the agent before generating. Pass `--project-id` (or `BT_PROJECT_ID`)
instead of `--project` if you have the id, and `--limit N` to cap the fetch. Then author
the env's `dataset_preprocess` to match those rows — see **design-environment**'s traces
note.

> Generated trace rows skew **text-reply heavy**, with action/tool rows landing late in
> file order. Since `validate` rolls out the first N rows, **interleave an action row near
> the front** before a small `--examples 2` validate, or the variance check only sees
> reply rows (see verify-environment).
