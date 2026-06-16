---
name: generate-data
description: Create the train/eval datasets for a castform run (prompt/ground_truth jsonl) and upload local files. Use when building or uploading training data.
---

# Generate the data

A run needs `train_dataset.jsonl` and `eval_dataset.jsonl` — one JSON object per
line. Each row needs at least a `prompt`; most rewards also want a `ground_truth`
(or whatever fields your `compute_reward` reads off `task`).

```jsonl
{"prompt": "Translate 'hello' to French.", "ground_truth": "bonjour"}
{"prompt": "Capital of Japan?", "ground_truth": "Tokyo"}
```

- Keep train and eval **disjoint**. Eval is what you watch for generalisation.
- The row is passed to `compute_reward` as `task`, so put any per-example scoring
  data (answers, rubrics, configs) right in the row.
- Start small (tens of rows) to iterate cheaply through `castform validate`, then
  grow.

## Uploading a local file

To push a local dataset to castform storage (e.g. to reference a large file):

```bash
castform data upload train_dataset.jsonl
# → blobPath: datasets/cli/train_dataset.jsonl
```

`castform launch` uploads your datasets automatically, so manual `data upload` is
only for sharing/inspecting a file out of band.

## First-party generators (RAG / traces)

For **search/RAG** (generate QA pairs from a corpus) or **traces** (build data
from collected agent traces), castform generates the data for you via the
`benchmax.rag.qa_generation` / `benchmax.traces` library pipelines. These need the
corpus/traces and the relevant extras — see `castform.com/docs/rag` and
`.../traces`. (Dedicated `castform data qa-gen` / `traces` verbs are coming.)
