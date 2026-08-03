---
name: generate-data
description: Prepare, clean, generate, or reference the data used by a Castform environment from project-owned Python scripts.
---

# Generate or reference data

Data preparation is a Python workflow. Keep it in `main.py`'s `generate_data`
stage or a nearby project script so its inputs, transformations and outputs are
reviewable. There is no separate data or corpus orchestration CLI.

## BaseEnv JSONL path

The generated BaseEnv seed reads `train.jsonl` and `eval.jsonl`. Each line is a
JSON object containing the fields the environment's row converter and reward
actually use:

```jsonl
{"prompt": "What is 2 + 2?", "ground_truth": "4"}
```

- Keep train and eval disjoint.
- Build stable example identity from semantic payload content, not row position or
  machine-local paths.
- Start small enough to inspect manually, then expand after validation shows a
  meaningful signal.
- Keep large integer identifiers as strings when data will cross JSON/JavaScript
  boundaries.
- Record provenance and make regeneration idempotent; never overwrite curated data
  without an explicit force flag.

`upload_assets` accepts optional train and eval rows. The launch script
decides what it uploads: omit a split when the environment resolves it at runtime,
and do not upload unrelated preparation artifacts. `None` means “do not upload”;
an empty list deliberately uploads an empty JSONL.

<!-- rag:start -->
## Hosted corpus and RAG

Install `castform[rag]` in the project and use public modules under
`castform.rag` from the data stage. Before implementing the workflow, inspect the
matching maintained example:

- `neon_rag`: https://github.com/castform-ai/benchmax/tree/main/examples/neon_rag
- `turbopuffer_rag`: https://github.com/castform-ai/benchmax/tree/main/examples/turbopuffer_rag
- `chroma_rag`: https://github.com/castform-ai/benchmax/tree/main/examples/chroma_rag
- `pinecone_rag`: https://github.com/castform-ai/benchmax/tree/main/examples/pinecone_rag

Use its `README.md`, `main.py`, `data.py`, `environment.py`, and `search.py` as the
reference for the provider. Typical data code composes:

- `castform.rag.chunkers` to turn source files into chunks;
- `castform.rag.corpus.postgres.client.CorpusClient` to create/find a corpus and
  upload chunks;
- `castform.rag.qa_generation` to build grounded QA rows;
- the provider's example-local search adapter for runtime reads.

Read the concrete class signatures before wiring them; these are library
components, not one magical pipeline command. Persist generated rows as ordinary
project data and test at least one known retrieval query before validation.

Replace the generic seed environment and rows with the selected RAG example's
structure. Confirm that each row contains `question`, `answer`, and
`reference_chunks`, and that every reference chunk carries the source metadata
expected by the citation reward.
<!-- rag:end -->

## Harbor-managed datasets

A Harbor dataset may be a local directory, Harbor package, registry reference or
Git repository resolved by `HarborEnv` during runtime. Do not duplicate that data
into JSONL merely to match the BaseEnv example. Only add an upload step when the
chosen Harbor workflow genuinely needs a folder or artifact uploaded.

## Traces

Install `castform[traces]`, normalize provider traces with the adapter for that
provider, and pass them through `castform.traces.TracesPipeline`. Keep the resulting
train/eval split and detected prompt/tool assumptions visible in the project.
Inspect for secrets, relayed tool output, duplicates and trivial examples before
using traces as training data.

## Verification

Before handing off to **verify-environment**:

1. validate the output schema with the environment's row converter;
2. check stable IDs and train/eval overlap;
3. inspect representative easy, hard and malformed rows;
4. confirm any corpus or Git reference is reachable in the rollout runtime;
5. run the deterministic reward against known answers where possible.
