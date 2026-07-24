# gitlab-handbook-bm25-neon

An all-capability **golden retrieval eval** for the Neon lakebase RAG provider,
built on the public GitLab handbook. It ports the corpus-build half of the older
`gitlab_handbook_bm25` loader onto the `castform.*` layout and the Neon provider,
then authors a frozen golden set that exercises every retrieval mode.

The corpus is reused, but the dataset is new and different from the lexical-only
BM25 golden: it covers **lexical, vector, filter, and hybrid** retrieval.

## What it contains

| file | role |
|---|---|
| `handbook_corpus.py` | pinned sparse-checkout + deterministic re-chunk with filterable metadata (`handbook_section`, `path_depth`) |
| `build_corpus.py` | one-shot ingest of the full corpus into Neon at 3072-dim |
| `build_golden.py` | authors the golden set: qa-gen (lexical + vector) via the DI seam + hand-curated filter/hybrid |
| `curated_rows.py` | the FILTER + HYBRID rows (Path Y), authored from chunk content |
| `datasets/gitlab_handbook_neon_golden.jsonl` | the **frozen** golden set (committed, never regenerated in CI) |
| `datasets/provenance.json` | provenance manifest (corpus SHA, chunker, embedder, models, counts) |
| `datasets/corpus_build.json` | the ingest provenance (chunk count, neon version) |

The live validity gate lives with the provider tests:
`packages/castform/tests/integration/rag/corpus/neon/test_golden_eval_live.py`.

## Design: non-circularity

The gold is authored **blind** — before any Neon retrieval runs — so it never
encodes what the provider happens to return:

* **exact chunk-hash gold ids.** qa-gen already carries the source chunk hash in
  `reference_chunks[].id`; that hash is the Neon row id by construction. Curated
  rows resolve their gold/decoy ids from chunk content. No id is ever read back
  from a retrieval result.
* **retrieval-decoupled qa-gen.** the only Neon-retrieval-coupled filter
  (`retrieval_too_easy_llm`) is dropped; the remaining filters are judge-LLM /
  heuristic only.
* **vector rows are not keyword-solvable.** the vector pass paraphrases (natural /
  expert query style, obfuscation on) and a local lexical-hardness filter drops
  questions that overlap their gold chunk too much. The live gate then *confirms*
  this with a Neon BM25 ablation.
* **decoys.** the curated filter/hybrid rows carry cross-section chunks that share
  the query token; the section filter must exclude them.

## Rebuilding (not a CI step)

Rebuilding spends embedding + generation credits and is a manual, one-shot
operation. The frozen JSONL is the source of truth.

```bash
set -a; source ~/.config/neon-benchmax.env; set +a   # NEON_CORPUS_DSN_*, PLATFORM_API_KEY

# 1. ingest the full corpus into Neon (embeds ~31,665 chunks at 3072-dim)
uv run --extra neon python examples/gitlab_handbook_bm25_neon/build_corpus.py \
    --work-dir /tmp/handbook_repo \
    --provenance-out examples/gitlab_handbook_bm25_neon/datasets/corpus_build.json

# 2. author the golden set (qa-gen passes + curated rows + provenance)
uv run --extra neon python examples/gitlab_handbook_bm25_neon/build_golden.py \
    --work-dir /tmp/handbook_repo \
    --out-dir examples/gitlab_handbook_bm25_neon/datasets \
    --build-timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 3. run the live validity + EXPLAIN-at-scale gate
uv run --extra neon python -m pytest -m integration \
    packages/castform/tests/integration/rag/corpus/neon/test_golden_eval_live.py
```

The chunk-level determinism (identical hashes on re-chunk) is covered without a DB
by `tests/test_chunking.py`.
