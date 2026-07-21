# GitLab handbook BM25 reference run

This example reproduces the gold Castform RAG RL checkpoint over the GitLab
handbook using the first-party Postgres BM25 corpus backend.

The script:

1. checks out the pinned GitLab handbook commit;
2. optionally ingests the handbook with pinned chunking;
3. renders the BM25 search environment to `run.py`;
4. stages the public train/eval JSONL split from Hugging Face;
5. validates the environment;
6. optionally launches and monitors a Qwen3.5-4B training run.

Launching is opt-in with `--launch` because it spends GPU credits.

## Gold pins

- GitLab commit: `3078d0213524f8ca0c0e3a70680a21929a9f65ff`
- GitLab project: `gitlab-com/content-sites/handbook`
- Source subdirectory: `content/handbook`
- Reference corpus: `gitlab-handbook-bm25-3078d0213524-staging`
- Corpus ID: `ca8b6f24-5837-4528-93ca-3970e9bddf63`
- Chunking: min `1024`, max `2048`, overlap `128`
- Dataset: `wingedbreadsticks/gitlab-handbook-bm25-3078d0213524`
- Dataset tag: `gold-qwen35-4b-bm25-2026-07-08`
- Dataset commit: `efafe62fe5da43904e46f90c8031f95b3304bce6`

Public dataset base URL:

```text
https://huggingface.co/datasets/wingedbreadsticks/gitlab-handbook-bm25-3078d0213524/resolve/gold-qwen35-4b-bm25-2026-07-08/
```

## Validate against the existing staging corpus

From the benchmax repo:

```bash
uv sync --all-extras

CASTFORM_BASE_DOMAIN=castform.dev \
uv run python examples/gitlab_handbook_bm25/reproduce_gitlab_bm25.py \
  --staging \
  --skip-ingest
```

The script defaults to the pinned GitLab SHA, the reference staging corpus name,
and the public Hugging Face train/eval files.

## Re-ingest and validate

Use this when you want to recreate the corpus instead of reusing the reference
staging corpus:

```bash
CASTFORM_BASE_DOMAIN=castform.dev \
uv run python examples/gitlab_handbook_bm25/reproduce_gitlab_bm25.py \
  --staging \
  --corpus-name gitlab-handbook-bm25-3078d0213524-replay
```

## Launch

After validation looks sane, add `--launch`:

```bash
CASTFORM_BASE_DOMAIN=castform.dev \
uv run python examples/gitlab_handbook_bm25/reproduce_gitlab_bm25.py \
  --staging \
  --skip-ingest \
  --launch \
  --monitor-until-terminal
```

Reference completed run:

- Run ID: `c428b173-3fb6-4b86-83c1-291075fd9c5d`
- URL: `https://app.castform.dev/train/c428b173-3fb6-4b86-83c1-291075fd9c5d`
- Model: `Qwen/Qwen3.5-4B`
- Final eval reward: `0.8888635829048844` at step `34`
- Final train reward: `1.093700763983043` at step `35`
- Final train/eval truncation: `0`

## Files created by the script

The generated files stay inside this example directory and are ignored by git:

- `run.py`
- `train_dataset.jsonl`
- `eval_dataset.jsonl`
- `gitlab_bm25_manifest.json`
- `artifacts/`
- `work/`
