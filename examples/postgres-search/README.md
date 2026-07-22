# postgres-search

A retrieval-augmented QA environment: the model answers questions by calling
a `search` tool backed by a provisioned corpora-service corpus (Postgres +
pg_search), with an LLM judge scoring correctness plus retrieval-hit,
citation-precision, and length components.

Purpose: the library base for RAG training environments. It ships no corpus —
subclass `SearchEnv` with your corpus's system prompt, provision the corpus,
and supply question datasets.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/postgres-search
uv run pytest tests            # the env's contract tests (no corpus needed)
uv run python main.py          # states exactly what a real run requires
```
