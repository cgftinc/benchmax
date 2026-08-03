# turbopuffer_rag

an end-to-end Benchmax retrieval-training example backed by TurboPuffer. the data
command chunks Markdown, writes the corpus to TurboPuffer, and generates grounded
train/eval questions. the environment supports lexical, vector, and hybrid
retrieval.

## example corpus

place the Markdown documents you want to search over in `documents/`.

if you want to use a dummy corpus, clone the public GitLab Handbook:

```bash
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git documents/gitlab-handbook
```

## build the corpus and dataset

configure TurboPuffer:

```bash
export TPUF_API_KEY="..."

uv run python main.py data --question-count 20
```

data generation uses Castform's configured LLM endpoint. set
`TURBOPUFFER_RAG_MODEL_BASE_URL` and `TURBOPUFFER_RAG_MODEL_API_KEY` to use another
OpenAI-compatible endpoint. `TURBOPUFFER_RAG_EMBEDDING_MODEL` and
`TURBOPUFFER_RAG_QA_MODEL` override the default models.

## validate and launch

```bash
uv run python main.py validate
uv run python main.py launch
```

hybrid search runs BM25 and ANN queries concurrently and combines their ranked lists with RRF. the environment receives `TPUF_API_KEY` explicitly so the same uploaded bundle works in hosted validation and training. runtime dependencies contain only `turbopuffer`; Castform, chunkers, and ingestion code are excluded.

## tests

```bash
uv run pytest examples/turbopuffer_rag/tests
```
