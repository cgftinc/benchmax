# chroma_rag

an end-to-end Benchmax retrieval-training example for Chroma Cloud or a
self-hosted Chroma HTTP server. the data command chunks Markdown, writes the
corpus to Chroma, and generates grounded train/eval questions. the environment
supports vector retrieval.

## example corpus

place the Markdown documents you want to search over in `documents/`.

if you want to use a dummy corpus, clone the public GitLab Handbook:

```bash
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git documents/gitlab-handbook
```

## configure and build data

for Chroma Cloud:

```bash
export CHROMA_TENANT="..."
export CHROMA_DATABASE="..."
export CHROMA_API_KEY="..."
```

for a self-hosted server, use `CHROMA_HOST`, with optional `CHROMA_PORT` and
`CHROMA_SSL=true`. data generation uses Castform's configured LLM endpoint:

```bash
uv run python main.py data --question-count 20
```

set `CHROMA_RAG_MODEL_BASE_URL` and `CHROMA_RAG_MODEL_API_KEY` to use another
OpenAI-compatible endpoint. the data command generates grounded train/eval rows from the documents.

## validate and launch

```bash
uv run python main.py validate
uv run python main.py launch
```

this example exposes vector search only. lexical and hybrid search require
Chroma's Search API and a configured sparse index, which this example does not
create. `available_modes` is declared locally and does not require a connection.

the environment receives `CHROMA_API_KEY` explicitly for Chroma Cloud, allowing
the same uploaded bundle to run in hosted validation and training. the runtime
dependency list declares only `chromadb`.

## tests

```bash
uv run pytest examples/chroma_rag/tests
```
