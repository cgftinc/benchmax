# pinecone_rag

an end-to-end Benchmax retrieval-training example backed by a Pinecone dense
index. the data command chunks Markdown, writes embeddings and metadata to
Pinecone, and generates grounded train/eval questions. the environment supports
vector retrieval.

## example corpus

place the Markdown documents you want to search over in `documents/`.

if you want to use a dummy corpus, clone the public GitLab Handbook:

```bash
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git documents/gitlab-handbook
```

## prepare the index and data

create a cosine index named `benchmax-rag` with 3,072 dimensions for
`text-embedding-3-large`, then set its data-plane host and credentials:

```bash
export PINECONE_API_KEY="..."
export PINECONE_INDEX_HOST="https://your-index-host.svc...pinecone.io"

uv run python main.py data --question-count 20
```

data generation uses Castform's configured LLM endpoint. set
`PINECONE_RAG_MODEL_BASE_URL` and `PINECONE_RAG_MODEL_API_KEY` to use another
OpenAI-compatible endpoint.

using the explicit index host avoids a control-plane name lookup in every
restored training worker. the shared Castform driver chunks the documents and generates grounded train/eval rows; this example owns only Pinecone ingestion.

## validate and launch

```bash
uv run python main.py validate
uv run python main.py launch
```

the vector-only runtime adapter queries with `include_metadata=True` and
`include_values=False`, because rollouts need source text and metadata but not
stored vectors. the environment receives its API key explicitly so the same
uploaded bundle works in hosted validation and training. the bundle dependency
list contains `pinecone`, not Castform or its RAG data tooling.

## tests

```bash
uv run pytest examples/pinecone_rag/tests
```
