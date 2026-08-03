# neon_rag

a retrieval-training example backed by Neon Postgres. the environment gives the model hybrid vector and full-text search over Markdown documents, then rewards grounded, correctly cited answers.

## example task

each trial asks the model a question that should be answered using the indexed documents:

```text
question: how are failed ingestion jobs recovered?

agent: searches the Neon corpus and answers using the retrieved passages

judge: scores correctness, retrieval quality, citations, and concision
```

## workflow

first, complete the one-time [Neon setup](setup.md). after Neon is configured, running the command below will:

1. transform and chunk the documents in `documents/`.
2. embed and upload the chunks to Neon.
3. generate 20 Q&A pairs by default, split into 16 training and 4 evaluation examples.
4. prepare and upload the Neon search environment.
5. validate the environment.
6. launch training (after a Y/N confirmation check in the terminal).

## example corpus

place the Markdown documents you want to search over in `documents/`.

if you want to use a dummy corpus, clone the public GitLab Handbook:

```bash
git clone --depth 1 https://gitlab.com/gitlab-com/content-sites/handbook.git documents/gitlab-handbook
```

## run the whole flow e2e

after completing the Neon setup and adding documents, run:

```bash
cd examples/neon_rag
source ./.env.neon

uv run python main.py launch \
  --question-count 20 \
  --neon-data-preparation-database-url "$NEON_DATA_PREPARATION_DATABASE_URL" \
  --neon-search-database-url "$NEON_SEARCH_DATABASE_URL"

# --force  # add this flag to rebuild the corpus and overwrite generated Q&A files.
# --yes    # add this flag to skip the launch confirmation.
```

the command performs all the steps listed above.

## environment

`NeonRagEnv` extends Benchmax's shared `RagEnv` and supplies an example-local Neon search implementation:

```python
class NeonRagEnv(RagEnv):
    def __init__(
        self,
        *,
        search_database_url,
        judge_base_url,
        embedding_base_url,
    ):
        super().__init__(
            search=NeonSearch(
                CORPUS_NAME,
                database_url=search_database_url,
                embed_fn=OpenAIEmbedder(...),
            ),
            judge_base_url=judge_base_url,
            judge_model="gpt-5.4-mini",
            ...
        )
```

> [!NOTE]
> this example is not production-ready. for production use, we recommend having the search tool interact with a dedicated search API rather than connecting directly to a toy Neon database with a Postgres database URL.

## run each stage separately

the e2e command above is recommended for the first run. the following commands are useful when iterating on individual parts of the example.

### prepare the corpus and Q&A data

add or replace the Markdown files in `documents/`, then run:

```bash
source ./.env.neon

uv run python main.py data \
  --question-count 20 \
  --neon-data-preparation-database-url "$NEON_DATA_PREPARATION_DATABASE_URL"

# --force  # add this flag to rebuild the corpus and overwrite generated Q&A files.
```

this command chunks and embeds the Markdown documents, uploads them to Neon, and generates the requested Q&A examples with an 80/20 train/eval split.

### update only the indexed documents

use this when you want to update the searchable Neon corpus without regenerating the Q&A dataset:

```bash
source ./.env.neon

uv run python main.py ingest \
  --neon-data-preparation-database-url "$NEON_DATA_PREPARATION_DATABASE_URL"
```

### validate the environment

use this after preparing the corpus and Q&A data to validate without launching training:

```bash
source ./.env.neon

uv run python main.py validate \
  --neon-search-database-url "$NEON_SEARCH_DATABASE_URL"
```

validation uploads the environment and dataset, then runs sample rollouts locally and in a hosted sandbox.

## model configuration

data generation uses the model endpoint from your active Castform profile. to use another OpenAI-compatible endpoint, export:

```bash
export NEON_RAG_MODEL_BASE_URL="https://..."
export NEON_RAG_MODEL_API_KEY="..."
```

the default embedding model produces 3,072-dimensional vectors. a replacement embedding model must match the vector dimensions configured during Neon setup.

## tests

```bash
uv run pytest examples/neon_rag/tests packages/benchmax/tests/unit/rag
```
