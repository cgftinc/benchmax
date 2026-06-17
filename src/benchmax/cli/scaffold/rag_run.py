"""RAG search environment (written by `castform setup --template rag`).

Post-trains a model to answer questions by SEARCHING a corpus (BM25/lexical) and
citing its sources. The search tool and the 5-component reward — answer
correctness + conciseness (LLM judge), citation recall + precision, and search
efficiency — come from `SearchEnv`; this subclass only supplies the corpus and
the judge. Edit the three constants below for your task.

Data: `train_dataset.jsonl` / `eval_dataset.jsonl` with `{question, answer,
reference_chunks}` rows — generate them from your corpus with
`castform data qa-gen --corpus-name <CORPUS_NAME> --fast`. `SearchEnv` reads those
fields directly (no remap). Build the corpus first with
`castform corpus ingest <folder> --name <CORPUS_NAME>`.

Footgun: do NOT pass `benchmax` in `local_modules` at launch — it re-imports the
package by value and breaks `issubclass(env, BaseEnv)`. benchmax is already on the
trainer image; only your own local modules need bundling.
"""

from __future__ import annotations

from benchmax import config
from benchmax.envs.postgres_search.search_env import SearchEnv
from benchmax.rag.corpus.postgres.search import PostgresSearch

# The corpus to search. It must already exist on the Corpora backend — create it
# with `castform corpus ingest <folder> --name <CORPUS_NAME>`. Resolved by name at
# rollout time. (An existing name resolves without prompting; a non-existent name
# can block on an interactive corpus-cap prompt — ingest it first.)
CORPUS_NAME = "my-corpus"

# Judge model for the correctness/conciseness reward components (LLM, no GPU).
JUDGE_MODEL = "gpt-5.4-mini"

# Search-call budget per rollout; the system prompt advertises this same number.
MAX_SEARCH_CALLS = 10


class CustomSearchEnv(SearchEnv):
    # Rendered once at class-definition so the dataset/prompt preprocessors read
    # the resolved value via `cls` (keep MAX_SEARCH_CALLS in sync with __init__).
    system_prompt = SearchEnv.render_system_prompt(
        corpus_description=f"the '{CORPUS_NAME}' corpus",
        max_search_calls=MAX_SEARCH_CALLS,
    )

    def __init__(self, **kwargs):
        super().__init__(
            # PostgresSearch is pickle-safe; the bearer is resolved per request,
            # nothing credential-shaped is frozen into the bundled env.
            search=PostgresSearch(CORPUS_NAME, base_url=config.platform_url()),
            judge_base_url=config.llm_url(),
            judge_model=JUDGE_MODEL,
            max_search_calls=MAX_SEARCH_CALLS,
            **kwargs,
        )
