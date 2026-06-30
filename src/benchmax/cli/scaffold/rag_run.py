"""RAG search environment (written by `castform setup --template rag`).

Post-trains a model to answer questions by SEARCHING a corpus (BM25/lexical) and
citing its sources. The search tool and stock multi-component reward come from
`SearchEnv`; this subclass supplies the corpus, judge, and rollout budget. Edit
the constants below for your task, and audit the reward on real transcripts before
a serious launch.

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
# Keep it <= 8 unless `castform launch --list-args` shows a higher launch tool-call
# cap. The rollout budget is separate and defaults lower (max_turns=4 /
# max_tool_calls=8), and the trainer ignores this value — so to actually use all N
# searches, set the rollout budget explicitly. Each search = one turn + one tool
# call; the final answer is inline (one extra TURN, not a tool call). For N=6:
#   castform validate --max-turns 7 --max-tool-calls 6   # both settable on validate
#   castform launch   --set max_turns=7                  # launch knob; see --list-args
# Tool output counts toward `max_rollout_len` across all turns, so trim result text
# or raise the launch budget if you increase this.
MAX_SEARCH_CALLS = 6


class CustomSearchEnv(SearchEnv):
    # Extra pip deps the rollout sandbox needs — `validate`/`launch` read this and
    # install it (the sandbox bundles only run.py + benchmax). Empty for the default
    # Postgres corpus; when you swap `search=` to a provider client, list its SDK
    # here (e.g. ["chromadb>=1.0.0", "snowballstemmer>=2.2.0"]) — or pass
    # `--provider <name>` to validate/launch and skip the bookkeeping.
    PIP_DEPENDENCIES: list[str] = []

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
