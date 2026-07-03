"""RAG search environment (written by `castform setup --template rag`).

Post-trains a model to answer questions by SEARCHING a corpus (BM25/lexical) and
citing its sources. The search tool, system prompt, and dataset wiring come from
`SearchEnv` (the convenience base); THIS file spells out the reward inline in
`compute_reward` — the reward is the whole training signal, so it's here to read
and edit, not buried in the library. The heavy pieces (the LLM judge, the citation
matcher) stay as named helpers imported from the lib so this file stays short.

The whole run is reproducible from this file: the reward is above, and the
`VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks bake in the rollout budgets so
`castform validate` / `castform launch` need no extra flags (a CLI flag still
overrides). Audit the reward on real transcripts before a serious launch:
`castform validate --reward-audit`.

Data: `train_dataset.jsonl` / `eval_dataset.jsonl` with `{question, answer,
reference_chunks}` rows — generate them from your corpus with
`castform data qa-gen --corpus-name <CORPUS_NAME> --fast`. Build the corpus first
with `castform corpus ingest <folder> --name <CORPUS_NAME>`.

Footgun: do NOT pass `benchmax` in `local_modules` at launch — it re-imports the
package by value and breaks `issubclass(env, BaseEnv)`. benchmax is already on the
trainer image; only your own local modules need bundling.
"""

from __future__ import annotations

import logging
from typing import Any

from benchmax import config
from benchmax.envs.postgres_search.search_env import (
    SearchEnv,
    extract_answer_block,
    judge_answer_quality,
    score_citations,
    score_search_efficiency,
)
from benchmax.envs.reward_helpers import (
    clip01,
    count_search_calls,
    extract_completion_text,
)
from benchmax.envs.types import Messages
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
# cap. Each search = one turn + one tool call; the final answer is one extra TURN.
# The rollout budget below (VALIDATE_CONFIG / LAUNCH_CONFIG) is sized off this.
MAX_SEARCH_CALLS = 6

# ── Reward weights ─────────────────────────────────────────────────────────
# All components are SUMMED into one scalar per rollout. Every secondary component
# is gated on correctness (see compute_reward), so brevity/citations can't trade
# off against being right. Scale these so correctness dominates.
W_CORRECTNESS = 1.0
W_CONCISENESS = 0.5
W_CITATION_RECALL = 0.5
W_CITATION_PRECISION = 0.5
W_SEARCH_EFFICIENCY = 0.1

REWARD_KEYS = (
    "answer_correctness",
    "conciseness",
    "citation_recall",
    "citation_precision",
    "search_efficiency",
)

logger = logging.getLogger(__name__)


class CustomSearchEnv(SearchEnv):
    # Extra pip deps the rollout sandbox needs — `validate`/`launch` read this and
    # install it (the sandbox bundles only run.py + benchmax). Empty for the default
    # Postgres corpus; when you swap `search=` to a provider client, list its SDK
    # here (e.g. ["chromadb>=1.0.0", "snowballstemmer>=2.2.0"]) — or pass
    # `--provider <name>` to validate/launch and skip the bookkeeping.
    PIP_DEPENDENCIES: list[str] = []

    # Rendered once at class-definition so the dataset/prompt preprocessors read
    # the resolved value via `cls` (keep MAX_SEARCH_CALLS in sync with __init__).
    # To change the prompt, override SYSTEM_PROMPT_TEMPLATE (see SearchEnv).
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

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """The reward — the whole training signal. Edit freely; audit with
        `castform validate --reward-audit` before launching.

        Every secondary component is multiplied by `correctness` (the 0/0.5/1.0
        judge score), so a wrong or missing answer earns no citation/brevity bonus
        and a partial answer earns partial bonuses. Return positive scores only.
        """
        zeros = {k: 0.0 for k in REWARD_KEYS}
        try:
            text = extract_completion_text(messages)
            if not text.strip():
                return zeros
            t = task or {}
            # Strict extraction: no committed <answer> → "" → scores 0 (the model's
            # reasoning is never scored as the answer).
            answer = extract_answer_block(text)
            reference_chunks = t.get("reference_chunks", [])

            # LLM judge (correctness + conciseness). The HTTP call lives in the lib
            # helper; the config comes from __init__ (JUDGE_MODEL / the platform LLM).
            correctness_raw, conciseness_raw = await judge_answer_quality(
                question=str(t.get("question") or t.get("prompt") or ""),
                ground_truth=str(t.get("ground_truth") or ""),
                response=answer,
                model=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_token_provider(),
                timeout=self._judge_timeout,
            )
            correctness = clip01(correctness_raw)  # the gate: 0 / 0.5 / 1.0

            # Citations: exact source-id match by default. For a corpus with duplicate
            # pages or bare-id citations, override _canonicalize_id (threaded below)
            # or pass a canonicalize= callable — see the design-environment skill.
            recall, precision = score_citations(
                answer, reference_chunks, canonicalize=self._canonicalize_id
            )
            calls = count_search_calls(messages)

            return {
                "answer_correctness": W_CORRECTNESS * correctness,
                "conciseness": W_CONCISENESS * clip01(conciseness_raw) * correctness,
                "citation_recall": W_CITATION_RECALL * recall * correctness,
                "citation_precision": W_CITATION_PRECISION * precision * correctness,
                "search_efficiency": score_search_efficiency(
                    calls=calls,
                    correctness=correctness_raw,
                    reference_chunk_count=len(reference_chunks),
                    max_search_calls=MAX_SEARCH_CALLS,
                    weight=W_SEARCH_EFFICIENCY,
                ),
            }
        except (KeyError, ValueError, TypeError, AttributeError):
            # A reward bug must not crash the rollout — score 0, but LOG it: a
            # silent all-zero reward is the hardest reward bug to diagnose.
            logger.exception("[CustomSearchEnv] compute_reward failed")
            return zeros


# ── Run config — validate/launch read these so the run reproduces from this file
#    alone (a CLI flag still overrides). See `castform validate/launch --help`.

# A search env needs a turn/tool budget above the 4/8 default, or the rollout is
# truncated below MAX_SEARCH_CALLS. N searches → N+1 turns (a final answer turn) and
# N tool calls.
VALIDATE_CONFIG = {
    "max_turns": MAX_SEARCH_CALLS + 1,
    "max_tool_calls": MAX_SEARCH_CALLS,
    "examples": 6,  # a few real rollouts make --reward-audit's per-component read sharper
}

# The trainer ignores an env's recommended_max_*, so bake the budget here. NOTE
# max_tool_calls is NOT a launch knob (stays 8) — keep MAX_SEARCH_CALLS <= 8. The
# accepted arg set is `castform launch --list-args`; an unknown key here is skipped
# with a warning.
LAUNCH_CONFIG = {
    "max_turns": MAX_SEARCH_CALLS + 1,
    # Total tokens across the WHOLE rollout (all turns). Search output is large — set
    # this generously or rollouts hit the cap, get truncated, and drop from the loss.
    "max_rollout_len": 16384,
    "num_epochs": 3,  # eval tends to peak before the overfit tail; keep epochs modest
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}
