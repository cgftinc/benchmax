"""GitLab handbook RAG env, BM25/Postgres version.

This is the current/gateway arm of the GitLab handbook gateway A/B. Its behavior
is kept aligned with ``../pre_harbor/environment.py`` while adapting to the
current Benchmax environment-owned rollout contract.

Compatibility note: the historical trainer advertised four searches but its
``>= max_tool_calls`` boundary stopped before executing search four. This arm
therefore keeps the prompt-visible limit at four while enforcing three executed
tool calls in the current gateway runtime.

The search backend is the first-party Corpora API/Postgres BM25 path:
    PostgresSearch(CORPUS_NAME, base_url=config.platform_url())

The reward keeps the lessons from the longer Chroma/vector iteration:
- score only a committed <answer> block;
- gate correctness and citation bonuses on actually retrieving a gold source;
- keep retrieval_hit ungated so search exploration has dense signal;
- enforce the search cap in run_tool, not only in the prompt;
- display top results with enough full text for extraction while keeping later
  results as source-bearing snippets.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from benchmax.envs import BaseRollout, InjectedAuth, Messages
from benchmax.rewards import Rubric, clip01, evaluate_single_rubric, extract_completion_text
from castform import config
from castform.rag.corpus.postgres.search import PostgresSearch
from search_env import SearchEnv

logger = logging.getLogger(__name__)

CORPUS_NAME = "gitlab-handbook-bm25-3078d0213524-staging"

JUDGE_MODEL = "gpt-5.4-mini"
MAX_SEARCH_CALLS = 4
# Pre-Harbor's declared limit was 4, but its off-by-one check executed at most 3.
EFFECTIVE_MAX_TOOL_CALLS = 3

W_CORRECTNESS = 1.0
W_CONCISENESS = 0.2
W_CITATION_RECALL = 0.3
W_CITATION_GROUNDING = 0.2
W_RETRIEVAL_HIT = 0.3

FULL_CONTENT_RANKS = 4
FULL_CHUNK_CHARS = 1600
SNIPPET_CHARS = 180
TOOL_OUTPUT_CHARS = 7000

REWARD_KEYS = (
    "answer_correctness",
    "conciseness",
    "citation_recall",
    "citation_grounding",
    "retrieval_hit",
)

LAUNCH_CONFIG = {
    "name": f"gitlab-handbook-bm25-{CORPUS_NAME}",
    "model": "Qwen/Qwen3.5-4B",
    "max_context_tokens": 16384,
    "num_epochs": 5,
    "learning_rate": 4e-6,
    "eval_interval": 5,
    "save_interval": 5,
    "type": "simple",
}

VALIDATE_CONFIG = {
    "max_turns": MAX_SEARCH_CALLS + 1,
    "max_tool_calls": EFFECTIVE_MAX_TOOL_CALLS,
    "examples": 6,
}

_SOURCE_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer\s*>", re.IGNORECASE)

_CORRECTNESS_RUBRIC = Rubric(
    title="Answer correctness (strict, question-grounded)",
    description=(
        "Grade whether the Response states the specific facts the QUESTION asks "
        "for, using the Ground Truth as the authoritative reference. Credit "
        "paraphrases and equivalent specifics. Restating or rephrasing the "
        "question itself is not an answer. A fact counts only when the Response "
        "gives the requested value, name, step, decision, or other concrete "
        "answer. Merely naming the topic or saying where to look does not count. "
        "Ignore citations while grading the prose."
    ),
    polarity="positive",
    score_map={
        0: ("Wrong, missing, contradicted, vague, or only an echo of the question."),
        0.5: (
            "Specific but incomplete: at least one requested fact is correct, "
            "but another requested fact is missing or wrong."
        ),
        1: ("Complete: correctly answers everything the QUESTION asks, with no contradiction."),
    },
)


def _extract_answer(text: str) -> str:
    text = text or ""
    blocks = list(_ANSWER_BLOCK_RE.finditer(text))
    if blocks:
        return blocks[-1].group(1).strip()
    opens = list(_ANSWER_OPEN_RE.finditer(text))
    if not opens:
        return ""
    return text[opens[-1].end() :].strip()


def _canonical_source_id(source_id: str) -> str:
    """Normalize file-path source ids without collapsing distinct pages."""
    s = str(source_id or "").strip().lower().replace("\\", "/")
    s = re.sub(r"^[a-z]+://", "", s)
    s = re.sub(r"[?#].*$", "", s)
    s = re.sub(r"^\./+", "", s)
    path_match = re.search(r"(.+?\.mdx?)(?:$|[^a-z0-9_./-])", s)
    if path_match:
        s = path_match.group(1)
    for prefix in ("content/handbook/", "handbook/"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    s = re.sub(r"/index\.mdx?$", "", s)
    s = re.sub(r"\.mdx?$", "", s)
    return re.sub(r"\s+", " ", s).strip("/")


class HandbookBm25SearchEnv(SearchEnv):
    system_prompt = (
        SearchEnv.render_system_prompt(
            corpus_description=f"the GitLab handbook corpus '{CORPUS_NAME}'",
            max_search_calls=MAX_SEARCH_CALLS,
        )
        .replace(
            "Cite your sources inline using [Source: <source_id>] next to each claim.",
            "Cite each claim inline with [Source: <source>], where <source> is copied "
            "verbatim from the [source: ...] label of the search result you used.",
        )
        .replace(
            "1. If initial results do not contain the answer, re-query with broadened or "
            "rephrased language.",
            "1. If initial results do not contain the answer, search again with the "
            "handbook's likely page title, team name, tool name, policy name, or other "
            "corpus-specific terms. Do not answer from memory when the cited source is "
            "not on screen.",
        )
    )

    def __init__(self, **kwargs: Any) -> None:
        self._search_calls: dict[str, int] = {}
        super().__init__(
            search=PostgresSearch(CORPUS_NAME, base_url=config.platform_url()),
            judge_base_url=config.llm_url(),
            judge_model=JUDGE_MODEL,
            judge_auth=InjectedAuth("judge"),
            max_search_calls=MAX_SEARCH_CALLS,
            max_turns=MAX_SEARCH_CALLS + 1,
            w_correctness=W_CORRECTNESS,
            **kwargs,
        )
        # Current BaseEnv correctly interprets max_tool_calls=N as "execute N".
        # Override its runtime budget to reproduce the historical arm's
        # effective behavior without changing the shared four-search prompt.
        self.max_tool_calls = EFFECTIVE_MAX_TOOL_CALLS

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name == "search" and not str(tool_args.get("query") or "").strip():
            return "Error: search needs a non-empty query string."
        if tool_name == "search":
            used = self._search_calls.get(rollout_id, 0) + 1
            if used > MAX_SEARCH_CALLS:
                return (
                    "You have used all allowed searches. Give your final answer now "
                    "inside <answer></answer> tags, citing sources you already saw."
                )
            self._search_calls[rollout_id] = used
            result = await super().run_tool(rollout_id, tool_name, **tool_args)
            if used == MAX_SEARCH_CALLS and isinstance(result, str):
                result += (
                    "\n\nThis was your final search. Give your final answer now "
                    "inside <answer></answer> tags, citing sources above."
                )
            return result
        return await super().run_tool(rollout_id, tool_name, **tool_args)

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        trimmed: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            cap = FULL_CHUNK_CHARS if i < FULL_CONTENT_RANKS else SNIPPET_CHARS
            body = r.get("content") or ""
            if len(body) > cap:
                body = body[:cap].rstrip() + (
                    " ...[chunk truncated; search again to promote this result]"
                )
            md = {
                k: v
                for k, v in (r.get("metadata") or {}).items()
                if k not in ("file", "file_path", "chunk_hash", "char_count")
            }
            trimmed.append({**r, "content": body, "metadata": md})
        return self._truncate_tool_output(
            super()._format_results(trimmed), max_chars=TOOL_OUTPUT_CHARS
        )

    @staticmethod
    def _truncate_tool_output(
        text: str,
        max_chars: int = TOOL_OUTPUT_CHARS,
        suffix: str = "\n...[truncated due to character limit]",
    ) -> str:
        if len(text) <= max_chars:
            return text
        keep = max(0, max_chars - len(suffix))
        return text[:keep].rstrip() + suffix

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> dict[str, float]:
        zeros = {k: 0.0 for k in REWARD_KEYS}
        rollout_id = rollout.rollout_id
        messages = rollout.messages
        self._search_calls.pop(rollout_id, None)
        try:
            text = extract_completion_text(messages)
            if not text.strip():
                return zeros

            t = rollout.example_args
            answer = _extract_answer(text)
            prompt = str(t.get("question") or t.get("prompt") or "")
            ground_truth = str(t.get("ground_truth") or t.get("answer") or "")
            reference_chunks = t.get("reference_chunks", []) or []

            gold = [self._source_keys(self._ref_file(c)) for c in reference_chunks]
            gold = [keys for keys in gold if keys]
            retrieved = self._source_key_sets(self._tool_text(messages))
            cited = self._source_key_sets(answer)

            retrieval_hit_raw = self._match_frac(gold, retrieved)
            correctness_raw = await self._judge_correctness(prompt, ground_truth, answer)
            correctness = correctness_raw if retrieval_hit_raw > 0 else 0.0

            cited_gold = self._match_frac(gold, cited)
            cited_retrieved = self._match_frac(cited, retrieved)
            brevity = self._brevity(answer, ground_truth)

            return {
                "answer_correctness": W_CORRECTNESS * correctness,
                "conciseness": W_CONCISENESS * brevity * correctness,
                "citation_recall": W_CITATION_RECALL * cited_gold * correctness,
                "citation_grounding": W_CITATION_GROUNDING * cited_retrieved * correctness,
                "retrieval_hit": W_RETRIEVAL_HIT * retrieval_hit_raw,
            }
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.exception("[HandbookBm25SearchEnv] compute_reward failed: %s", exc)
            return zeros

    async def _judge_correctness(self, question: str, ground_truth: str, response: str) -> float:
        if not (response or "").strip():
            return 0.0
        last_err: object = None
        for attempt in range(3):
            try:
                res = await evaluate_single_rubric(
                    rubric=_CORRECTNESS_RUBRIC,
                    question=question,
                    ground_truth=ground_truth,
                    response=response,
                    judge=self._judge,
                )
                reasoning = str(res.reasoning or "")
                if reasoning.startswith("Error:"):
                    last_err = reasoning[:300]
                else:
                    return clip01(res.score)
            except Exception as exc:  # noqa: BLE001
                last_err = repr(exc)
            await asyncio.sleep(1.5 * (attempt + 1))
        logger.error("[HandbookBm25SearchEnv] judge failed after retries: %s", last_err)
        return 0.0

    @staticmethod
    def _brevity(answer: str, ground_truth: str) -> float:
        prose = _SOURCE_RE.sub("", answer or "").strip()
        n = len(prose)
        if n == 0:
            return 0.0
        target = max(300.0, 1.5 * len(ground_truth or ""))
        return 1.0 if n <= target else clip01(target / n)

    @staticmethod
    def _ref_file(chunk: Any) -> str:
        md = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        return str((md or {}).get("file") or (md or {}).get("file_path") or "")

    @staticmethod
    def _tool_text(messages: Messages) -> str:
        return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "tool")

    @staticmethod
    def _source_keys(source: str) -> set[str]:
        canon = _canonical_source_id(source)
        keys = {canon} if canon else set()
        if canon.endswith("/index"):
            keys.add(canon[: -len("/index")])
        return {k for k in keys if k}

    def _source_key_sets(self, text: str) -> list[set[str]]:
        return [self._source_keys(m.group(1)) for m in _SOURCE_RE.finditer(text or "")]

    @staticmethod
    def _match_frac(targets: list[set[str]], pool: list[set[str]]) -> float:
        if not targets:
            return 0.0
        pool_union: set[str] = set().union(*pool) if pool else set()
        return sum(1 for t in targets if t & pool_union) / len(targets)
