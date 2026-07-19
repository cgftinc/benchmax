"""SearchEnv — multi-component reward search environment for RL training.

The default reward is the AUDITED 4-component shape (one LLM judge call, the rest
deterministic):
1. **answer_correctness** — LLM judge scores factual accuracy (the GATE)
2. **retrieval_hit** — fraction of gold sources cited in the final ``<answer>``
   block (UNGATED: citing gold is rewarded even on a wrong answer, so the model
   keeps learning to search). An answer-side proxy for retrieval — raw tool
   traffic is never inspected.
3. **citation_precision** — fraction of cited sources that are gold (gated on
   correctness)
4. **answer_length** — deterministic brevity term (gated on correctness; replaces
   the LLM conciseness judge)

The old components stay available as opt-in helpers (:data:`CONCISENESS_RUBRIC`,
:func:`judge_answer_quality`, :func:`score_search_efficiency`) for subclasses that
override ``compute_reward``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Example,
    InjectedAuth,
    JsonRow,
    JsonlDataset,
    Messages,
    ModelAuth,
    Tool,
    canonical_example_id,
)
from benchmax.envs.reward_helpers import (
    clip01,
    extract_completion_text,
    search_within_budget,
)
from castform.rag.corpus.search_client import SearchClient
from benchmax.rubrics.rubric import Rubric, evaluate_single_rubric

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)

_ANSWER_BLOCK_RE = re.compile(
    r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)
_ANSWER_OPEN_RE = re.compile(r"<answer\s*>", re.IGNORECASE)


def extract_answer_block(text: str) -> str:
    """Extract the model's committed answer from <answer> tags.

    Strict + last-committed: returns "" when the model never opens an
    <answer> tag, so its reasoning is never scored as the answer. Prefers the
    LAST properly-closed <answer>...</answer> block — a self-correction wins,
    and a stray literal "<answer>" later in the prose can't hijack scoring.
    Only when no block is closed does it forgive a missing close tag and take
    everything after the final opener (a truncated final answer).

    Public reward helper: a scaffold ``main.py`` imports this to score answers
    inline. NOTE this is the strict extractor — do NOT use
    ``reward_helpers.extract_answer_block``, which falls back to the full
    completion when there's no tag (scores reasoning as the answer).
    """
    text = text or ""
    blocks = list(_ANSWER_BLOCK_RE.finditer(text))
    if blocks:
        return blocks[-1].group(1).strip()
    opens = list(_ANSWER_OPEN_RE.finditer(text))
    if not opens:
        return ""
    return text[opens[-1].end() :].strip()


# Underscore alias kept for internal call sites / tests that import the old name.
_extract_answer_block = extract_answer_block

# Match Python-style `{name}` placeholders with word-char names only —
# leaves JSON-like literals (e.g. `{"answer": "X"}`) and unknown keys
# untouched, so a user-edited SYSTEM_PROMPT_TEMPLATE that contains JSON
# examples doesn't blow up at env construction time.
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _render_template(template: str, **vars: Any) -> str:
    """Substitute `{name}` placeholders, leaving unknown matches verbatim.

    Safer than ``str.format`` for templates that may legitimately contain
    raw `{` / `}` characters (JSON examples, escape sequences). Only word-
    character placeholders are considered; ``{"answer": "X"}`` passes
    through unchanged.
    """
    return _TEMPLATE_PLACEHOLDER_RE.sub(
        lambda m: str(vars[m.group(1)]) if m.group(1) in vars else m.group(0),
        template,
    )


CORRECTNESS_RUBRIC = Rubric(
    title="Answer correctness",
    description=(
        "Response correctly answers the question and is "
        "factually consistent with the reference answer."
    ),
    type="positive",
    score_map={
        0: "Provided answer is missing or incorrect.",
        1: "Fully correct and factually consistent.",
    },
)

# Opt-in rubric — NOT part of the default reward (brevity is the deterministic
# answer_length term). Pass it to `judge_answer_quality` from a custom reward.
CONCISENESS_RUBRIC = Rubric(
    title="Answer conciseness",
    description=(
        "Response is concise and avoids unnecessary verbosity "
        "while still directly answering the question."
    ),
    type="positive",
)

MAX_TOOL_OUTPUT_CHARS = 10000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated due to character limit]"
SEARCH_EFFICIENCY_DECAY_RATE = 0.2

# Deterministic brevity cap: an answer at/above this many chars earns no length
# bonus; shorter (still-correct) answers earn more. Dense signal on every correct
# rollout, no second LLM call.
ANSWER_LENGTH_CAP = 600


# ----------------------------------------------------------------------------
# Reward helpers — the reward *arithmetic* lives in the env's compute_reward
# (and, for the scaffold, in main.py so it's visible/editable); these free
# functions are the reusable pieces it calls by name. The heavy plumbing (the
# HTTP judge, the citation matcher) stays here so main.py stays short.
# ----------------------------------------------------------------------------


async def judge_answer_quality(
    *,
    question: str,
    ground_truth: str,
    response: str,
    model: str,
    base_url: str,
    api_key: str,
    auth: ModelAuth | None = None,
    timeout: float = 30.0,
    correctness_rubric: Rubric = CORRECTNESS_RUBRIC,
    conciseness_rubric: Rubric = CONCISENESS_RUBRIC,
) -> tuple[float, float]:
    """LLM judge → ``(correctness, conciseness)``, both in [0, 1].

    Opt-in helper — NOT called by the default reward, which makes ONE
    ``evaluate_single_rubric`` call (correctness only) and scores brevity
    deterministically. Use this from a custom ``compute_reward`` when you want
    a judged conciseness component too.

    Empty response → ``(0.0, 0.0)``. The two rubric calls run concurrently.
    Pass custom rubrics to change what "correct"/"concise" mean.
    """
    if not response.strip():
        return (0.0, 0.0)
    try:
        correctness_task = evaluate_single_rubric(
            rubric=correctness_rubric,
            question=question,
            ground_truth=ground_truth,
            response=response,
            model_name=model,
            base_url=base_url,
            api_key=api_key,
            auth=auth,
            timeout=timeout,
        )
        conciseness_task = evaluate_single_rubric(
            rubric=conciseness_rubric,
            question=question,
            ground_truth=ground_truth,
            response=response,
            model_name=model,
            base_url=base_url,
            api_key=api_key,
            auth=auth,
            timeout=timeout,
        )
        correctness_result, conciseness_result = await asyncio.gather(
            correctness_task, conciseness_task
        )
        return (
            clip01(correctness_result.get("score", 0.0)),
            clip01(conciseness_result.get("score", 0.0)),
        )
    except Exception:
        return (0.0, 0.0)


def canonicalize_source_id(source_id: str) -> str:
    """Normalize a citation/source id by whitespace-strip → exact match.

    This is the strict matcher and the default for the free citation helpers
    (:func:`parse_citations` / :func:`extract_reference_ids` /
    :func:`score_citations`). :class:`SearchEnv` itself defaults to the loose
    :func:`canonicalize_source_id_loose`; pass ``canonicalize=`` (or override
    ``_canonicalize_id``) to choose per corpus."""
    return str(source_id or "").strip()


def canonicalize_source_id_loose(source_id: str) -> str:
    """Match citations by id-hash OR title-path: lowercase, drop any directory
    prefix and file extension, so ``docs/Geography.md``, ``geography.md`` and a
    bare ``geography`` canonicalize alike (dup-heavy Notion/GitLab exports).
    Apply symmetrically to the cited id and the gold ``metadata.file``.

    The :class:`SearchEnv` default canonicalizer."""
    s = str(source_id or "").strip().lower().rsplit("/", 1)[-1]
    return s.rsplit(".", 1)[0] if "." in s else s


def parse_citations(
    text: str, *, canonicalize: Callable[[str], str] = canonicalize_source_id
) -> set[str]:
    """Parse ``[Source: <id>]`` citations from the model's answer."""
    ids: set[str] = set()
    for match in _CITATION_RE.finditer(text or ""):
        cid = canonicalize(match.group(1).strip())
        if cid:
            ids.add(cid)
    return ids


def extract_reference_ids(
    reference_chunks: list[dict[str, Any]],
    *,
    canonicalize: Callable[[str], str] = canonicalize_source_id,
) -> set[str]:
    """Document-level source ids from the gold reference chunks (``file`` /
    ``file_path`` metadata)."""
    ids: set[str] = set()
    for chunk in reference_chunks:
        if not isinstance(chunk, dict):
            continue
        md = chunk.get("metadata", {})
        if not isinstance(md, dict):
            continue
        file_id = str(md.get("file") or md.get("file_path") or "").strip()
        if file_id:
            ids.add(canonicalize(file_id))
    return ids


def score_citations(
    answer_text: str,
    reference_chunks: list[dict[str, Any]],
    *,
    canonicalize: Callable[[str], str] = canonicalize_source_id,
) -> tuple[float, float]:
    """``(recall, precision)`` of the answer's citations vs the gold chunks, by
    exact source-id match. No gold → ``(0, 0)``. Pass ``canonicalize=`` to make
    matching corpus-robust."""
    ref_ids = extract_reference_ids(reference_chunks, canonicalize=canonicalize)
    cited_ids = parse_citations(answer_text, canonicalize=canonicalize)
    if not ref_ids:
        return 0.0, 0.0
    overlap = cited_ids & ref_ids
    recall = len(overlap) / len(ref_ids)
    precision = len(overlap) / len(cited_ids) if cited_ids else 0.0
    return recall, precision


def score_search_efficiency(
    *,
    calls: int,
    correctness: float,
    reference_chunk_count: int,
    max_search_calls: int,
    weight: float,
    decay_rate: float = SEARCH_EFFICIENCY_DECAY_RATE,
) -> float:
    """Bonus for a CORRECT answer that doesn't over-search past ~``ref_chunks + 2``.

    Opt-in helper — NOT part of the default reward (a "fewer searches" bonus can
    fight multi-hop exploration; the hard ``max_search_calls`` cap covers safety).
    Call it from a custom ``compute_reward`` to re-enable search shaping.

    ``0`` when the answer is incorrect (``correctness <= 0``) or the run is over
    the hard ``max_search_calls`` budget. ``correctness`` is the raw judge score
    (it both gates and scales)."""
    if correctness <= 0:
        return 0.0
    if not search_within_budget(calls, max_search_calls):
        return 0.0
    baseline_calls = reference_chunk_count + 2
    excess_calls = max(0, calls - baseline_calls)
    decay = math.exp(-decay_rate * excess_calls)
    return weight * correctness * decay


class SearchEnv(BaseEnv):
    """Backend-agnostic search environment with the audited 4-component reward.

    ``answer_correctness`` (one LLM judge call) is the GATE: every component
    EXCEPT ``retrieval_hit`` is × correctness, so brevity/precision can't be
    earned on a wrong answer. ``retrieval_hit`` is UNGATED — citing a gold
    source is rewarded even when the answer is wrong. ``answer_length`` is a
    deterministic brevity term (no second judge call).

    Requires an LLM judge for correctness scoring.

    Args:
        search: A :class:`SearchClient` instance (pickle-safe).
        judge_base_url: Base URL for the LLM judge API (required).
        judge_model: Model name for the LLM judge (required).
        judge_auth: Explicit, serializable judge auth declaration. Defaults to
            ``InjectedAuth("judge")`` for runtime-provided call-time credentials.
        judge_timeout: Timeout for judge API calls.
        w_correctness: Weight for the correctness component (the gate).
        w_retrieval_hit: Weight for the UNGATED retrieval_hit component (recall
            of gold sources among the final-answer citations — a proxy for
            retrieval; tool traffic is not inspected).
        w_citation_precision: Weight for citation precision (gated).
        w_length: Weight for the deterministic brevity component (gated).
        max_search_calls: Hard search call budget (advertised in the prompt).
    """

    # The gate component — inspect it in the script's validation reward output.
    # secondary components for redundancy against this key.
    PRIMARY_REWARD_KEY = "answer_correctness"
    system_prompt: str | None = None
    _ZERO_REWARDS = {
        "answer_correctness": 0.0,
        "retrieval_hit": 0.0,
        "citation_precision": 0.0,
        "answer_length": 0.0,
    }

    SYSTEM_PROMPT_TEMPLATE = """\
Answer the given question by searching over {corpus_description}.

First, reason about the question inside <think>...</think>. You may want to rephrase the
question or break it down into sub-questions.

Call the search tool to retrieve relevant results. After receiving information, reason
about it inside <think>...</think> before either:
(1) issuing a new search query
(2) providing the final answer

Each reasoning step should be grounded in retrieved information.

You can search up to {max_search_calls} times. Break the question down across multiple
search queries to gather comprehensive information.

Recommended approach:
1. If initial results do not contain the answer, re-query with broadened or rephrased language.
2. Reference retrieved chunks to formulate more specific follow-up queries
(e.g. using keywords in chunk content or using metadata).

When you have gathered enough information, return your final answer inside <answer>...</answer>
tags. Cite your sources inline using [Source: <source_id>] next to each claim.
"""

    @classmethod
    def render_system_prompt(
        cls, *, corpus_description: str, max_search_calls: int
    ) -> str:
        """Render :attr:`SYSTEM_PROMPT_TEMPLATE` into a system-prompt string.

        Assign the result to a subclass's ``system_prompt`` class attribute
        (rendered once at class-definition, not in ``__init__``), and pass the
        same ``max_search_calls`` to ``__init__`` so the prompt's stated budget
        matches the enforced one::

            class MyEnv(SearchEnv):
                system_prompt = SearchEnv.render_system_prompt(
                    corpus_description="support docs", max_search_calls=10
                )
        """
        return _render_template(
            cls.SYSTEM_PROMPT_TEMPLATE,
            corpus_description=corpus_description,
            max_search_calls=max_search_calls,
        )

    def __init__(
        self,
        search: SearchClient,
        *,
        judge_base_url: str,
        judge_model: str,
        judge_auth: ModelAuth = InjectedAuth("judge"),
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        w_retrieval_hit: float = 0.3,
        w_citation_precision: float = 0.3,
        w_length: float = 0.2,
        max_search_calls: int = 10,
        max_turns: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if not judge_base_url or not judge_model:
            raise ValueError(
                "SearchEnv requires judge_base_url and judge_model; both must be "
                "non-empty. Judge authentication is declared separately with "
                "judge_auth."
            )
        if (
            isinstance(max_search_calls, bool)
            or not isinstance(max_search_calls, int)
            or max_search_calls < 1
        ):
            raise ValueError("max_search_calls must be a positive integer")
        super().__init__(
            max_turns=max_search_calls + 1 if max_turns is None else max_turns,
            max_tool_calls=max_search_calls,
        )
        self._system_prompt = (
            self.system_prompt if system_prompt is None else system_prompt
        )

        self._search = search
        self._judge_base_url = judge_base_url
        self._judge_model = judge_model
        self._judge_auth = judge_auth
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
        self._w_retrieval_hit = w_retrieval_hit
        self._w_citation_precision = w_citation_precision
        self._w_length = w_length
        self._max_search_calls = max_search_calls

        # Determine default search mode.
        modes = sorted(search.available_modes)
        if "hybrid" in modes:
            self._default_mode = "hybrid"
        elif "lexical" in modes:
            self._default_mode = "lexical"
        elif modes:
            self._default_mode = modes[0]
        else:
            self._default_mode = "lexical"

        # Build tool schema.
        search_props: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of results to return (default 10).",
            },
        }
        if len(modes) > 1:
            search_props["mode"] = {
                "type": "string",
                "enum": modes,
                "description": (
                    f"Search mode. Available: {modes}. Default: {self._default_mode}."
                ),
            }

        search_tool: Tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the corpus.",
                "parameters": {
                    "type": "object",
                    "properties": search_props,
                    "required": ["query"],
                },
                "strict": False,
            },
        }
        self._tools: dict[str, tuple[Tool, Callable]] = {
            "search": (search_tool, self._search_tool),
        }

        # `system_prompt` is a static class attribute (default ""). Subclasses
        # set it at class-definition — e.g.
        # `system_prompt = SearchEnv.render_system_prompt(...)` — so the
        # classmethod preprocessors read the resolved value via cls.

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    async def create_dataset(
        self, split: DatasetSplit, base_dir: Path
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            base_dir / f"{split}.jsonl",
            row_to_example=self._example_from_row,
        )

    async def list_tools(self) -> list[Tool]:
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'"
        _, tool_function = self._tools[tool_name]
        return await tool_function(**tool_args)

    def _example_from_row(self, row: JsonRow) -> Example[JsonRow]:
        question = row.get("question", "")
        if not isinstance(question, str):
            raise TypeError("SearchEnv dataset 'question' must be a string")
        messages: Messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": question})
        payload: JsonRow = {
            "prompt_messages": messages,
            "question": question,
            "ground_truth": row.get("answer"),
            "reference_chunks": row.get("reference_chunks", []),
        }
        return Example(
            id=canonical_example_id(payload),
            payload=payload,
        )

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> dict[str, float]:
        """The audited 4-component reward.

        ``answer_correctness`` (from the judge rubric) is the GATE: every
        component EXCEPT ``retrieval_hit`` is × correctness, so brevity/precision
        can't be earned on a wrong answer. ``retrieval_hit`` is UNGATED — citing
        a gold source is rewarded even when the answer is wrong.
        """
        # No committed answer is a valid terminal attempt with zero reward.
        answer = _extract_answer_block(extract_completion_text(rollout.messages))
        if not answer.strip():
            return dict(self._ZERO_REWARDS)

        t = rollout.example_args
        prompt = str(t.get("question") or t.get("prompt") or "")
        gt_str = str(t.get("ground_truth") or "")
        reference_chunks = t.get("reference_chunks", [])

        logger.info(
            "[SearchEnv] Q: %s\n  GT: %s\n  A: %s",
            prompt[:200],
            gt_str[:200],
            answer[:200],
        )

        # A judge/verifier failure is infrastructure failure, not evidence that
        # the model earned a zero. Let it propagate to the rollout executor.
        result = await evaluate_single_rubric(
            rubric=CORRECTNESS_RUBRIC,
            question=prompt,
            ground_truth=gt_str,
            response=answer,
            model_name=self._judge_model,
            base_url=self._judge_base_url,
            auth=self._judge_auth,
            timeout=self._judge_timeout,
        )
        correctness = clip01(result.get("score", 0.0))

        recall, precision = self._score_citations(answer, reference_chunks)
        length_score = clip01(1.0 - len(answer) / ANSWER_LENGTH_CAP)
        rewards: dict[str, float] = {
            "answer_correctness": self._w_correctness * correctness,
            "retrieval_hit": self._w_retrieval_hit * recall,
            "citation_precision": self._w_citation_precision * precision * correctness,
            "answer_length": self._w_length * length_score * correctness,
        }
        logger.info("[SearchEnv] rewards=%s", rewards)
        return rewards

    # ------------------------------------------------------------------
    # Search tool
    # ------------------------------------------------------------------

    async def _search_tool(
        self,
        query: str,
        mode: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        """Execute search via the SearchClient."""
        if not query:
            return "Error: Missing required parameter: 'query'"

        effective_mode = mode or self._default_mode
        results = await asyncio.to_thread(
            self._search.search,
            query=query,
            mode=effective_mode,
            top_k=limit,
        )
        return self._format_results(results)

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results with source labels and metadata."""
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "")
            score = r.get("score", 0.0)
            metadata = r.get("metadata", {})
            content = r.get("content", "")

            header = f"{i}."
            if source:
                header += f" — [source: {source}]"
            if score:
                header += f" (score: {score:.2f})"

            parts = [header]
            if metadata:
                display_md = {
                    k: v
                    for k, v in metadata.items()
                    if k not in ("content", "_local_hash", "chunk_hash", "char_count")
                    and not k.startswith("_")
                    and v is not None
                    and v != ""
                }
                if display_md:
                    parts.append(f"   Metadata: {display_md}")
            parts.append(f"   Content: {content}")
            lines.append("\n".join(parts))

        output = "\n".join(lines)
        return self._truncate_tool_output(output)

    @staticmethod
    def _truncate_tool_output(
        text: str,
        max_chars: int = MAX_TOOL_OUTPUT_CHARS,
        suffix: str = TOOL_OUTPUT_TRUNCATION_SUFFIX,
    ) -> str:
        if len(text) <= max_chars:
            return text
        keep = max(0, max_chars - len(suffix))
        return text[:keep].rstrip() + suffix

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    async def _judge_answer_quality(
        self,
        question: str,
        ground_truth: str,
        response: str,
    ) -> tuple[float, float]:
        """Evaluate correctness + conciseness — delegates to the free
        :func:`judge_answer_quality` helper with this env's judge config.

        Opt-in: the default reward makes ONE correctness rubric call instead;
        call this from a custom ``compute_reward`` for a judged conciseness."""
        return await judge_answer_quality(
            question=question,
            ground_truth=ground_truth,
            response=response,
            model=self._judge_model,
            base_url=self._judge_base_url,
            api_key="",
            auth=self._judge_auth,
            timeout=self._judge_timeout,
        )

    # ------------------------------------------------------------------
    # Citation scoring
    # ------------------------------------------------------------------

    def _score_citations(
        self,
        answer_text: str,
        reference_chunks: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """Citation (recall, precision) via the free :func:`score_citations`
        helper, honoring a subclass's ``_canonicalize_id`` override."""
        return score_citations(
            answer_text, reference_chunks, canonicalize=self._canonicalize_id
        )

    def _extract_reference_ids(
        self, reference_chunks: list[dict[str, Any]]
    ) -> set[str]:
        """Document-level source IDs from reference chunks (uses ``_canonicalize_id``,
        which subclasses may override for corpus-specific extraction)."""
        return extract_reference_ids(
            reference_chunks, canonicalize=self._canonicalize_id
        )

    def _parse_citations(self, text: str) -> set[str]:
        """Parse ``[Source: <id>]`` citations, honoring ``_canonicalize_id``."""
        return parse_citations(text, canonicalize=self._canonicalize_id)

    def _canonicalize_id(self, source_id: str) -> str:
        """Normalize a source ID (default: the loose id-hash OR title-path
        matcher, :func:`canonicalize_source_id_loose`). Override for
        corpus-specific rules — e.g. return :func:`canonicalize_source_id`
        for strict exact-path matching."""
        return canonicalize_source_id_loose(source_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
