"""SearchEnv — multi-component reward search environment for RL training.

Provides 5 reward components:
1. **answer_correctness** — LLM judge scores factual accuracy (0, 0.5, 1.0)
2. **conciseness** — LLM judge scores brevity (gated on correctness)
3. **citation_recall** — fraction of reference sources cited (gated on correctness)
4. **citation_precision** — fraction of cited sources that are relevant (gated on correctness)
5. **search_efficiency** — shaped bonus based on search count vs. gold chunk count
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.reward_helpers import (
    clip01,
    count_search_calls,
    extract_completion_text,
    search_within_budget,
)
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.platform.credentials import (
    TokenProvider,
    as_token_provider,
    platform_bearer,
)
from benchmax.rag.corpus.search_client import SearchClient
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

    Public reward helper: a scaffold ``run.py`` imports this to score answers
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


# ----------------------------------------------------------------------------
# Reward helpers — the reward *arithmetic* lives in the env's compute_reward
# (and, for the scaffold, in run.py so it's visible/editable); these free
# functions are the reusable pieces it calls by name. The heavy plumbing (the
# HTTP judge, the citation matcher) stays here so run.py stays short.
# ----------------------------------------------------------------------------


async def judge_answer_quality(
    *,
    question: str,
    ground_truth: str,
    response: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 30.0,
    correctness_rubric: Rubric = CORRECTNESS_RUBRIC,
    conciseness_rubric: Rubric = CONCISENESS_RUBRIC,
) -> tuple[float, float]:
    """LLM judge → ``(correctness, conciseness)``, both in [0, 1].

    Empty response → ``(0.0, 0.0)``. The two rubric calls run concurrently. This
    is the heavy HTTP leg of the reward — a scaffold ``run.py`` calls it as a
    one-liner and does the weighting/gating arithmetic itself. Pass custom
    rubrics to change what "correct"/"concise" mean.
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
    """Normalize a citation/source id (default: whitespace-strip → exact match).

    Pass a custom callable to :func:`score_citations` for corpus-specific,
    robust matching (id-hash / title / path). The default is exact-path *by
    design* — the design-environment skill covers when to override it."""
    return str(source_id or "").strip()


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
    """Backend-agnostic search environment with multi-component rewards.

    Requires an LLM judge for correctness and conciseness scoring.

    Args:
        search: A :class:`SearchClient` instance (pickle-safe).
        judge_base_url: Base URL for the LLM judge API (required).
        judge_model: Model name for the LLM judge (required).
        judge_token_provider: Optional; resolves the judge bearer per call.
            Defaults to ``platform_bearer`` (the credential seam).
        judge_timeout: Timeout for judge API calls.
        w_correctness: Weight for correctness reward component.
        w_conciseness: Weight for conciseness reward component.
        w_citation_recall: Weight for citation recall component.
        w_citation_precision: Weight for citation precision component.
        w_search_efficiency: Weight for search efficiency reward component.
        max_search_calls: Hard search call budget (0 reward if exceeded).
    """

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
        judge_token_provider: str | TokenProvider | None = None,
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        w_conciseness: float = 0.5,
        w_citation_recall: float = 0.5,
        w_citation_precision: float = 0.5,
        w_search_efficiency: float = 0.1,
        max_search_calls: int = 10,
        **kwargs: Any,
    ) -> None:
        if not judge_base_url or not judge_model:
            raise ValueError(
                "SearchEnv requires judge_base_url and judge_model; both must be "
                "non-empty. The judge credential is resolved at call time via "
                "judge_token_provider (default: platform_bearer)."
            )

        self._search = search
        self._judge_base_url = judge_base_url
        self._judge_model = judge_model
        # Resolved per call (default: the platform credential seam). A customer
        # may inject their own provider; baking their own key stays supported
        # but discouraged — see docs/design/env-credential-model.md §7.1.
        self._judge_token_provider = as_token_provider(
            judge_token_provider, platform_bearer
        )
        self._judge_timeout = judge_timeout
        self._w_correctness = w_correctness
        self._w_conciseness = w_conciseness
        self._w_citation_recall = w_citation_recall
        self._w_citation_precision = w_citation_precision
        self._w_search_efficiency = w_search_efficiency
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

        search_tool = ToolDefinition(
            name="search",
            description="Search the corpus.",
            input_schema={
                "type": "object",
                "properties": search_props,
                "required": ["query"],
            },
        )
        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            "search": (search_tool, self._search_tool),
        }

        # `system_prompt` is a static class attribute (default ""). Subclasses
        # set it at class-definition — e.g.
        # `system_prompt = SearchEnv.render_system_prompt(...)` — so the
        # classmethod preprocessors read the resolved value via cls.

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[k][0] for k in sorted(self._tools)]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name not in self._tools:
            return f"Error: Unknown tool '{tool_name}'"
        _, tool_function = self._tools[tool_name]
        return await tool_function(**tool_args)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        question = example.get("question", "")
        return make_example(
            prompt_messages=[{"role": "user", "content": question}],
            task={
                "question": question,
                "ground_truth": example.get("answer"),
                "reference_chunks": example.get("reference_chunks", []),
            },
            system_prompt=cls.system_prompt,
        )

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute 5-component reward."""
        zeros = self._zero_rewards()
        try:
            text = extract_completion_text(messages)
            if not text.strip():
                return zeros

            t = task or {}
            answer = _extract_answer_block(text)
            prompt = str(t.get("question") or t.get("prompt") or "")
            gt_str = str(t.get("ground_truth") or "")
            reference_chunks = t.get("reference_chunks", [])
            reference_chunk_count = len(reference_chunks)

            logger.info(
                "[SearchEnv] Q: %s\n  GT: %s\n  A: %s",
                prompt[:200],
                gt_str[:200],
                answer[:200],
            )

            # 1. Correctness + Conciseness (concurrent judge calls)
            correctness_raw, conciseness_raw = await self._judge_answer_quality(
                question=prompt,
                ground_truth=gt_str,
                response=answer,
            )

            # Gate every secondary bonus on correctness (0 / 0.5 / 1.0): a wrong
            # or missing answer earns no citation/brevity reward, and a partial
            # answer earns only partial bonuses — so brevity/citations can't
            # trade off against being right. (search_efficiency already scales
            # this way.)
            correctness = clip01(correctness_raw)
            rewards: dict[str, float] = {
                "answer_correctness": self._w_correctness * correctness,
                "conciseness": self._w_conciseness
                * clip01(conciseness_raw)
                * correctness,
            }

            # 2. Citation recall / precision (gated on correctness)
            recall, precision = self._score_citations(answer, reference_chunks)
            rewards["citation_recall"] = self._w_citation_recall * recall * correctness
            rewards["citation_precision"] = (
                self._w_citation_precision * precision * correctness
            )

            # 3. Search efficiency (shaped by search count vs. gold chunk baseline)
            calls = count_search_calls(messages)
            rewards["search_efficiency"] = self._score_search_efficiency(
                calls=calls,
                correctness_raw=correctness_raw,
                reference_chunk_count=reference_chunk_count,
            )

            logger.info("[SearchEnv] rewards=%s", rewards)
            return rewards

        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.exception("[SearchEnv] compute_reward failed: %s", exc)
            return zeros

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
        try:
            results = self._search.search(query=query, mode=effective_mode, top_k=limit)
            return self._format_results(results)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

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
        :func:`judge_answer_quality` helper with this env's judge config."""
        return await judge_answer_quality(
            question=question,
            ground_truth=ground_truth,
            response=response,
            model=self._judge_model,
            base_url=self._judge_base_url,
            api_key=self._judge_token_provider(),
            timeout=self._judge_timeout,
        )

    # ------------------------------------------------------------------
    # Citation scoring
    # ------------------------------------------------------------------

    def _score_search_efficiency(
        self,
        *,
        calls: int,
        correctness_raw: float,
        reference_chunk_count: int,
    ) -> float:
        """Reward correct answers that don't search past the gold-chunk baseline —
        delegates to the free :func:`score_search_efficiency` helper."""
        return score_search_efficiency(
            calls=calls,
            correctness=correctness_raw,
            reference_chunk_count=reference_chunk_count,
            max_search_calls=self._max_search_calls,
            weight=self._w_search_efficiency,
        )

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
        """Normalize a source ID. Override for corpus-specific rules."""
        return canonicalize_source_id(source_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zero_rewards(self) -> dict[str, float]:
        return {
            "answer_correctness": 0.0,
            "conciseness": 0.0,
            "citation_recall": 0.0,
            "citation_precision": 0.0,
            "search_efficiency": 0.0,
        }
