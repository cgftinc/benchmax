"""SearchEnv — multi-component reward search environment for RL training.

Provides 5 reward components:
1. **answer_correctness** — LLM judge scores factual accuracy (0, 0.5, 1.0)
2. **conciseness** — LLM judge scores brevity (gated on correctness)
3. **citation_recall** — fraction of reference sources cited
4. **citation_precision** — fraction of cited sources that are relevant
5. **search_efficiency** — shaped bonus based on search count vs. gold chunk count
"""

from __future__ import annotations

import asyncio
import math
import re
import traceback
from collections.abc import Callable
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.tracking import log_env
from benchmax.envs.types import StandardizedExample, ToolDefinition

from benchmax.rag.corpus.search_client import SearchClient
from benchmax.envs.postgres_search.reward_helpers import (
    clip01,
    count_search_calls,
    extract_answer_block,
    extract_completion_text,
    search_within_budget,
)
from benchmax.rubrics.rubric import Rubric, evaluate_single_rubric

_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)

_CORRECTNESS_RUBRIC = Rubric(
    title="Answer correctness",
    description=(
        "Response correctly answers the question and is "
        "factually consistent with the reference answer."
    ),
    type="positive",
    score_map={
        0: "Provided answer is missing or incorrect.",
        0.5: "Partially correct — captures some facts but missing key details.",
        1: "Fully correct and factually consistent.",
    },
)

_CONCISENESS_RUBRIC = Rubric(
    title="Answer conciseness",
    description=(
        "Response is concise and avoids unnecessary verbosity "
        "while still directly answering the question."
    ),
    type="positive",
)

SYSTEM_PROMPT_TEMPLATE = """\
Answer the given question by using a search engine over {corpus_description}.

You will first reason about the question inside <think> and </think>. For instance you may want
to rephrase the question or break down the question into multiple sub-questions that you will search for.

You must call the search engine with <tool_call> and it will return the top searched results.
After receiving the information, you must reason about it inside <think> and </think> before either
(1) issuing a new query with <tool_call>
(2) providing the final answer inside <answer> and </answer> tags.

Each of your reasoning steps should be grounded in the retrieved information.

You can search up to {max_search_calls} times. Try to break down the question for each search query \
and gather comprehensive information.

Recommended approach:
1. If initial results do not contain the answer, try to re-query with broadened or rephrased language.
2. Reference retrieved chunks to formulate more specific follow-up queries \
(e.g. using keywords in chunk content or using metadata)

If you have gathered enough information to answer the question,
return your final answer inside <answer>...</answer> and cite supporting sources as [Source: <source_id>].\
"""

MAX_TOOL_OUTPUT_CHARS = 10000
TOOL_OUTPUT_TRUNCATION_SUFFIX = "\n...[truncated due to character limit]"
SEARCH_EFFICIENCY_DECAY_RATE = 0.2


class SearchEnv(BaseEnv):
    """Backend-agnostic search environment with multi-component rewards.

    Requires an LLM judge for correctness and conciseness scoring.

    Args:
        search: A :class:`SearchClient` instance (pickle-safe).
        judge_base_url: Base URL for the LLM judge API (required).
        judge_api_key: API key for the LLM judge (required).
        judge_model: Model name for the LLM judge (required).
        corpus_description: Description injected into system prompt.
        judge_timeout: Timeout for judge API calls.
        w_correctness: Weight for correctness reward component.
        w_conciseness: Weight for conciseness reward component.
        w_citation_recall: Weight for citation recall component.
        w_citation_precision: Weight for citation precision component.
        w_search_efficiency: Weight for search efficiency reward component.
        max_search_calls: Hard search call budget (0 reward if exceeded).
    """

    def __init__(
        self,
        search: SearchClient,
        *,
        judge_base_url: str,
        judge_api_key: str,
        judge_model: str,
        corpus_description: str = "a document corpus",
        judge_timeout: float = 30.0,
        w_correctness: float = 1.0,
        w_conciseness: float = 0.5,
        w_citation_recall: float = 0.5,
        w_citation_precision: float = 0.5,
        w_search_efficiency: float = 0.1,
        max_search_calls: int = 10,
        **kwargs: Any,
    ) -> None:
        if not judge_base_url or not judge_api_key or not judge_model:
            raise ValueError(
                "SearchEnv requires judge_base_url, judge_api_key, and judge_model. "
                "All three must be non-empty."
            )

        self._search = search
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._judge_model = judge_model
        self._judge_timeout = judge_timeout
        self._corpus_description = corpus_description
        self._w_correctness = w_correctness
        self._w_conciseness = w_conciseness
        self._w_citation_recall = w_citation_recall
        self._w_citation_precision = w_citation_precision
        self._w_search_efficiency = w_search_efficiency
        self._max_search_calls = max_search_calls
        self._experiment_id = kwargs.get("experiment_id")
        self._rollout_api_key = kwargs.get("api_key")

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
                "description": (f"Search mode. Available: {modes}. Default: {self._default_mode}."),
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

        # Build system prompt.
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            corpus_description=corpus_description,
            max_search_calls=max_search_calls,
        )

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
    def dataset_preprocess(cls, example: Any, **kwargs) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("question", ""),
            ground_truth=example.get("answer", None),
            init_rollout_args={
                "reference_chunks": example.get("reference_chunks", []),
            },
        )

    async def compute_reward(
        self,
        rollout_id: str,
        completion: str | list[dict[str, Any]],
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute 5-component reward."""
        zeros = self._zero_rewards()
        try:
            text = extract_completion_text(completion)
            if not text.strip():
                return zeros

            answer = extract_answer_block(text)
            prompt = str(kwargs.get("prompt") or kwargs.get("question") or "")
            gt_str = str(ground_truth or "")
            reference_chunks = kwargs.get("reference_chunks", [])
            reference_chunk_count = len(reference_chunks)

            log_env(
                rollout_id,
                f"[SearchEnv] Q: {prompt[:200]}\n  GT: {gt_str[:200]}\n  A: {answer[:200]}",
            )

            # 1. Correctness + Conciseness (concurrent judge calls)
            correctness_raw, conciseness_raw = await self._judge_answer_quality(
                question=prompt,
                ground_truth=gt_str,
                response=answer,
            )

            correctness_ok = correctness_raw > 0
            rewards: dict[str, float] = {
                "answer_correctness": self._w_correctness * clip01(correctness_raw),
                "conciseness": (
                    self._w_conciseness * clip01(conciseness_raw) if correctness_ok else 0.0
                ),
            }

            # 2. Citation recall / precision
            recall, precision = self._score_citations(answer, reference_chunks)
            rewards["citation_recall"] = self._w_citation_recall * recall
            rewards["citation_precision"] = self._w_citation_precision * precision

            # 3. Search efficiency (shaped by search count vs. gold chunk baseline)
            calls = count_search_calls(completion)
            rewards["search_efficiency"] = self._score_search_efficiency(
                calls=calls,
                correctness_raw=correctness_raw,
                reference_chunk_count=reference_chunk_count,
            )

            log_env(rollout_id, f"[SearchEnv] rewards={rewards}")
            return rewards

        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            log_env(rollout_id, f"[SearchEnv] compute_reward failed: {exc}")
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
        """Evaluate correctness + conciseness concurrently.

        Returns (correctness_score, conciseness_score) both in [0, 1].
        """
        if not response.strip():
            return (0.0, 0.0)

        try:
            correctness_task = evaluate_single_rubric(
                rubric=_CORRECTNESS_RUBRIC,
                question=question,
                ground_truth=ground_truth,
                response=response,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
            )
            conciseness_task = evaluate_single_rubric(
                rubric=_CONCISENESS_RUBRIC,
                question=question,
                ground_truth=ground_truth,
                response=response,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_api_key,
                timeout=self._judge_timeout,
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
        """Reward correct answers that do not search much past the gold chunk baseline."""
        if correctness_raw <= 0:
            return 0.0
        if not search_within_budget(calls, self._max_search_calls):
            return 0.0

        baseline_calls = reference_chunk_count + 2
        excess_calls = max(0, calls - baseline_calls)
        decay = math.exp(-SEARCH_EFFICIENCY_DECAY_RATE * excess_calls)
        return self._w_search_efficiency * correctness_raw * decay

    def _score_citations(
        self,
        answer_text: str,
        reference_chunks: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """Score citation recall and precision via exact source match.

        Returns (recall, precision).
        """
        ref_ids = self._extract_reference_ids(reference_chunks)
        cited_ids = self._parse_citations(answer_text)

        if not ref_ids:
            return 0.0, 0.0

        overlap = cited_ids & ref_ids
        recall = len(overlap) / len(ref_ids)
        precision = len(overlap) / len(cited_ids) if cited_ids else 0.0
        return recall, precision

    def _extract_reference_ids(self, reference_chunks: list[dict[str, Any]]) -> set[str]:
        """Extract document-level source IDs from reference chunks.

        Default uses the ``file`` metadata key. Override in subclasses
        for corpus-specific ID extraction.
        """
        ids: set[str] = set()
        for chunk in reference_chunks:
            if not isinstance(chunk, dict):
                continue
            md = chunk.get("metadata", {})
            if not isinstance(md, dict):
                continue
            file_id = str(md.get("file") or md.get("file_path") or "").strip()
            if file_id:
                ids.add(self._canonicalize_id(file_id))
        return ids

    def _parse_citations(self, text: str) -> set[str]:
        """Parse [Source: <id>] citations from model answer."""
        ids: set[str] = set()
        for match in _CITATION_RE.finditer(text or ""):
            cid = self._canonicalize_id(match.group(1).strip())
            if cid:
                ids.add(cid)
        return ids

    def _canonicalize_id(self, source_id: str) -> str:
        """Normalize a source ID. Override for corpus-specific rules."""
        return str(source_id or "").strip()

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
