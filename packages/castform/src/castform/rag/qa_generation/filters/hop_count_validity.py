"""Hop-count validity filter for Pipeline.

Validates that multi-hop questions genuinely require all their reference
chunks by testing leave-one-out subsets.  A true N-hop question must be
unanswerable when any single chunk is removed.

This filter replaces both the planned "multi-hop validity filter" and
the ``retrieval_too_easy_llm`` filter with a unified, stronger approach.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from benchmax.auth import StaticBearerAuth
from castform.model_auth import (
    create_async_openai_client,
    create_openai_client,
    model_auth_for_endpoint,
)
from castform.rag.qa_generation.batch_processor import batch_process_async
from castform.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from castform.rag.qa_generation.pipeline_config import PipelineContext
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_FILTER_MODE = "hop_count_validity"
_FAILURE_TYPE_REDUNDANT_CHUNK = "redundant_chunk"
_FAILURE_TYPE_LOPSIDED = "lopsided_chunk_contribution"
_FAILURE_TYPE_LOW_CONTRIBUTION = "low_contribution_chunk"
_FAILURE_TYPE_NONE = "none"

# Sentinel used by _build_subsets in "primary_only" mode to indicate that all
# non-primary chunks were omitted (i.e., testing whether the primary chunk alone
# suffices). A value of -1 is safe because ref_chunk indices are always >= 0.
_PRIMARY_ONLY_SENTINEL = -1
_PRIMARY_ONLY_OMITTED_ID = "all_non_primary"
_MAX_CONSECUTIVE_ERRORS = 5  # circuit-breaker: abort filter after this many consecutive failures

_HOP_COUNT_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating whether a question can be fully answered \
using ONLY the provided evidence chunks.

Rules:
- Consider ONLY the information explicitly stated in the provided chunks.
- The answer must be FULLY supported. Partial support is NOT enough.
- Do NOT use external knowledge.
- If any part of the answer requires information not present in the \
provided chunks, the answer is NOT fully supported.

Also rate the contribution of the OMITTED chunk to the gold answer:
- "high": the omitted chunk provides core facts that are central to the answer
- "medium": the omitted chunk provides supporting context or one key fact
- "low": the omitted chunk provides a single minor detail that barely affects the answer

Respond with JSON only:
{"answerable": <bool>, "confidence": <float 0-1>, \
"reasoning": "<brief explanation>", \
"missing_facts": ["<fact not in provided chunks>", ...], \
"contribution_level": "high" | "medium" | "low"}\
"""  # noqa: E501

_HOP_COUNT_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Evidence chunks:
{chunks_text}

Can the gold answer be fully derived using ONLY the evidence chunks above?\
"""

# QA types considered multi-hop (skip validation for single-hop types).
_MULTI_HOP_TYPES = frozenset({"multi_hop"})


@dataclass
class HopCountValidityConfig:
    """Configuration for the hop-count validity filter."""

    enabled: bool = True
    mode: str = "leave_one_out"  # "primary_only" | "leave_one_out" | "budget"
    max_judge_calls: int = 3  # for "budget" mode
    judge_model: str = "gpt-5.4-mini"
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_system_prompt: str = _HOP_COUNT_JUDGE_SYSTEM_PROMPT
    judge_user_template: str = _HOP_COUNT_JUDGE_USER_TEMPLATE
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True
    stats_key: str = "hop_count_validity_stats"
    # Word-overlap pre-gate thresholds (leave_one_out mode only).
    # A chunk with overlap > lopsided_high_threshold dominates the answer;
    # if another chunk has overlap < lopsided_low_threshold they are complementary
    # but imbalanced, flagging the item as lopsided.
    lopsided_high_threshold: float = 0.60
    lopsided_low_threshold: float = 0.15


class HopCountValidityFilter:
    """Validates that multi-hop questions require all reference chunks.

    Three modes:

    - ``primary_only``: Test with only the primary chunk. If answerable,
      the question is single-hop in disguise. 1 judge call per item.
    - ``leave_one_out``: Test all N-1 subsets. If any subset is
      sufficient, that chunk is redundant. N calls per item.
    - ``budget``: Primary-only first, then additional leave-one-out
      checks up to ``max_judge_calls``.
    """

    def __init__(self, *, cfg: HopCountValidityConfig) -> None:
        self.cfg = cfg
        if cfg.enabled:
            if not cfg.judge_base_url:
                raise ValueError(
                    "HopCountValidityFilter: judge_base_url must be set. "
                    "Set it to your LLM gateway URL (e.g. 'https://llm.castform.com/v1'). "
                    "Without it, requests go to the OpenAI default endpoint, which does not "
                    "recognise the default judge_model."
                )
            judge_auth = model_auth_for_endpoint(
                api_key=cfg.judge_api_key,
                base_url=cfg.judge_base_url,
                purpose="hop-count judge",
            )
        else:
            judge_auth = StaticBearerAuth(cfg.judge_api_key or "disabled")
        self._judge_auth = judge_auth
        judge_base_url = cfg.judge_base_url or "https://disabled.invalid/v1"
        self.judge_client = create_openai_client(
            model=cfg.judge_model,
            base_url=judge_base_url,
            auth=judge_auth,
            request_id="hop-count-judge",
        )
        # Lazily-built loop-bound AsyncOpenAI judge client (see _get_async_judge_client).
        self._async_judge_client: AsyncOpenAI | None = None
        self._async_judge_client_loop: asyncio.AbstractEventLoop | None = None

    def _get_async_judge_client(self) -> AsyncOpenAI:
        """Loop-bound AsyncOpenAI judge client, built on first use. Rebuilds on loop
        change or close, comparing the loop object (not id(loop), which aliases after
        GC)."""
        loop = asyncio.get_running_loop()
        client = self._async_judge_client
        stale = client is None or client.is_closed() or self._async_judge_client_loop is not loop
        if stale:
            client = create_async_openai_client(
                model=self.cfg.judge_model,
                base_url=self.cfg.judge_base_url,
                auth=self._judge_auth,
                request_id="hop-count-judge",
            )
            self._async_judge_client = client
            self._async_judge_client_loop = loop
        return client

    async def aclose(self) -> None:
        """Close the loop-bound AsyncOpenAI judge client built during the run
        (best-effort)."""
        client = self._async_judge_client
        self._async_judge_client = None
        self._async_judge_client_loop = None
        if client is not None and not client.is_closed():
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Error closing async judge client", exc_info=True)

    async def evaluate(
        self, items: list[GeneratedQA], context: PipelineContext
    ) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        self._corpus_language = str(context.get("corpus_language", "") or "").strip()
        max_refinements = context.config.refinement.max_refinements_per_item
        stats = context.setdefault(
            self.cfg.stats_key,
            {
                "passed": 0,
                "needs_refinement": 0,
                "rejected": 0,
                "errors": 0,
                "skipped_single_hop": 0,
                "judge_calls": 0,
                "redundant_chunks_found": 0,
            },
        )

        consecutive_errors = 0

        for item in items:
            if item.filter_verdict is not None and not item.is_passed:
                continue

            # In primary_only mode the only outcomes are "skip" (return None)
            # or "passed" — neither changes the pipeline's pass/fail decision.
            # Skip items that already have a passing verdict from an earlier
            # filter to avoid wasteful sequential judge calls on every round.
            if (
                self.cfg.mode == "primary_only"
                and item.filter_verdict is not None
                and item.is_passed
            ):
                continue

            qa_type = (
                str(
                    item.qa.get("qa_type", "") or item.generation_metadata.get("qa_type_target", "")
                )
                .strip()
                .lower()
            )

            # Skip single-hop types — they don't need multi-hop validation.
            if qa_type and qa_type not in _MULTI_HOP_TYPES:
                stats["skipped_single_hop"] = stats.get("skipped_single_hop", 0) + 1
                continue

            ref_chunks = item.qa.get("reference_chunks", [])
            if len(ref_chunks) <= 1:
                stats["skipped_single_hop"] = stats.get("skipped_single_hop", 0) + 1
                continue

            try:
                verdict = await self._validate_item(
                    item,
                    ref_chunks=ref_chunks,
                    max_refinements=max_refinements,
                    stats=stats,
                    context=context,
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                stats["errors"] = stats.get("errors", 0) + 1
                logger.exception("HopCountValidityFilter failed for one item")
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"HopCountValidityFilter: {consecutive_errors} consecutive errors — "
                        "aborting. Check judge_model, judge_api_key, and judge_base_url config. "
                        f"Last error: {exc}"
                    ) from exc
                continue

            if verdict is not None:
                item.filter_verdict = verdict
                self._update_stats(stats, verdict)

        return items

    async def _validate_item(
        self,
        item: GeneratedQA,
        *,
        ref_chunks: list[dict[str, Any]],
        max_refinements: int,
        stats: dict[str, Any],
        context: PipelineContext,
    ) -> FilterVerdict | None:
        """Run hop-count validation on a single item."""
        question = str(item.qa.get("question", ""))
        answer = str(item.qa.get("answer", ""))
        refinements = int(item.generation_metadata.get("refinement_count", 0))

        mode = self.cfg.mode

        # Word-overlap pre-gate: runs only in leave_one_out mode with 2+ chunks.
        # Detects lopsided contribution without any LLM call and short-circuits.
        if mode == "leave_one_out" and len(ref_chunks) > 1:
            lopsided_verdict = self._check_lopsided_overlap(
                answer=answer,
                ref_chunks=ref_chunks,
                refinements=refinements,
                max_refinements=max_refinements,
            )
            if lopsided_verdict is not None:
                return lopsided_verdict

        subsets_to_test = self._build_subsets(ref_chunks, mode)

        if self.cfg.batch_enabled and len(subsets_to_test) > 1:
            return await self._validate_item_batched(
                item=item,
                question=question,
                answer=answer,
                ref_chunks=ref_chunks,
                subsets_to_test=subsets_to_test,
                refinements=refinements,
                max_refinements=max_refinements,
                stats=stats,
                context=context,
            )

        per_subset_results: list[dict[str, Any]] = []
        redundant_chunks: list[str] = []
        # Chunks that passed the redundancy check but have low contribution.
        low_contribution_chunks: list[dict[str, str]] = []

        for omitted_idx, subset in subsets_to_test:
            if omitted_idx == _PRIMARY_ONLY_SENTINEL:
                # primary_only mode: testing whether the primary chunk alone suffices.
                # If it does, this is a single-hop question — skip the multi-hop
                # validator entirely and let it pass through to other filters.
                async with context.model_semaphore(self.cfg.judge_model, self.cfg.judge_base_url):
                    result = await asyncio.to_thread(
                        self._judge_subset, question, answer, subset, stats
                    )
                if result["answerable"]:
                    logger.debug(
                        "hop_count_validity: question is single-hop (primary chunk alone "
                        "suffices) — skipping multi-hop validation"
                    )
                    stats["skipped_single_hop"] = stats.get("skipped_single_hop", 0) + 1
                    return None
                # Primary chunk alone is NOT sufficient — multi-hop is genuine, continue.
                per_subset_results.append(
                    {
                        "omitted": _PRIMARY_ONLY_OMITTED_ID,
                        "answerable": False,
                        "confidence": result["confidence"],
                        "missing_facts": result.get("missing_facts", []),
                        "contribution_level": result.get("contribution_level", ""),
                    }
                )
                continue

            omitted_chunk = ref_chunks[omitted_idx]
            omitted_id = str(omitted_chunk.get("id", f"chunk_{omitted_idx}"))
            omitted_file = str(
                omitted_chunk.get("metadata", {}).get("file", omitted_chunk.get("file", omitted_id))
            )

            async with context.model_semaphore(self.cfg.judge_model, self.cfg.judge_base_url):
                result = await asyncio.to_thread(
                    self._judge_subset, question, answer, subset, stats
                )
            contribution_level = str(result.get("contribution_level", "")).lower().strip()
            per_subset_results.append(
                {
                    "omitted": omitted_id,
                    "answerable": result["answerable"],
                    "confidence": result["confidence"],
                    "missing_facts": result.get("missing_facts", []),
                    "contribution_level": contribution_level,
                }
            )

            if result["answerable"]:
                redundant_chunks.append(omitted_id)
            elif contribution_level == "low":
                # Chunk is technically essential but barely contributes — flag for refinement.
                low_contribution_chunks.append({"chunk_id": omitted_id, "file": omitted_file})

        hop_count_claimed = len(ref_chunks)
        validated = len(redundant_chunks) == 0
        difficulty = _compute_difficulty(hop_count_claimed, per_subset_results)

        # If all redundancy checks pass but some chunks have low contribution, flag for refinement.
        if validated and low_contribution_chunks:
            low_ids = ", ".join(c["chunk_id"] for c in low_contribution_chunks)
            feedback = (
                f"Chunks with low contribution detected: {low_ids}. "
                "Consider deepening the question's reliance on these chunks."
            )
            if refinements >= max_refinements:
                return FilterVerdict(
                    status="rejected",
                    reason="hop_count_validity_rejected",
                    reasoning=feedback,
                    metadata={
                        **self._build_metadata(
                            hop_count_claimed,
                            per_subset_results,
                            redundant_chunks=[],
                            validated=True,
                        ),
                        "failure_type": _FAILURE_TYPE_LOW_CONTRIBUTION,
                        "low_contribution_chunks": low_contribution_chunks,
                    },
                )
            return FilterVerdict(
                status="needs_refinement",
                reason="hop_count_validity_needs_refinement",
                reasoning=feedback,
                metadata={
                    **self._build_metadata(
                        hop_count_claimed,
                        per_subset_results,
                        redundant_chunks=[],
                        validated=True,
                    ),
                    "failure_type": _FAILURE_TYPE_LOW_CONTRIBUTION,
                    "low_contribution_chunks": low_contribution_chunks,
                    "refinement_hint": (
                        "Increase the question's dependence on all reference chunks. " + feedback
                    ),
                },
            )

        if validated:
            return FilterVerdict(
                status="passed",
                reason="hop_count_validity_passed",
                reasoning="All chunks are essential to the answer.",
                metadata={
                    "filter_mode": _FILTER_MODE,
                    "reason_code": "hop_count_valid",
                    "confidence": 1.0,
                    "hop_count_claimed": hop_count_claimed,
                    "hop_count_validated": True,
                    "subsets_tested": len(per_subset_results),
                    "subsets_total": hop_count_claimed,
                    "redundant_chunks": [],
                    "per_subset": per_subset_results,
                    "failure_type": _FAILURE_TYPE_NONE,
                    "difficulty_score": difficulty,
                },
            )

        stats["redundant_chunks_found"] = stats.get("redundant_chunks_found", 0) + len(
            redundant_chunks
        )

        # If removing redundant chunks still leaves 2+ essential chunks,
        # demote by stripping the redundant ones instead of rejecting.
        essential_count = hop_count_claimed - len(redundant_chunks)
        if essential_count >= 2:
            redundant_set = set(redundant_chunks)
            essential_chunks = [c for c in ref_chunks if str(c.get("id", "")) not in redundant_set]
            removed_chunks = [c for c in ref_chunks if str(c.get("id", "")) in redundant_set]
            item.qa["reference_chunks"] = essential_chunks
            if removed_chunks:
                existing = list(item.qa.get("removed_reference_chunks", []))
                existing.extend(
                    {"chunk": c, "reason": "redundant", "filter": "hop_count_validity"}
                    for c in removed_chunks
                )
                item.qa["removed_reference_chunks"] = existing
            item.qa["min_hop_count"] = essential_count
            stats["demoted"] = stats.get("demoted", 0) + 1
            return FilterVerdict(
                status="passed",
                reason="hop_count_validity_passed",
                reasoning=(
                    f"Removed {len(redundant_chunks)} redundant chunk(s);"
                    f" {essential_count} essential chunks remain."
                ),
                metadata={
                    **self._build_metadata(
                        hop_count_claimed,
                        per_subset_results,
                        redundant_chunks,
                        validated=False,
                    ),
                    "demoted": True,
                    "original_chunk_count": hop_count_claimed,
                    "essential_chunk_count": essential_count,
                    "removed_chunk_ids": redundant_chunks,
                },
            )

        # Only 1 or 0 essential chunks remain — reject or refine.
        feedback_parts = []
        for rc_id in redundant_chunks:
            subset_result = next(
                (r for r in per_subset_results if r["omitted"] == rc_id),
                None,
            )
            if subset_result and subset_result.get("missing_facts"):
                feedback_parts.append(
                    f"Chunk {rc_id} is redundant. Missing facts: {subset_result['missing_facts']}"
                )
            else:
                feedback_parts.append(
                    f"Chunk {rc_id} is redundant — the answer can be derived without it."
                )

        feedback = "; ".join(feedback_parts)

        if refinements >= max_refinements:
            return FilterVerdict(
                status="rejected",
                reason="hop_count_validity_rejected",
                reasoning=feedback,
                metadata=self._build_metadata(
                    hop_count_claimed,
                    per_subset_results,
                    redundant_chunks,
                    validated=False,
                ),
            )

        return FilterVerdict(
            status="needs_refinement",
            reason="hop_count_validity_needs_refinement",
            reasoning=feedback,
            metadata={
                **self._build_metadata(
                    hop_count_claimed,
                    per_subset_results,
                    redundant_chunks,
                    validated=False,
                ),
                "failure_type": _FAILURE_TYPE_REDUNDANT_CHUNK,
                "refinement_hint": (
                    "Make all reference chunks essential to the answer. " + feedback
                ),
            },
        )

    async def _validate_item_batched(  # noqa: E501
        self,
        *,
        item: GeneratedQA,
        question: str,
        answer: str,
        ref_chunks: list[dict[str, Any]],
        subsets_to_test: list[tuple[int, list[dict[str, Any]]]],
        refinements: int,
        max_refinements: int,
        stats: dict[str, Any],
        context: PipelineContext,
    ) -> FilterVerdict | None:
        """Batched version of the subset-judging loop in _validate_item.

        Sends all judge prompts through ``batch_process_async`` (bounded by the
        run's per-model semaphore) so they run concurrently, then processes the
        results with the same logic as the sequential path.
        """
        # Build all prompts up-front.
        prompts: list[str] = []
        for _omitted_idx, subset in subsets_to_test:
            prompts.append(self._build_judge_prompt(question, answer, subset))

        stats["judge_calls"] = stats.get("judge_calls", 0) + len(prompts)

        system_prompt = self.cfg.judge_system_prompt
        corpus_language = getattr(self, "_corpus_language", "")
        if corpus_language:
            system_prompt = (
                f"NOTE: The question, answer, and evidence are in {corpus_language}. "
                f"Evaluate them in {corpus_language}.\n\n{system_prompt}"
            )

        batch_result = await batch_process_async(
            client=self._get_async_judge_client(),
            model=self.cfg.judge_model,
            prompts=prompts,
            system_prompt=system_prompt,
            max_tokens=500,
            timeout=60.0,
            max_concurrent=self.cfg.max_concurrent,
            show_progress=self.cfg.show_batch_progress,
            temperature=0.0,
            desc="Hop-count validity",
            semaphore=context.model_semaphore(self.cfg.judge_model, self.cfg.judge_base_url),
        )

        # Process results — mirrors the sequential loop exactly.
        per_subset_results: list[dict[str, Any]] = []
        redundant_chunks: list[str] = []
        low_contribution_chunks: list[dict[str, str]] = []

        for idx, (omitted_idx, _subset) in enumerate(subsets_to_test):
            resp = batch_result.responses[idx]
            raw = resp.answer if resp is not None else "{}"
            result = self._parse_judge_response(raw)

            if omitted_idx == _PRIMARY_ONLY_SENTINEL:
                if result["answerable"]:
                    logger.debug(
                        "hop_count_validity: question is "
                        "single-hop (primary chunk alone "
                        "suffices) — skipping multi-hop "
                        "validation"
                    )
                    stats["skipped_single_hop"] = stats.get("skipped_single_hop", 0) + 1
                    return None
                per_subset_results.append(
                    {
                        "omitted": _PRIMARY_ONLY_OMITTED_ID,
                        "answerable": False,
                        "confidence": result["confidence"],
                        "missing_facts": result.get("missing_facts", []),
                        "contribution_level": result.get("contribution_level", ""),
                    }
                )
                continue

            omitted_chunk = ref_chunks[omitted_idx]
            omitted_id = str(omitted_chunk.get("id", f"chunk_{omitted_idx}"))
            omitted_file = str(
                omitted_chunk.get("metadata", {}).get(
                    "file",
                    omitted_chunk.get("file", omitted_id),
                )
            )

            contribution_level = str(result.get("contribution_level", "")).lower().strip()
            per_subset_results.append(
                {
                    "omitted": omitted_id,
                    "answerable": result["answerable"],
                    "confidence": result["confidence"],
                    "missing_facts": result.get("missing_facts", []),
                    "contribution_level": contribution_level,
                }
            )

            if result["answerable"]:
                redundant_chunks.append(omitted_id)
            elif contribution_level == "low":
                low_contribution_chunks.append(
                    {
                        "chunk_id": omitted_id,
                        "file": omitted_file,
                    }
                )

        # From here the logic is identical to the sequential path.
        hop_count_claimed = len(ref_chunks)
        validated = len(redundant_chunks) == 0
        difficulty = _compute_difficulty(hop_count_claimed, per_subset_results)

        if validated and low_contribution_chunks:
            low_ids = ", ".join(c["chunk_id"] for c in low_contribution_chunks)
            feedback = (
                f"Chunks with low contribution detected: "
                f"{low_ids}. Consider deepening the question's "
                "reliance on these chunks."
            )
            if refinements >= max_refinements:
                return FilterVerdict(
                    status="rejected",
                    reason="hop_count_validity_rejected",
                    reasoning=feedback,
                    metadata={
                        **self._build_metadata(
                            hop_count_claimed,
                            per_subset_results,
                            redundant_chunks=[],
                            validated=True,
                        ),
                        "failure_type": (_FAILURE_TYPE_LOW_CONTRIBUTION),
                        "low_contribution_chunks": (low_contribution_chunks),
                    },
                )
            return FilterVerdict(
                status="needs_refinement",
                reason="hop_count_validity_needs_refinement",
                reasoning=feedback,
                metadata={
                    **self._build_metadata(
                        hop_count_claimed,
                        per_subset_results,
                        redundant_chunks=[],
                        validated=True,
                    ),
                    "failure_type": (_FAILURE_TYPE_LOW_CONTRIBUTION),
                    "low_contribution_chunks": (low_contribution_chunks),
                    "refinement_hint": (
                        "Increase the question's dependence on all reference chunks. " + feedback
                    ),
                },
            )

        if validated:
            return FilterVerdict(
                status="passed",
                reason="hop_count_validity_passed",
                reasoning=("All chunks are essential to the answer."),
                metadata={
                    "filter_mode": _FILTER_MODE,
                    "reason_code": "hop_count_valid",
                    "confidence": 1.0,
                    "hop_count_claimed": hop_count_claimed,
                    "hop_count_validated": True,
                    "subsets_tested": len(per_subset_results),
                    "subsets_total": hop_count_claimed,
                    "redundant_chunks": [],
                    "per_subset": per_subset_results,
                    "failure_type": _FAILURE_TYPE_NONE,
                    "difficulty_score": difficulty,
                },
            )

        stats["redundant_chunks_found"] = stats.get("redundant_chunks_found", 0) + len(
            redundant_chunks
        )

        essential_count = hop_count_claimed - len(redundant_chunks)
        if essential_count >= 2:
            redundant_set = set(redundant_chunks)
            essential_chunks = [c for c in ref_chunks if str(c.get("id", "")) not in redundant_set]
            removed_chunks = [c for c in ref_chunks if str(c.get("id", "")) in redundant_set]
            item.qa["reference_chunks"] = essential_chunks
            if removed_chunks:
                existing = list(item.qa.get("removed_reference_chunks", []))
                existing.extend(
                    {"chunk": c, "reason": "redundant", "filter": "hop_count_validity"}
                    for c in removed_chunks
                )
                item.qa["removed_reference_chunks"] = existing
            item.qa["min_hop_count"] = essential_count
            stats["demoted"] = stats.get("demoted", 0) + 1
            return FilterVerdict(
                status="passed",
                reason="hop_count_validity_passed",
                reasoning=(
                    f"Removed {len(redundant_chunks)} redundant"
                    f" chunk(s); {essential_count} essential"
                    " chunks remain."
                ),
                metadata={
                    **self._build_metadata(
                        hop_count_claimed,
                        per_subset_results,
                        redundant_chunks,
                        validated=False,
                    ),
                    "demoted": True,
                    "original_chunk_count": hop_count_claimed,
                    "essential_chunk_count": essential_count,
                    "removed_chunk_ids": redundant_chunks,
                },
            )

        feedback_parts = []
        for rc_id in redundant_chunks:
            subset_result = next(
                (r for r in per_subset_results if r["omitted"] == rc_id),
                None,
            )
            if subset_result and subset_result.get("missing_facts"):
                feedback_parts.append(
                    f"Chunk {rc_id} is redundant. Missing facts: {subset_result['missing_facts']}"
                )
            else:
                feedback_parts.append(
                    f"Chunk {rc_id} is redundant — the answer can be derived without it."
                )

        feedback = "; ".join(feedback_parts)

        if refinements >= max_refinements:
            return FilterVerdict(
                status="rejected",
                reason="hop_count_validity_rejected",
                reasoning=feedback,
                metadata=self._build_metadata(
                    hop_count_claimed,
                    per_subset_results,
                    redundant_chunks,
                    validated=False,
                ),
            )

        return FilterVerdict(
            status="needs_refinement",
            reason="hop_count_validity_needs_refinement",
            reasoning=feedback,
            metadata={
                **self._build_metadata(
                    hop_count_claimed,
                    per_subset_results,
                    redundant_chunks,
                    validated=False,
                ),
                "failure_type": _FAILURE_TYPE_REDUNDANT_CHUNK,
                "refinement_hint": (
                    "Make all reference chunks essential to the answer. " + feedback
                ),
            },
        )

    def _check_lopsided_overlap(
        self,
        *,
        answer: str,
        ref_chunks: list[dict[str, Any]],
        refinements: int,
        max_refinements: int,
    ) -> FilterVerdict | None:
        """Word-overlap pre-gate: detect lopsided multi-hop QA pairs without an LLM call.

        Returns a FilterVerdict if the item is lopsided (one chunk dominates the answer
        while another barely overlaps), otherwise returns None to continue to judge evaluation.
        """
        answer_words = set(answer.lower().split())
        if not answer_words:
            return None

        overlaps: list[tuple[str, str, float]] = []  # (chunk_id, file, overlap_ratio)
        for chunk in ref_chunks:
            chunk_id = str(chunk.get("id", "unknown"))
            chunk_file = str(chunk.get("metadata", {}).get("file", chunk.get("file", chunk_id)))
            chunk_words = set(str(chunk.get("content", "")).lower().split())
            if not chunk_words:
                overlap = 0.0
            else:
                overlap = len(answer_words & chunk_words) / len(answer_words)
            overlaps.append((chunk_id, chunk_file, overlap))

        high_threshold = self.cfg.lopsided_high_threshold
        low_threshold = self.cfg.lopsided_low_threshold

        over_represented = [(cid, f, r) for cid, f, r in overlaps if r > high_threshold]
        under_represented = [(cid, f, r) for cid, f, r in overlaps if r < low_threshold]

        if not (over_represented and under_represented):
            # Not lopsided — proceed to judge evaluation.
            return None

        logger.debug(
            "hop_count_validity: lopsided overlap detected — over: %s, under: %s",
            [(cid, round(r, 3)) for cid, _, r in over_represented],
            [(cid, round(r, 3)) for cid, _, r in under_represented],
        )

        over_info = [
            {"chunk_id": cid, "file": f, "overlap": round(r, 4)} for cid, f, r in over_represented
        ]
        under_info = [
            {"chunk_id": cid, "file": f, "overlap": round(r, 4)} for cid, f, r in under_represented
        ]
        feedback = (
            "lopsided_chunk_contribution: "
            f"over-represented chunks (overlap>{high_threshold}): "
            f"{[c['chunk_id'] for c in over_info]}; "
            f"under-represented chunks (overlap<{low_threshold}): "
            f"{[c['chunk_id'] for c in under_info]}"
        )

        hop_count_claimed = len(ref_chunks)
        base_metadata: dict[str, Any] = {
            "filter_mode": _FILTER_MODE,
            "reason_code": _FAILURE_TYPE_LOPSIDED,
            "confidence": 0.5,
            "hop_count_claimed": hop_count_claimed,
            "hop_count_validated": False,
            "subsets_tested": 0,
            "subsets_total": hop_count_claimed,
            "redundant_chunks": [],
            "per_subset": [],
            "failure_type": _FAILURE_TYPE_LOPSIDED,
            "over_represented_chunks": over_info,
            "under_represented_chunks": under_info,
            "difficulty_score": 0.0,
        }

        if refinements >= max_refinements:
            return FilterVerdict(
                status="rejected",
                reason="hop_count_validity_rejected",
                reasoning=feedback,
                metadata=base_metadata,
            )

        return FilterVerdict(
            status="needs_refinement",
            reason="hop_count_validity_needs_refinement",
            reasoning=feedback,
            metadata={
                **base_metadata,
                "refinement_hint": (
                    "Rewrite the question so that all reference chunks contribute "
                    "meaningfully to the answer. " + feedback
                ),
            },
        )

    def _build_subsets(
        self,
        ref_chunks: list[dict[str, Any]],
        mode: str,
    ) -> list[tuple[int, list[dict[str, Any]]]]:
        """Build the (omitted_idx, subset) pairs to test."""
        n = len(ref_chunks)

        if mode == "primary_only":
            # Test whether the primary chunk alone suffices to answer the question.
            # Uses _PRIMARY_ONLY_SENTINEL as omitted_idx so _validate_item knows to
            # emit "question is single-hop" feedback rather than naming a specific
            # chunk as redundant.
            if n > 1:
                return [(_PRIMARY_ONLY_SENTINEL, [ref_chunks[0]])]
            return []

        if mode == "leave_one_out":
            return [(i, ref_chunks[:i] + ref_chunks[i + 1 :]) for i in range(n)]

        # "budget" mode: primary_only first, then additional leave-one-out.
        subsets: list[tuple[int, list[dict[str, Any]]]] = []
        # First: primary-only check using the sentinel so feedback is correct.
        if n > 1:
            subsets.append((_PRIMARY_ONLY_SENTINEL, [ref_chunks[0]]))

        # Additional leave-one-out checks up to budget.
        remaining_budget = self.cfg.max_judge_calls - len(subsets)
        for i in range(n):
            if remaining_budget <= 0:
                break
            subset = ref_chunks[:i] + ref_chunks[i + 1 :]
            entry = (i, subset)
            if entry not in subsets:
                subsets.append(entry)
                remaining_budget -= 1

        return subsets

    def _build_judge_prompt(
        self,
        question: str,
        answer: str,
        subset: list[dict[str, Any]],
    ) -> str:
        """Build the user prompt for the LLM judge."""
        chunks_text = "\n\n".join(
            f"[Chunk {i}] {chunk.get('content', '')}" for i, chunk in enumerate(subset)
        )
        return self.cfg.judge_user_template.format(
            question=question,
            answer=answer,
            chunks_text=chunks_text,
        )

    @staticmethod
    def _parse_judge_response(raw: str) -> dict[str, Any]:
        """Parse the JSON response from the LLM judge."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse judge response: %s", raw[:200])
            data = {}
        return {
            "answerable": bool(data.get("answerable", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasoning": str(data.get("reasoning", "")),
            "missing_facts": list(data.get("missing_facts", [])),
            "contribution_level": str(data.get("contribution_level", "")).lower().strip(),
        }

    def _judge_subset(
        self,
        question: str,
        answer: str,
        subset: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the LLM judge on a chunk subset."""
        prompt = self._build_judge_prompt(question, answer, subset)
        stats["judge_calls"] = stats.get("judge_calls", 0) + 1
        system_prompt = self.cfg.judge_system_prompt
        corpus_language = getattr(self, "_corpus_language", "")
        if corpus_language:
            system_prompt = (
                f"NOTE: The question, answer, and evidence are in {corpus_language}. "
                f"Evaluate them in {corpus_language}.\n\n{system_prompt}"
            )
        completion = self.judge_client.chat.completions.create(
            model=self.cfg.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or "{}"
        return self._parse_judge_response(raw)

    def _build_metadata(
        self,
        hop_count_claimed: int,
        per_subset: list[dict[str, Any]],
        redundant_chunks: list[str],
        *,
        validated: bool,
    ) -> dict[str, Any]:
        return {
            "filter_mode": _FILTER_MODE,
            "reason_code": ("hop_count_valid" if validated else "redundant_chunk_found"),
            "confidence": 1.0 if validated else 0.5,
            "hop_count_claimed": hop_count_claimed,
            "hop_count_validated": validated,
            "subsets_tested": len(per_subset),
            "subsets_total": hop_count_claimed,
            "redundant_chunks": redundant_chunks,
            "per_subset": per_subset,
            "difficulty_score": _compute_difficulty(hop_count_claimed, per_subset),
        }

    @staticmethod
    def _update_stats(stats: dict[str, Any], verdict: FilterVerdict) -> None:
        if verdict.status == "passed":
            stats["passed"] = stats.get("passed", 0) + 1
        elif verdict.status == "needs_refinement":
            stats["needs_refinement"] = stats.get("needs_refinement", 0) + 1
        elif verdict.status == "rejected":
            stats["rejected"] = stats.get("rejected", 0) + 1


def _compute_difficulty(
    hop_count: int,
    per_subset: list[dict[str, Any]],
) -> float:
    """Compute a difficulty score (0.0-1.0) from hop-count validation results.

    The score combines two signals:

    - **Hop depth**: more chunks required → harder. Normalized as
      ``min(hop_count / 4, 1.0)`` so a 4-hop question scores 1.0.
    - **Judge confidence**: average judge confidence that subsets are
      NOT answerable (i.e., each chunk is essential). Higher confidence
      in "not answerable" → the chunks are more tightly coupled → harder.

    The final score is a weighted average: 40% hop depth + 60% confidence.
    """
    # Hop depth component: 1-hop=0.25, 2-hop=0.50, 3-hop=0.75, 4+=1.0
    hop_component = min(hop_count / 4.0, 1.0)

    # Confidence component: average confidence that subsets are NOT answerable
    if not per_subset:
        confidence_component = 0.5
    else:
        confidences = []
        for result in per_subset:
            if not result.get("answerable", True):
                # Not answerable → chunk is essential. Use confidence directly.
                confidences.append(float(result.get("confidence", 0.5)))
            else:
                # Answerable → chunk is redundant. Invert confidence.
                confidences.append(1.0 - float(result.get("confidence", 0.5)))
        confidence_component = sum(confidences) / len(confidences)

    return round(0.4 * hop_component + 0.6 * confidence_component, 4)
