"""Grounding + LLM-judge filter for Pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from benchmax.auth import StaticBearerAuth
from castform.model_auth import (
    create_async_openai_client,
    create_openai_client,
    model_auth_for_endpoint,
)
from castform.rag.qa_generation.batch_processor import batch_process_async
from castform.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from castform.rag.qa_generation.models import ReferenceChunk
from castform.rag.qa_generation.pipeline_config import (
    DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT,
    DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE,
    GroundingLLMFilterConfig,
    PipelineContext,
)
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_FAILURE_TYPE_UNSUPPORTED = "unsupported"
_FAILURE_TYPE_NONE = "none"


@dataclass
class _GroundingItemData:
    """Prepared data for one item in batch mode."""

    item: GeneratedQA
    ref_chunks: list[dict[str, Any]]
    evidence_blocks: list[str]


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class GroundingLLMFilter:
    """Flags QA pairs whose gold answer is not fully grounded in available evidence."""

    def __init__(
        self,
        *,
        chunk_source: Any,
        cfg: GroundingLLMFilterConfig,
    ) -> None:
        self.chunk_source = chunk_source
        self.cfg = cfg
        self._judge_auth = (
            model_auth_for_endpoint(
                api_key=cfg.judge_api_key,
                base_url=cfg.judge_base_url,
                purpose="grounding judge",
            )
            if cfg.enabled
            else StaticBearerAuth(cfg.judge_api_key or "disabled")
        )
        self.judge_client = create_openai_client(
            model=cfg.judge_model,
            base_url=cfg.judge_base_url,
            auth=self._judge_auth,
            request_id="grounding-judge",
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
                request_id="grounding-judge",
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
        stats_key = str(self.cfg.stats_key or "").strip() or "grounding_filter_stats"
        stats = context.setdefault(
            stats_key,
            {
                "passed": 0,
                "needs_refinement": 0,
                "rejected": 0,
                "errors": 0,
                "judge_answerable_true": 0,
                "judge_answerable_false": 0,
                "unsupported_total": 0,
            },
        )
        self._ensure_stats_shape(stats)

        if not self.cfg.batch_enabled:
            for item in items:
                if item.filter_verdict is not None and not item.is_passed:
                    continue
                try:
                    async with context.model_semaphore(
                        self.cfg.judge_model, self.cfg.judge_base_url
                    ):
                        verdict = await asyncio.to_thread(
                            self._evaluate_item,
                            item,
                            max_refinements=max_refinements,
                        )
                except Exception:
                    logger.exception("GroundingLLMFilter failed for one item")
                    refinements = int(item.generation_metadata.get("refinement_count", 0))
                    verdict = self._error_verdict(
                        str(item.qa.get("question", "")),
                        refinements=refinements,
                        max_refinements=max_refinements,
                    )
                    stats["errors"] = int(stats.get("errors", 0)) + 1
                item.filter_verdict = verdict
                self._update_stats(stats, verdict)
            return items

        return await self._evaluate_batch(
            items,
            context=context,
            stats=stats,
            max_refinements=max_refinements,
        )

    async def _evaluate_batch(
        self,
        items: list[GeneratedQA],
        *,
        context: PipelineContext,
        stats: dict[str, Any],
        max_refinements: int,
    ) -> list[GeneratedQA]:
        """Batch evaluation: prep evidence from reference chunks, judge in parallel."""
        prepared: list[_GroundingItemData] = []
        for item in items:
            if item.filter_verdict is not None and not item.is_passed:
                continue

            question = str(item.qa.get("question", "")).strip()
            if not question:
                item.filter_verdict = FilterVerdict(
                    status="rejected",
                    reason="grounding_filter_rejected",
                    reasoning="Missing question.",
                    metadata={
                        "filter_mode": "grounding_llm",
                        "reason_code": "missing_question",
                        "confidence": 1.0,
                        "failure_type": _FAILURE_TYPE_NONE,
                        "force_reanchor": False,
                        "feedback_type": None,
                        "refinement_hint": None,
                    },
                )
                self._update_stats(stats, item.filter_verdict)
                continue

            ref_chunks = list(item.qa.get("reference_chunks", []) or [])
            evidence_blocks = self._build_evidence_blocks(ref_chunks)
            prepared.append(
                _GroundingItemData(
                    item=item,
                    ref_chunks=ref_chunks,
                    evidence_blocks=evidence_blocks,
                )
            )

        if not prepared:
            return items

        # Phase 2: batch LLM judge calls (skip items with no evidence)
        judge_prompts: list[str] = []
        needs_judge_indices: list[int] = []  # indices into `prepared`
        for i, data in enumerate(prepared):
            prompt = self._build_judge_prompt(data.item, data.evidence_blocks)
            if prompt is not None:
                judge_prompts.append(prompt)
                needs_judge_indices.append(i)

        judge_results: dict[int, dict[str, Any]] = {}
        if judge_prompts:
            system_prompt = self.cfg.judge_system_prompt or DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
            corpus_language = getattr(self, "_corpus_language", "")
            if corpus_language:
                system_prompt = (
                    f"NOTE: The question, answer, and evidence are in {corpus_language}. "
                    f"Evaluate them in {corpus_language}.\n\n{system_prompt}"
                )
            batch_result = await batch_process_async(
                client=self._get_async_judge_client(),
                model=self.cfg.judge_model,
                prompts=judge_prompts,
                system_prompt=system_prompt,
                max_tokens=500,
                timeout=60.0,
                max_concurrent=self.cfg.max_concurrent,
                show_progress=self.cfg.show_batch_progress,
                temperature=0.0,
                desc="Grounding filter",
                semaphore=context.model_semaphore(self.cfg.judge_model, self.cfg.judge_base_url),
            )
            for j, i in enumerate(needs_judge_indices):
                response = batch_result.responses[j]
                if response is None:
                    judge_results[i] = {
                        "answerable": False,
                        "confidence": 0.0,
                        "reasoning": "batch_call_failed",
                    }
                else:
                    judge_results[i] = self._parse_judge_response(response.answer or "{}")

        # Phase 3: apply verdicts
        no_evidence_result = {
            "answerable": False,
            "confidence": 1.0,
            "reasoning": "No evidence chunks available.",
        }
        for i, data in enumerate(prepared):
            try:
                judge_result = judge_results.get(i, no_evidence_result)
                verdict = self._verdict_from_judge_result(
                    data.item,
                    judge_result=judge_result,
                    ref_chunks=data.ref_chunks,
                    evidence_blocks=data.evidence_blocks,
                    max_refinements=max_refinements,
                )
            except Exception:
                logger.exception("GroundingLLMFilter failed for one item")
                refinements = int(data.item.generation_metadata.get("refinement_count", 0))
                verdict = self._error_verdict(
                    str(data.item.qa.get("question", "")),
                    refinements=refinements,
                    max_refinements=max_refinements,
                )
                stats["errors"] = int(stats.get("errors", 0)) + 1
            data.item.filter_verdict = verdict
            self._update_stats(stats, verdict)

        return items

    def _evaluate_item(
        self,
        item: GeneratedQA,
        *,
        max_refinements: int,
    ) -> FilterVerdict:
        question = str(item.qa.get("question", "")).strip()
        if not question:
            return FilterVerdict(
                status="rejected",
                reason="grounding_filter_rejected",
                reasoning="Missing question.",
                metadata={
                    "filter_mode": "grounding_llm",
                    "reason_code": "missing_question",
                    "confidence": 1.0,
                    "failure_type": _FAILURE_TYPE_NONE,
                    "force_reanchor": False,
                    "feedback_type": None,
                    "refinement_hint": None,
                },
            )

        ref_chunks = list(item.qa.get("reference_chunks", []) or [])
        evidence_blocks = self._build_evidence_blocks(ref_chunks)

        judge_result = self._run_judge(item, evidence_blocks)
        return self._verdict_from_judge_result(
            item,
            judge_result=judge_result,
            ref_chunks=ref_chunks,
            evidence_blocks=evidence_blocks,
            max_refinements=max_refinements,
        )

    @staticmethod
    def _build_evidence_blocks(ref_chunks: list[dict[str, Any]]) -> list[str]:
        blocks: list[str] = []
        seen: set[str] = set()
        for ref in ref_chunks:
            ref_id = str(ref.get("id", "")).strip()
            content = str(ref.get("content", "")).strip()
            if not content:
                continue
            key = ref_id or content
            if key in seen:
                continue
            seen.add(key)
            label = ref_id or "unknown_ref"
            blocks.append(f"[{label}]\n{content}")
        return blocks

    def _build_judge_prompt(self, item: GeneratedQA, evidence_blocks: list[str]) -> str | None:
        """Build judge user prompt. Returns None if no evidence (no judge call needed)."""
        if not evidence_blocks:
            return None
        chunks_text = "\n---\n".join(evidence_blocks)
        prompt_vars = {
            "question": item.qa.get("question", ""),
            "answer": item.qa.get("answer", ""),
            "chunks_text": chunks_text,
        }
        user_template = (
            str(self.cfg.judge_user_template or "").strip() or DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE
        )
        try:
            return user_template.format(**prompt_vars)
        except KeyError:
            logger.warning(
                "Grounding judge user template requires unknown placeholders; "
                "falling back to default template."
            )
            return DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE.format(**prompt_vars)

    @staticmethod
    def _parse_judge_response(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
            return payload
        except json.JSONDecodeError:
            logger.warning("Grounding judge response parse failure: %s", raw[:240])
            return {"answerable": False, "confidence": 0.0, "reasoning": "parse_error"}

    def _run_judge(self, item: GeneratedQA, evidence_blocks: list[str]) -> dict[str, Any]:
        prompt = self._build_judge_prompt(item, evidence_blocks)
        if prompt is None:
            return {
                "answerable": False,
                "confidence": 1.0,
                "reasoning": "No evidence chunks available.",
            }

        system_prompt = self.cfg.judge_system_prompt or DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
        corpus_language = getattr(self, "_corpus_language", "")
        if corpus_language:
            system_prompt = (
                f"NOTE: The question, answer, and evidence are in {corpus_language}. "
                f"Evaluate them in {corpus_language}.\n\n{system_prompt}"
            )
        response = self.judge_client.chat.completions.create(
            model=self.cfg.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return self._parse_judge_response(response.choices[0].message.content or "{}")

    def _verdict_from_judge_result(
        self,
        item: GeneratedQA,
        *,
        judge_result: dict[str, Any],
        ref_chunks: list[dict[str, Any]],
        evidence_blocks: list[str],
        max_refinements: int,
    ) -> FilterVerdict:
        answerable = bool(judge_result.get("answerable", False))
        confidence = _safe_float(judge_result.get("confidence", 0.0), default=0.0)
        judge_reasoning = str(judge_result.get("reasoning", "")).strip()
        refinements = int(item.generation_metadata.get("refinement_count", 0))

        # Update reference_chunks from judge's supporting_chunk_ids; track removed chunks.
        raw_supporting_ids = judge_result.get("supporting_chunk_ids")
        supporting_ids: set[str] = set()
        if isinstance(raw_supporting_ids, list):
            supporting_ids = {str(sid) for sid in raw_supporting_ids if sid}
        if answerable and supporting_ids:
            verified_chunks = [c for c in ref_chunks if str(c.get("id", "")) in supporting_ids]
            removed_chunks = [c for c in ref_chunks if str(c.get("id", "")) not in supporting_ids]
            # Fall back to all ref_chunks if judge cited IDs we can't match
            if verified_chunks:
                item.qa["reference_chunks"] = cast(list[ReferenceChunk], verified_chunks)
                if removed_chunks:
                    existing = list(item.qa.get("removed_reference_chunks", []))
                    existing.extend(
                        {
                            "chunk": c,
                            "reason": "not_supporting",
                            "filter": "grounding_llm",
                        }
                        for c in removed_chunks
                    )
                    item.qa["removed_reference_chunks"] = existing
            else:
                item.qa["reference_chunks"] = cast(list[ReferenceChunk], list(ref_chunks))
        elif answerable:
            # Judge said answerable but gave no IDs — keep all ref_chunks unchanged
            pass

        is_unsupported = not answerable
        failure_type = _FAILURE_TYPE_UNSUPPORTED if is_unsupported else _FAILURE_TYPE_NONE
        reason_code = "unsupported_or_unanswerable" if is_unsupported else "grounding_supported"
        feedback_type = "reanchor_feedback" if is_unsupported else None
        force_reanchor = bool(is_unsupported)
        refinement_hint = (
            "Reground answer with chunk evidence and revise unsupported facts."
            if is_unsupported
            else None
        )

        metadata = {
            "filter_mode": "grounding_llm",
            "reason_code": reason_code,
            "confidence": confidence,
            "failure_type": failure_type,
            "force_reanchor": force_reanchor,
            "feedback_type": feedback_type,
            "refinement_hint": refinement_hint,
            "judge_answerable": answerable,
            "judge_reasoning": judge_reasoning,
            "supporting_chunk_ids": sorted(supporting_ids),
            "reference_chunk_count": len(ref_chunks),
        }

        if is_unsupported and refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="grounding_filter_needs_refinement",
                reasoning=(
                    "Answer is not fully grounded in available evidence "
                    f"(answerable={answerable}, confidence={confidence:.2f})."
                ),
                metadata=metadata,
            )

        if is_unsupported:
            return FilterVerdict(
                status="rejected",
                reason="grounding_filter_rejected",
                reasoning=(
                    f"Refinement budget exhausted for failure_type={failure_type} "
                    f"({refinements}/{max_refinements})."
                ),
                metadata=metadata,
            )

        return FilterVerdict(
            status="passed",
            reason="grounding_filter_passed",
            reasoning=(
                "Gold answer is fully grounded in combined evidence "
                f"(answerable={answerable}, confidence={confidence:.2f})."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _error_verdict(
        query: str,
        *,
        refinements: int,
        max_refinements: int,
    ) -> FilterVerdict:
        metadata = {
            "filter_mode": "grounding_llm",
            "reason_code": "filter_error",
            "confidence": 0.0,
            "retrieval_query": query,
            "ref_overlap_ratio": 0.0,
            "failure_type": _FAILURE_TYPE_NONE,
            "force_reanchor": False,
            "feedback_type": "retry_feedback",
            "refinement_hint": "Retry due to transient retrieval/filter failure.",
        }
        if refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="grounding_filter_needs_refinement",
                reasoning="Grounding filter error; retrying via refinement.",
                metadata=metadata,
            )
        return FilterVerdict(
            status="rejected",
            reason="grounding_filter_rejected",
            reasoning=(
                "Refinement budget exhausted after grounding filter errors "
                f"({refinements}/{max_refinements})."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _update_stats(stats: dict[str, Any], verdict: FilterVerdict) -> None:
        if verdict.status == "passed":
            stats["passed"] = int(stats.get("passed", 0)) + 1
        elif verdict.status == "needs_refinement":
            stats["needs_refinement"] = int(stats.get("needs_refinement", 0)) + 1
        else:
            stats["rejected"] = int(stats.get("rejected", 0)) + 1
        GroundingLLMFilter._record_diagnostics(stats, verdict)

    @staticmethod
    def _ensure_stats_shape(stats: dict[str, Any]) -> None:
        stats.setdefault("passed", 0)
        stats.setdefault("needs_refinement", 0)
        stats.setdefault("rejected", 0)
        stats.setdefault("errors", 0)
        stats.setdefault("judge_answerable_true", 0)
        stats.setdefault("judge_answerable_false", 0)
        stats.setdefault("unsupported_total", 0)

    @staticmethod
    def _record_diagnostics(stats: dict[str, Any], verdict: FilterVerdict) -> None:
        metadata = verdict.metadata if isinstance(verdict.metadata, dict) else {}
        judge_answerable = metadata.get("judge_answerable")
        if isinstance(judge_answerable, bool):
            if judge_answerable:
                stats["judge_answerable_true"] = int(stats.get("judge_answerable_true", 0)) + 1
            else:
                stats["judge_answerable_false"] = int(stats.get("judge_answerable_false", 0)) + 1
        if str(metadata.get("failure_type", "")).strip() == _FAILURE_TYPE_UNSUPPORTED:
            stats["unsupported_total"] = int(stats.get("unsupported_total", 0)) + 1
