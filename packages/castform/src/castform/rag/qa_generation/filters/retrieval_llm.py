"""Retrieval + LLM-judge too-easy filter for Pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from benchmax.auth import StaticBearerAuth
from openai import AsyncOpenAI

from castform.model_auth import (
    create_async_openai_client,
    create_openai_client,
    model_auth_for_endpoint,
)
from castform.rag.chunkers.models import Chunk
from castform.rag.qa_generation.batch_processor import batch_process_async
from castform.rag.qa_generation.pipeline_config import (
    DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT,
    DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE,
    PipelineContext,
    RetrievalLLMFilterConfig,
)
from castform.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from castform.rag.qa_generation.retrieval_query import (
    QueryRewriteConfig,
    resolve_retrieval_query,
)

logger = logging.getLogger(__name__)

_DUMMY_CHUNK = Chunk(content="", metadata=(("_dummy", True),))


@dataclass
class _RetrievalItemData:
    """Intermediate search data for one item in batch mode."""

    item: GeneratedQA
    query: str
    ref_chunks: list[dict[str, Any]]
    retrieved_chunks: list[Any]
    search_results: list[Any]
    overlap_ratio: float
    overlap_triggered: bool
    too_easy_overlap_triggered: bool
    matched_reference_ids: list[str] = field(default_factory=list)


_FILTER_MODE = "retrieval_too_easy_llm"
_JUDGE_SYSTEM_PROMPT = DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
_JUDGE_USER_TEMPLATE = DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
_JUDGE_TAG_TOO_EASY = "too_easy_lexical"
_JUDGE_TAG_PASS = "challenging_retrieval_pass"
_JUDGE_TAG_UNKNOWN = "unknown"
_FAILURE_TYPE_TOO_EASY = "too_easy"
_FAILURE_TYPE_NONE = "none"


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RetrievalLLMFilter:
    """Flags QA pairs that appear too easy for naive retrieval."""

    def __init__(
        self,
        *,
        chunk_source: Any,
        cfg: RetrievalLLMFilterConfig,
    ) -> None:
        self.chunk_source = chunk_source
        self.cfg = cfg
        self._judge_auth = (
            model_auth_for_endpoint(
                api_key=cfg.judge_api_key,
                base_url=cfg.judge_base_url,
                purpose="retrieval judge",
            )
            if cfg.enabled
            else StaticBearerAuth(cfg.judge_api_key or "disabled")
        )
        self.judge_client = create_openai_client(
            model=cfg.judge_model,
            base_url=cfg.judge_base_url,
            auth=self._judge_auth,
            request_id="retrieval-judge",
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
        stale = (
            client is None
            or client.is_closed()
            or self._async_judge_client_loop is not loop
        )
        if stale:
            client = create_async_openai_client(
                model=self.cfg.judge_model,
                base_url=self.cfg.judge_base_url,
                auth=self._judge_auth,
                request_id="retrieval-judge",
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

    def _resolve_search_mode(self, context: PipelineContext) -> str | None:
        """Use the best search mode available on the corpus backend."""
        profile = getattr(context, "profile", None)
        if profile is None:
            return None
        mode = getattr(profile, "best_search_mode", None)
        if mode and mode != "lexical":
            return mode
        return None

    async def evaluate(
        self, items: list[GeneratedQA], context: PipelineContext
    ) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        self._corpus_language = str(context.get("corpus_language", "") or "").strip()

        max_refinements = context.config.refinement.max_refinements_per_item
        stats_key = (
            str(self.cfg.stats_key or "").strip() or "retrieval_too_easy_filter_stats"
        )
        stats = context.setdefault(
            stats_key,
            {
                "passed": 0,
                "needs_refinement": 0,
                "rejected": 0,
                "errors": 0,
                "judge_calls": 0,
                "judge_answerable_true": 0,
                "judge_answerable_false": 0,
                "judge_reason_tags": {
                    _JUDGE_TAG_TOO_EASY: 0,
                    _JUDGE_TAG_PASS: 0,
                    _JUDGE_TAG_UNKNOWN: 0,
                },
                "too_easy_total": 0,
                "too_easy_due_to_overlap_pre_gate": 0,
                "too_easy_due_to_judge": 0,
                "overlap_threshold_triggered": 0,
                "too_easy_overlap_threshold_triggered": 0,
            },
        )
        self._ensure_stats_shape(stats)
        rewrite_cfg = context.config.filtering.query_rewrite

        search_mode = self._resolve_search_mode(context)

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
                            rewrite_cfg=rewrite_cfg,
                            search_mode=search_mode,
                        )
                except Exception:
                    logger.exception("RetrievalLLMFilter failed for one item")
                    query = resolve_retrieval_query(item.qa, rewrite_cfg=rewrite_cfg)
                    refinements = int(
                        item.generation_metadata.get("refinement_count", 0)
                    )
                    verdict = self._error_verdict(
                        query,
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
            rewrite_cfg=rewrite_cfg,
            search_mode=search_mode,
        )

    async def _evaluate_batch(
        self,
        items: list[GeneratedQA],
        *,
        context: PipelineContext,
        stats: dict[str, Any],
        max_refinements: int,
        rewrite_cfg: QueryRewriteConfig,
        search_mode: str | None = None,
    ) -> list[GeneratedQA]:
        """Batch evaluation: search sequentially, judge in parallel."""
        # Phase 1: search + prep for all eligible items
        prepared: list[_RetrievalItemData] = []
        for item in items:
            if item.filter_verdict is not None and not item.is_passed:
                continue

            query = resolve_retrieval_query(item.qa, rewrite_cfg=rewrite_cfg)
            if not query:
                item.filter_verdict = FilterVerdict(
                    status="rejected",
                    reason="retrieval_filter_rejected",
                    reasoning="Missing question/retrieval_query.",
                    metadata={
                        "filter_mode": _FILTER_MODE,
                        "reason_code": "missing_query",
                        "confidence": 1.0,
                        "retrieval_query": "",
                        "ref_overlap_ratio": 0.0,
                        "failure_type": _FAILURE_TYPE_NONE,
                        "force_reanchor": False,
                        "feedback_type": None,
                        "refinement_hint": None,
                        "judge_called": False,
                    },
                )
                self._update_stats(stats, item.filter_verdict)
                continue

            try:
                async with context.search_semaphore():
                    search_results = await self.chunk_source.search_related(
                        source=_DUMMY_CHUNK,
                        queries=[query],
                        top_k=self.cfg.top_k,
                        mode=search_mode,
                    )
            except Exception:
                logger.warning(
                    "RetrievalLLMFilter search failed for one item; routing to refinement/rejection.",
                    exc_info=True,
                )
                refinements = int(item.generation_metadata.get("refinement_count", 0))
                item.filter_verdict = self._error_verdict(
                    query,
                    refinements=refinements,
                    max_refinements=max_refinements,
                )
                stats["errors"] = int(stats.get("errors", 0)) + 1
                self._update_stats(stats, item.filter_verdict)
                continue
            retrieved_chunks: list[Chunk] = [
                row["chunk"] for row in search_results if "chunk" in row
            ]
            ref_chunks = list(item.qa.get("reference_chunks", []) or [])
            overlap_ratio, matched_reference_ids = self._compute_overlap(
                ref_chunks, retrieved_chunks
            )
            overlap_triggered = overlap_ratio >= self.cfg.overlap_threshold
            too_easy_overlap_triggered = (
                overlap_ratio >= self.cfg.too_easy_overlap_threshold
            )
            prepared.append(
                _RetrievalItemData(
                    item=item,
                    query=query,
                    ref_chunks=ref_chunks,
                    retrieved_chunks=retrieved_chunks,
                    search_results=search_results,
                    overlap_ratio=overlap_ratio,
                    overlap_triggered=overlap_triggered,
                    too_easy_overlap_triggered=too_easy_overlap_triggered,
                    matched_reference_ids=matched_reference_ids,
                )
            )

        if not prepared:
            return items

        # Phase 2: batch LLM judge calls for items that pass the overlap pre-gate
        judge_prompts: list[str] = []
        needs_judge_indices: list[int] = []  # indices into `prepared`
        for i, data in enumerate(prepared):
            if data.too_easy_overlap_triggered:
                continue  # verdict determined without judge
            prompt = self._build_judge_prompt(data.item, data.retrieved_chunks)
            if prompt is not None:
                judge_prompts.append(prompt)
                needs_judge_indices.append(i)

        judge_results: dict[int, dict[str, Any]] = {}
        if judge_prompts:
            system_prompt = self.cfg.judge_system_prompt or _JUDGE_SYSTEM_PROMPT
            corpus_language = getattr(self, "_corpus_language", "")
            if corpus_language:
                system_prompt = (
                    f"NOTE: The question, answer, and chunks are in {corpus_language}. "
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
                desc="Retrieval filter",
                semaphore=context.model_semaphore(
                    self.cfg.judge_model, self.cfg.judge_base_url
                ),
            )
            for j, i in enumerate(needs_judge_indices):
                response = batch_result.responses[j]
                if response is None:
                    judge_results[i] = {
                        "answerable": False,
                        "confidence": 0.0,
                        "reasoning": "batch_call_failed",
                        "reason_tag": _JUDGE_TAG_UNKNOWN,
                    }
                else:
                    judge_results[i] = self._parse_judge_response(
                        response.answer or "{}"
                    )

        # Phase 3: apply verdicts
        no_chunks_result: dict[str, Any] = {
            "answerable": False,
            "confidence": 1.0,
            "reasoning": "No chunks retrieved.",
            "reason_tag": _JUDGE_TAG_UNKNOWN,
        }
        for i, data in enumerate(prepared):
            try:
                refinements = int(
                    data.item.generation_metadata.get("refinement_count", 0)
                )
                shared_metadata = {
                    "filter_mode": _FILTER_MODE,
                    "search_mode": search_mode or "default",
                    "retrieval_query": data.query,
                    "ref_overlap_ratio": data.overlap_ratio,
                    "overlap_threshold": self.cfg.overlap_threshold,
                    "overlap_triggered": data.overlap_triggered,
                    "too_easy_overlap_threshold": self.cfg.too_easy_overlap_threshold,
                    "too_easy_overlap_triggered": data.too_easy_overlap_triggered,
                    "too_easy_confidence_threshold": self.cfg.too_easy_confidence_threshold,
                    "force_reanchor": False,
                    "matched_reference_ids": data.matched_reference_ids,
                    "matched_reference_chunks": len(data.matched_reference_ids),
                    "retrieved_chunk_count": len(data.retrieved_chunks),
                    "top_score": (
                        data.search_results[0].get("max_score", 0.0)
                        if data.search_results
                        else 0.0
                    ),
                }

                if data.too_easy_overlap_triggered:
                    metadata = {
                        **shared_metadata,
                        "reason_code": "naive_retrieval_sufficient",
                        "confidence": 1.0,
                        "failure_type": _FAILURE_TYPE_TOO_EASY,
                        "feedback_type": "same_anchor_feedback",
                        "refinement_hint": "Increase retrieval difficulty and avoid directly retrievable terms.",
                        "judge_called": False,
                        "judge_answerable": None,
                        "judge_reasoning": "",
                        "judge_reason_tag": _JUDGE_TAG_TOO_EASY,
                        "too_easy_source": "overlap_pre_gate",
                    }
                    verdict = self._too_easy_verdict(
                        metadata=metadata,
                        overlap_ratio=data.overlap_ratio,
                        refinements=refinements,
                        max_refinements=max_refinements,
                    )
                else:
                    judge_result = judge_results.get(i, no_chunks_result)
                    verdict = self._verdict_from_judge_result(
                        data.item,
                        judge_result=judge_result,
                        shared_metadata=shared_metadata,
                        refinements=refinements,
                        max_refinements=max_refinements,
                        overlap_ratio=data.overlap_ratio,
                    )
            except Exception:
                logger.exception("RetrievalLLMFilter failed for one item")
                refinements = int(
                    data.item.generation_metadata.get("refinement_count", 0)
                )
                verdict = self._error_verdict(
                    data.query,
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
        rewrite_cfg: QueryRewriteConfig,
        search_mode: str | None = None,
    ) -> FilterVerdict:
        query = resolve_retrieval_query(item.qa, rewrite_cfg=rewrite_cfg)
        if not query:
            return FilterVerdict(
                status="rejected",
                reason="retrieval_filter_rejected",
                reasoning="Missing question/retrieval_query.",
                metadata={
                    "filter_mode": _FILTER_MODE,
                    "reason_code": "missing_query",
                    "confidence": 1.0,
                    "retrieval_query": "",
                    "ref_overlap_ratio": 0.0,
                    "failure_type": _FAILURE_TYPE_NONE,
                    "force_reanchor": False,
                    "feedback_type": None,
                    "refinement_hint": None,
                    "judge_called": False,
                },
            )

        search_results = self.chunk_source.search_related(
            source=_DUMMY_CHUNK,
            queries=[query],
            top_k=self.cfg.top_k,
            mode=search_mode,
        )
        retrieved_chunks: list[Chunk] = [
            row["chunk"] for row in search_results if "chunk" in row
        ]
        ref_chunks = list(item.qa.get("reference_chunks", []) or [])
        overlap_ratio, matched_reference_ids = self._compute_overlap(
            ref_chunks, retrieved_chunks
        )
        overlap_triggered = overlap_ratio >= self.cfg.overlap_threshold
        too_easy_overlap_triggered = (
            overlap_ratio >= self.cfg.too_easy_overlap_threshold
        )
        refinements = int(item.generation_metadata.get("refinement_count", 0))

        shared_metadata = {
            "filter_mode": _FILTER_MODE,
            "search_mode": search_mode or "default",
            "retrieval_query": query,
            "ref_overlap_ratio": overlap_ratio,
            "overlap_threshold": self.cfg.overlap_threshold,
            "overlap_triggered": overlap_triggered,
            "too_easy_overlap_threshold": self.cfg.too_easy_overlap_threshold,
            "too_easy_overlap_triggered": too_easy_overlap_triggered,
            "too_easy_confidence_threshold": self.cfg.too_easy_confidence_threshold,
            "force_reanchor": False,
            "matched_reference_ids": matched_reference_ids,
            "matched_reference_chunks": len(matched_reference_ids),
            "retrieved_chunk_count": len(retrieved_chunks),
            "top_score": search_results[0].get("max_score", 0.0)
            if search_results
            else 0.0,
        }

        if too_easy_overlap_triggered:
            metadata = {
                **shared_metadata,
                "reason_code": "naive_retrieval_sufficient",
                "confidence": 1.0,
                "failure_type": _FAILURE_TYPE_TOO_EASY,
                "feedback_type": "same_anchor_feedback",
                "refinement_hint": "Increase retrieval difficulty and avoid directly retrievable terms.",
                "judge_called": False,
                "judge_answerable": None,
                "judge_reasoning": "",
                "judge_reason_tag": _JUDGE_TAG_TOO_EASY,
                "too_easy_source": "overlap_pre_gate",
            }
            return self._too_easy_verdict(
                metadata=metadata,
                overlap_ratio=overlap_ratio,
                refinements=refinements,
                max_refinements=max_refinements,
            )

        judge_result = self._run_judge(item, retrieved_chunks)
        return self._verdict_from_judge_result(
            item,
            judge_result=judge_result,
            shared_metadata=shared_metadata,
            refinements=refinements,
            max_refinements=max_refinements,
            overlap_ratio=overlap_ratio,
        )

    @staticmethod
    def _too_easy_verdict(
        *,
        metadata: dict[str, Any],
        overlap_ratio: float,
        refinements: int,
        max_refinements: int,
    ) -> FilterVerdict:
        if refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="retrieval_filter_needs_refinement",
                reasoning=(
                    "Naive retrieval appears sufficient "
                    f"(overlap={overlap_ratio:.2f}, source={metadata.get('too_easy_source', 'unknown')})."
                ),
                metadata=metadata,
            )

        return FilterVerdict(
            status="rejected",
            reason="retrieval_filter_rejected",
            reasoning=(
                "Refinement budget exhausted for failure_type=too_easy "
                f"({refinements}/{max_refinements})."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _compute_overlap(
        ref_chunks: list[dict[str, Any]],
        retrieved_chunks: list[Chunk],
    ) -> tuple[float, list[str]]:
        if not ref_chunks:
            return 0.0, []
        retrieved_hashes = {chunk.hash for chunk in retrieved_chunks}
        matched: list[str] = []
        for ref in ref_chunks:
            ref_id = str(ref.get("id", "")).strip()
            if ref_id and ref_id in retrieved_hashes:
                matched.append(ref_id)
        return len(matched) / max(1, len(ref_chunks)), matched

    def _build_judge_prompt(
        self, item: GeneratedQA, retrieved_chunks: list[Chunk]
    ) -> str | None:
        """Build judge user prompt. Returns None if no chunks (no judge call needed)."""
        if not retrieved_chunks:
            return None
        chunks_text = "\n---\n".join(
            f"[Chunk {idx + 1}]\n{chunk.content}"
            for idx, chunk in enumerate(retrieved_chunks)
        )
        prompt_vars = {
            "question": item.qa.get("question", ""),
            "answer": item.qa.get("answer", ""),
            "chunks_text": chunks_text,
        }
        user_template = (
            str(self.cfg.judge_user_template or "").strip() or _JUDGE_USER_TEMPLATE
        )
        try:
            return user_template.format(**prompt_vars)
        except KeyError:
            logger.warning(
                "Retrieval judge user template requires unknown placeholders; "
                "falling back to default template."
            )
            return _JUDGE_USER_TEMPLATE.format(**prompt_vars)

    def _parse_judge_response(self, raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {
                    "answerable": False,
                    "confidence": 0.0,
                    "reasoning": "parse_error",
                    "reason_tag": _JUDGE_TAG_UNKNOWN,
                }
            raw_tag = str(payload.get("reason_tag", "")).strip()
            payload["reason_tag_raw"] = raw_tag
            payload["reason_tag"] = self._extract_judge_reason_tag(
                reason_tag=raw_tag,
                reasoning=payload.get("reasoning", ""),
            )
            return payload
        except json.JSONDecodeError:
            logger.warning("Judge response parse failure: %s", raw[:240])
            return {
                "answerable": False,
                "confidence": 0.0,
                "reasoning": "parse_error",
                "reason_tag": _JUDGE_TAG_UNKNOWN,
            }

    def _run_judge(
        self, item: GeneratedQA, retrieved_chunks: list[Chunk]
    ) -> dict[str, Any]:
        prompt = self._build_judge_prompt(item, retrieved_chunks)
        if prompt is None:
            return {
                "answerable": False,
                "confidence": 1.0,
                "reasoning": "No chunks retrieved.",
                "reason_tag": _JUDGE_TAG_UNKNOWN,
            }

        system_prompt = self.cfg.judge_system_prompt or _JUDGE_SYSTEM_PROMPT
        corpus_language = getattr(self, "_corpus_language", "")
        if corpus_language:
            system_prompt = (
                f"NOTE: The question, answer, and chunks are in {corpus_language}. "
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
        shared_metadata: dict[str, Any],
        refinements: int,
        max_refinements: int,
        overlap_ratio: float,
    ) -> FilterVerdict:
        answerable = bool(judge_result.get("answerable", False))
        confidence = _safe_float(judge_result.get("confidence", 0.0), default=0.0)
        judge_reasoning = str(judge_result.get("reasoning", "")).strip()
        reason_tag_raw = str(judge_result.get("reason_tag_raw", "")).strip()
        judge_reason_tag = self._extract_judge_reason_tag(
            reason_tag=judge_result.get("reason_tag", ""),
            reasoning=judge_reasoning,
        )
        lexical_anchor_evidence = (
            str(judge_result.get("lexical_anchor_evidence", "") or "").strip() or None
        )
        too_easy_high_confidence = confidence >= self.cfg.too_easy_confidence_threshold
        too_easy_due_to_judge = (
            judge_reason_tag == _JUDGE_TAG_TOO_EASY and too_easy_high_confidence
        )

        # Discount the judge verdict when reference chunks weren't retrieved.
        # If BM25 couldn't find the reference chunks (overlap=0), the question
        # isn't "too easy" regardless of surface keyword overlap in other chunks.
        discounted_no_ref_overlap = too_easy_due_to_judge and overlap_ratio == 0.0
        if discounted_no_ref_overlap:
            too_easy_due_to_judge = False

        too_easy_calibrated_out = (
            judge_reason_tag == _JUDGE_TAG_TOO_EASY and not too_easy_due_to_judge
        )

        if too_easy_due_to_judge:
            metadata = {
                **shared_metadata,
                "reason_code": "naive_retrieval_sufficient",
                "confidence": confidence,
                "failure_type": _FAILURE_TYPE_TOO_EASY,
                "feedback_type": "same_anchor_feedback",
                "refinement_hint": "Increase retrieval difficulty and avoid directly retrievable terms.",
                "judge_called": True,
                "judge_answerable": answerable,
                "judge_reasoning": judge_reasoning,
                "judge_reason_tag": judge_reason_tag,
                "judge_reason_tag_raw": reason_tag_raw,
                "judge_lexical_anchor_evidence": lexical_anchor_evidence,
                "too_easy_high_confidence": too_easy_high_confidence,
                "too_easy_calibrated_out": False,
                "too_easy_source": "judge",
            }
            return self._too_easy_verdict(
                metadata=metadata,
                overlap_ratio=overlap_ratio,
                refinements=refinements,
                max_refinements=max_refinements,
            )

        if discounted_no_ref_overlap:
            reasoning = (
                "Judge flagged too_easy_lexical but reference chunks were not in "
                f"BM25 results (overlap={overlap_ratio:.2f}); verdict discounted. "
                f"(confidence={confidence:.2f})."
            )
        elif too_easy_calibrated_out:
            reasoning = (
                "Borderline too-easy signal did not meet confidence threshold. "
                f"(overlap={overlap_ratio:.2f}, confidence={confidence:.2f})."
            )
        else:
            reasoning = (
                "Not classified as too easy by overlap gate or judge. "
                f"(overlap={overlap_ratio:.2f}, confidence={confidence:.2f})."
            )

        return FilterVerdict(
            status="passed",
            reason="retrieval_filter_passed",
            reasoning=reasoning,
            metadata={
                **shared_metadata,
                "reason_code": "retrieval_difficulty_passed",
                "confidence": confidence,
                "failure_type": _FAILURE_TYPE_NONE,
                "feedback_type": None,
                "refinement_hint": None,
                "judge_called": True,
                "judge_answerable": answerable,
                "judge_reasoning": judge_reasoning,
                "judge_reason_tag": judge_reason_tag,
                "judge_reason_tag_raw": reason_tag_raw,
                "judge_lexical_anchor_evidence": lexical_anchor_evidence,
                "too_easy_high_confidence": too_easy_high_confidence,
                "too_easy_calibrated_out": too_easy_calibrated_out,
                "discounted_no_ref_overlap": discounted_no_ref_overlap,
                "too_easy_source": "none",
            },
        )

    @staticmethod
    def _error_verdict(
        query: str,
        *,
        refinements: int,
        max_refinements: int,
    ) -> FilterVerdict:
        metadata = {
            "filter_mode": _FILTER_MODE,
            "reason_code": "filter_error",
            "confidence": 0.0,
            "retrieval_query": query,
            "ref_overlap_ratio": 0.0,
            "failure_type": _FAILURE_TYPE_NONE,
            "force_reanchor": False,
            "feedback_type": "retry_feedback",
            "refinement_hint": "Retry due to transient retrieval/filter failure.",
            "judge_called": False,
        }
        if refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="retrieval_filter_needs_refinement",
                reasoning="Retrieval/filter error; retrying via refinement.",
                metadata=metadata,
            )
        return FilterVerdict(
            status="rejected",
            reason="retrieval_filter_rejected",
            reasoning=(
                "Refinement budget exhausted after retrieval/filter errors "
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
        RetrievalLLMFilter._record_diagnostics(stats, verdict)

    @staticmethod
    def _extract_judge_reason_tag(*, reason_tag: Any = "", reasoning: Any = "") -> str:
        explicit = str(reason_tag or "").strip().lower()
        if explicit in {_JUDGE_TAG_TOO_EASY, "too_easy"}:
            return _JUDGE_TAG_TOO_EASY
        # Anything that isn't too_easy is a pass — including "unsupported"
        # (means BM25 couldn't retrieve the right chunks = retrieval is hard)
        # and old tag names like "challenging_answerable_pass".
        if explicit and explicit != _JUDGE_TAG_UNKNOWN:
            return _JUDGE_TAG_PASS

        # Fallback: scan reasoning text for known tag values.
        text = str(reasoning or "").strip().lower()
        if "too_easy" in text:
            return _JUDGE_TAG_TOO_EASY
        if text:
            return _JUDGE_TAG_PASS
        return _JUDGE_TAG_UNKNOWN

    @staticmethod
    def _ensure_stats_shape(stats: dict[str, Any]) -> None:
        stats.setdefault("passed", 0)
        stats.setdefault("needs_refinement", 0)
        stats.setdefault("rejected", 0)
        stats.setdefault("errors", 0)
        stats.setdefault("judge_calls", 0)
        stats.setdefault("judge_answerable_true", 0)
        stats.setdefault("judge_answerable_false", 0)
        stats.setdefault("too_easy_total", 0)
        stats.setdefault("too_easy_due_to_overlap_pre_gate", 0)
        stats.setdefault("too_easy_due_to_judge", 0)
        stats.setdefault("overlap_threshold_triggered", 0)
        stats.setdefault("too_easy_overlap_threshold_triggered", 0)

        tags = stats.setdefault("judge_reason_tags", {})
        if not isinstance(tags, dict):
            stats["judge_reason_tags"] = {}
            tags = stats["judge_reason_tags"]
        tags.setdefault(_JUDGE_TAG_TOO_EASY, 0)
        tags.setdefault(_JUDGE_TAG_PASS, 0)
        tags.setdefault(_JUDGE_TAG_UNKNOWN, 0)

    @staticmethod
    def _record_diagnostics(stats: dict[str, Any], verdict: FilterVerdict) -> None:
        metadata = verdict.metadata if isinstance(verdict.metadata, dict) else {}

        overlap_triggered = bool(metadata.get("overlap_triggered", False))
        if overlap_triggered:
            stats["overlap_threshold_triggered"] = (
                int(stats.get("overlap_threshold_triggered", 0)) + 1
            )
        too_easy_overlap_triggered = bool(
            metadata.get("too_easy_overlap_triggered", False)
        )
        if too_easy_overlap_triggered:
            stats["too_easy_overlap_threshold_triggered"] = (
                int(stats.get("too_easy_overlap_threshold_triggered", 0)) + 1
            )

        judge_called = bool(metadata.get("judge_called", False))
        if judge_called:
            stats["judge_calls"] = int(stats.get("judge_calls", 0)) + 1
            judge_answerable = metadata.get("judge_answerable")
            if isinstance(judge_answerable, bool):
                if judge_answerable:
                    stats["judge_answerable_true"] = (
                        int(stats.get("judge_answerable_true", 0)) + 1
                    )
                else:
                    stats["judge_answerable_false"] = (
                        int(stats.get("judge_answerable_false", 0)) + 1
                    )

            judge_reason_tag = (
                str(metadata.get("judge_reason_tag", "")).strip().lower()
                or _JUDGE_TAG_UNKNOWN
            )
            tags = stats.get("judge_reason_tags", {})
            if isinstance(tags, dict):
                if judge_reason_tag not in tags:
                    tags[judge_reason_tag] = 0
                tags[judge_reason_tag] = int(tags.get(judge_reason_tag, 0)) + 1

        failure_type = str(metadata.get("failure_type", "")).strip()
        if failure_type == _FAILURE_TYPE_TOO_EASY:
            stats["too_easy_total"] = int(stats.get("too_easy_total", 0)) + 1
            source = str(metadata.get("too_easy_source", "")).strip()
            if source == "overlap_pre_gate":
                stats["too_easy_due_to_overlap_pre_gate"] = (
                    int(stats.get("too_easy_due_to_overlap_pre_gate", 0)) + 1
                )
            elif source == "judge":
                stats["too_easy_due_to_judge"] = (
                    int(stats.get("too_easy_due_to_judge", 0)) + 1
                )
