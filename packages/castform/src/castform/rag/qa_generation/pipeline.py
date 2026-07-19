"""Unified QA generation pipeline."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from tqdm.auto import tqdm as _tqdm

from castform.rag.corpus.postgres.source import PostgresChunkSource
from castform.rag.qa_generation.auto_tune import (
    auto_tune,
    compute_batch_heuristics,
    emit_corpus_warnings,
    should_early_stop,
)
from castform.rag.qa_generation.pipeline_config import (
    PipelineContext,
    PipelineConfig,
    RunStats,
    GenerationTask,
    load_pipeline_config,
)
from castform.rag.qa_generation.corpus_profile import (
    CorpusProfile,
    _get_doc_id,
    compute_chunk_suitability,
    compute_header_prevalence,
    compute_metadata_census,
    compute_token_document_frequency,
    detect_search_capabilities,
    diverse_profile_sample,
    select_diverse,
)
from castform.rag.qa_generation.filters import (
    DeterministicGuardsFilter,
    GroundingLLMFilter,
    QualityGateFilter,
    RetrievalLLMFilter,
)
from castform.rag.qa_generation.formatters import TrainEvalFormatter
from castform.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from castform.rag.qa_generation.generators import DirectLLMGenerator
from castform.rag.qa_generation.helpers import render_template
from castform.rag.qa_generation.metadata_linker import MetadataChunkLinker
from castform.rag.qa_generation.metrics import (
    BatchMetrics,
    PipelineMetrics,
    stage_timer,
)
from castform.rag.qa_generation.protocols import (
    ChunkLinker,
    EvaluatorFilter,
    QuestionGenerator,
    QuestionTransformer,
)
from castform.rag.qa_generation.response_parsers import parse_corpus_summary_response
from castform.rag.qa_generation.scoring import (
    compute_eval_scores,
    extract_filter_scores,
)
from castform.rag.qa_generation.transformers import BaseQuestionTransformer
from castform.rag.qa_generation.transformers.dedup import IncrementalDeduplicator
from castform.platform.client import RolloutClient
from castform.platform.credentials import resolve_judge_key, resolve_token_provider

logger = logging.getLogger(__name__)

_QUALITY_GATE_FILTER_STAGE = "quality_gate"
_RETRIEVAL_TOO_EASY_FILTER_STAGE = "retrieval_too_easy_llm"
_GROUNDING_FILTER_STAGE = "grounding_llm"
_HOP_COUNT_VALIDITY_FILTER_STAGE = "hop_count_validity"
_ENV_ROLLOUT_FILTER_STAGE = "env_rollout"
_SUPPORTED_FILTER_STAGES = (
    _QUALITY_GATE_FILTER_STAGE,
    _GROUNDING_FILTER_STAGE,
    _RETRIEVAL_TOO_EASY_FILTER_STAGE,
    _HOP_COUNT_VALIDITY_FILTER_STAGE,
    _ENV_ROLLOUT_FILTER_STAGE,
)


def _build_openai_client(*, api_key: str, base_url: str) -> OpenAI:
    # An empty key resolves the platform bearer per the credential seam (cached
    # session / ACT_AS_TOKEN_PATH / PLATFORM_API_KEY); an explicit key wins.
    return OpenAI(api_key=resolve_judge_key(api_key, base_url), base_url=base_url)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    # BadRequestError is a subclass of APIError — check it first so the specific
    # logic below takes precedence over the broad APIError fallthrough.
    if isinstance(exc, BadRequestError):
        code = str(getattr(exc, "code", "") or "").strip().lower()
        body = str(getattr(exc, "body", "") or "").strip().lower()
        message = str(exc).strip().lower()
        return (
            code == "upstream_error"
            or "internal error" in body
            or "internal error" in message
        )
    if isinstance(exc, APIError):
        return True
    return False


def _chat_completion_with_retry(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, str] | None = None,
    temperature: float | None = None,
    max_attempts: int = 3,
    initial_delay_seconds: float = 0.75,
) -> Any:
    delay = max(0.1, float(initial_delay_seconds))
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if temperature is not None:
                kwargs["temperature"] = temperature
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_openai_error(exc):
                raise
            logger.warning(
                "Transient LLM error on attempt %d/%d; retrying in %.2fs: %s",
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
            delay *= 2.0
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Completion retry loop exited unexpectedly.")


def _load_source(cfg: PipelineConfig) -> PostgresChunkSource:
    source = PostgresChunkSource(
        api_key=cfg.platform.api_key,
        corpus_name=cfg.corpus.corpus_name,
        base_url=cfg.platform.base_url,
    )
    if cfg.corpus.docs_path:
        source.populate_from_folder(
            cfg.corpus.docs_path, show_summary=cfg.corpus.show_summary
        )
        return source
    if cfg.corpus.corpus_id:
        source.populate_from_existing_corpus(
            cfg.corpus.corpus_id, show_summary=cfg.corpus.show_summary
        )
        return source
    source.populate_from_existing_corpus_name(
        cfg.corpus.corpus_name,
        show_summary=cfg.corpus.show_summary,
    )
    return source


def _build_rollout_client(cfg: PipelineConfig) -> RolloutClient:
    return RolloutClient(api_key=cfg.platform.api_key)


def _build_metadata_linker(
    cfg: PipelineConfig,
    source: Any,
    profile: CorpusProfile,
    wiki_index: Any = None,
) -> MetadataChunkLinker:
    from castform.rag.qa_generation.metadata_linker import MetadataLinkerConfig

    mcfg = cfg.linker.metadata
    return MetadataChunkLinker(
        source=source,
        profile=profile,
        config=MetadataLinkerConfig(
            max_candidates=mcfg.max_candidates,
            max_secondaries=mcfg.max_secondaries,
            min_chunk_chars=mcfg.min_chunk_chars,
            filter_same_file=mcfg.filter_same_file,
            min_coherence=mcfg.min_coherence,
            max_secondary_similarity=mcfg.max_secondary_similarity,
            max_primary_similarity=mcfg.max_primary_similarity,
            retry_confidence=mcfg.retry_confidence,
            header_keys=mcfg.header_keys,
        ),
        wiki_index=wiki_index,
    )


def _build_linker(
    cfg: PipelineConfig,
    source: Any,
    profile: CorpusProfile | None = None,
    wiki_index: Any = None,
) -> ChunkLinker:
    if cfg.linker.type not in ("metadata", "search_agent", "wiki"):
        logger.warning(
            "Unknown linker type '%s'; falling back to 'metadata'.",
            cfg.linker.type,
        )
    if profile is None:
        raise ValueError(
            "Linker requires a CorpusProfile but none is available "
            "(corpus_context must be enabled in stages 2/3)."
        )

    if cfg.linker.type == "wiki":
        if not profile.entity_chunk_index:
            logger.warning(
                "Wiki linker requested but entity-chunk graph is empty; "
                "falling back to metadata linker.",
            )
            return _build_metadata_linker(cfg, source, profile, wiki_index=wiki_index)
        from castform.rag.qa_generation.wiki_chunk_linker import (  # noqa: PLC0415
            WikiChunkLinker,
            WikiChunkLinkerConfig,
        )

        mcfg = cfg.linker.metadata
        return WikiChunkLinker(
            source=source,
            profile=profile,
            config=WikiChunkLinkerConfig(
                max_secondaries=mcfg.max_secondaries,
                min_chunk_chars=mcfg.min_chunk_chars,
                max_primary_similarity=mcfg.max_primary_similarity,
            ),
        )

    if cfg.linker.type == "search_agent":
        from castform.rag.corpus.postgres.search import PostgresSearch
        from castform.rag.qa_generation.search_agent_linker import SearchAgentLinker

        metadata_linker = _build_metadata_linker(
            cfg, source, profile, wiki_index=wiki_index
        )
        search_client = PostgresSearch(
            corpus_name=cfg.corpus.corpus_name,
            base_url=cfg.platform.base_url,
            corpus_id=cfg.corpus.corpus_id or None,
            # Explicit key wins; empty → the credential seam (cached session).
            token_provider=resolve_token_provider(cfg.platform.api_key),
        )
        llm_cfg = cfg.generation.llm_direct
        return SearchAgentLinker(
            metadata_linker=metadata_linker,
            source=source,
            cfg=cfg.linker.search_agent,
            search_agent_pct=cfg.linker.search_agent_pct,
            rollout_client=_build_rollout_client(cfg),
            search_client=search_client,
            llm_model=llm_cfg.model,
            llm_base_url=llm_cfg.base_url,
            llm_api_key=llm_cfg.api_key,
        )

    return _build_metadata_linker(cfg, source, profile, wiki_index=wiki_index)


def _build_generator(
    cfg: PipelineConfig,
    *,
    linker: ChunkLinker,
) -> QuestionGenerator:
    generation_cfg = cfg.generation.llm_direct
    generation_client = _build_openai_client(
        api_key=generation_cfg.api_key,
        base_url=generation_cfg.base_url,
    )
    return DirectLLMGenerator(
        client=generation_client,
        linker=linker,
        cfg=generation_cfg,
    )


def _build_transformer(cfg: PipelineConfig) -> QuestionTransformer:
    del cfg
    return BaseQuestionTransformer()


def _build_filter_from_stage_name(
    stage_name: str,
    cfg: PipelineConfig,
    *,
    source: Any,
    rollout_client_factory: Callable[[PipelineConfig], RolloutClient] | None = None,
) -> EvaluatorFilter:
    stage = str(stage_name or "").strip().lower()
    if stage == _QUALITY_GATE_FILTER_STAGE:
        return QualityGateFilter(cfg.filtering.quality_gate)
    if stage == _GROUNDING_FILTER_STAGE:
        return GroundingLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.grounding_llm,
        )
    if stage == _RETRIEVAL_TOO_EASY_FILTER_STAGE:
        return RetrievalLLMFilter(
            chunk_source=source,
            cfg=cfg.filtering.retrieval_llm,
        )
    if stage == _HOP_COUNT_VALIDITY_FILTER_STAGE:
        from castform.rag.qa_generation.filters.hop_count_validity import (
            HopCountValidityConfig,
            HopCountValidityFilter,
        )

        hcfg = cfg.filtering.hop_count_validity
        return HopCountValidityFilter(
            cfg=HopCountValidityConfig(
                enabled=hcfg.enabled,
                mode=hcfg.mode,
                max_judge_calls=hcfg.max_judge_calls,
                judge_model=hcfg.judge_model,
                judge_api_key=hcfg.judge_api_key,
                judge_base_url=hcfg.judge_base_url,
                max_concurrent=hcfg.max_concurrent,
                batch_enabled=hcfg.batch_enabled,
                show_batch_progress=hcfg.show_batch_progress,
                stats_key=hcfg.stats_key,
                lopsided_high_threshold=hcfg.lopsided_high_threshold,
                lopsided_low_threshold=hcfg.lopsided_low_threshold,
            ),
        )
    if stage == _ENV_ROLLOUT_FILTER_STAGE:
        from castform.rag.qa_generation.filters.env_rollout import EnvRolloutFilter

        if rollout_client_factory is None:
            raise ValueError(
                "env_rollout filter requires a rollout_client_factory. "
                "Pass one when constructing Pipeline."
            )
        return EnvRolloutFilter(
            rollout_client=rollout_client_factory(cfg),
            cfg=cfg.filtering.env_rollout,
        )
    raise ValueError(
        f"Unknown filter stage '{stage_name}'. "
        f"Supported filter stage names: {', '.join(_SUPPORTED_FILTER_STAGES)}."
    )


def _build_filter_chain(
    cfg: PipelineConfig,
    *,
    source: Any,
    rollout_client_factory: Callable[[PipelineConfig], RolloutClient] | None = None,
) -> tuple[list[str], list[EvaluatorFilter]]:
    chain_names = [
        str(name).strip().lower()
        for name in (cfg.filtering.filters or [])
        if str(name).strip()
    ]
    filters = [
        _build_filter_from_stage_name(
            stage_name,
            cfg,
            source=source,
            rollout_client_factory=rollout_client_factory,
        )
        for stage_name in chain_names
    ]
    return chain_names, filters


def _mark_rejected(
    items: list[GeneratedQA], *, reason: str, reason_code: str
) -> list[GeneratedQA]:
    for item in items:
        _reject_item(item, reason=reason, reason_code=reason_code)
        qa_type = str(item.qa.get("qa_type", "")).strip()
        item.append_journey_event(
            stage="pipeline",
            event_type="filter_rejected",
            qa_type_before=qa_type,
            qa_type_after=qa_type,
            reason_code=reason_code,
        )
    return items


def _serialize_qa_with_filter_details(item: GeneratedQA) -> dict[str, Any]:
    row = dict(item.qa)
    row["journey_events"] = list(item.journey_events)
    row["generation_metadata"] = dict(item.generation_metadata)
    verdict = item.filter_verdict
    if verdict is None:
        return row
    row["filter_status"] = verdict.status
    row["filter_reason"] = verdict.reason
    row["filter_reasoning"] = verdict.reasoning
    row["filter_metadata"] = (
        dict(verdict.metadata) if isinstance(verdict.metadata, dict) else {}
    )
    return row


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _reject_item(
    item: GeneratedQA,
    *,
    reason: str,
    reason_code: str,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Set a rejection verdict while preserving the original verdict metadata."""
    original_verdict = item.filter_verdict
    original_meta: dict[str, Any] = {}
    if original_verdict is not None:
        original_meta = {
            "original_filter_status": original_verdict.status,
            "original_filter_reason": original_verdict.reason,
            "original_filter_reasoning": original_verdict.reasoning,
            "original_filter_metadata": (
                dict(original_verdict.metadata)
                if isinstance(original_verdict.metadata, dict)
                else {}
            ),
        }
    extra = dict(metadata_extra) if isinstance(metadata_extra, dict) else {}
    item.filter_verdict = FilterVerdict(
        status="rejected",
        reason=reason,
        reasoning=reason,
        metadata={
            "filter_mode": "pipeline",
            "reason_code": reason_code,
            **original_meta,
            **extra,
        },
    )


def _reason_code_from_verdict(verdict: FilterVerdict | None) -> str:
    if verdict is None:
        return ""
    metadata = dict(verdict.metadata) if isinstance(verdict.metadata, dict) else {}
    return str(metadata.get("reason_code", "")).strip() or str(verdict.reason).strip()


def _resolve_effective_qa_type(item: GeneratedQA) -> tuple[str, bool, str]:
    """Return the effective QA type implied by the item's reference structure.

    Delegates to ``GeneratedQA.resolve_effective_qa_type`` for the core logic
    and adds relabel metadata (was_relabeled flag and direction string).
    """
    original_type = str(item.qa.get("qa_type", "")).strip().lower()
    if not original_type:
        original_type = (
            str(item.generation_metadata.get("qa_type_target", "")).strip().lower()
            or "lookup"
        )
    effective_type = item.resolve_effective_qa_type()
    if effective_type != original_type:
        direction = f"{original_type}_to_{effective_type}"
        return effective_type, True, direction
    return effective_type, False, ""


def _compute_target_type_counts(cfg: PipelineConfig) -> dict[str, int]:
    from castform.rag.qa_generation.pipeline_config import (
        allocate_largest_remainder_generic,
    )

    counts = allocate_largest_remainder_generic(
        cfg.targets.total_samples,
        cfg.targets.primary_type_distribution,
    )
    counts.setdefault("lookup", 0)
    counts.setdefault("multi_hop", 0)
    return {str(k): int(v) for k, v in counts.items()}


def _count_items_by_effective_type(items: list[GeneratedQA]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        qa_type, _, _ = _resolve_effective_qa_type(item)
        counts[qa_type] += 1
    return dict(counts)


def _record_filter_events(items: list[GeneratedQA], *, event_type: str) -> None:
    for item in items:
        qa_type = str(item.qa.get("qa_type", "")).strip()
        item.append_journey_event(
            stage="filter",
            event_type=event_type,
            qa_type_before=qa_type,
            qa_type_after=qa_type,
            reason_code=_reason_code_from_verdict(item.filter_verdict),
            details={
                "filter_reason": str(
                    item.filter_verdict.reason if item.filter_verdict else ""
                ).strip(),
            },
        )


def _annotate_generated_items(items: list[GeneratedQA]) -> None:
    for item in items:
        qa_type = (
            str(item.generation_metadata.get("qa_type_target", "")).strip()
            or str(item.qa.get("qa_type", "")).strip()
        )
        item.append_journey_event(
            stage="generator",
            event_type="generated",
            qa_type_before=qa_type,
            qa_type_after=qa_type,
            details={
                "generation_mode": str(
                    item.generation_metadata.get("generation_mode", "")
                ).strip(),
                "source_task_id": str(
                    item.generation_metadata.get("source_task_id", "")
                ).strip(),
            },
        )


def _annotate_transformed_items(items: list[GeneratedQA]) -> None:
    for item in items:
        transform_meta = dict(item.generation_metadata.get("transformation", {}) or {})
        steps = list(transform_meta.get("steps", []) or [])
        latest_step = steps[-1] if steps else {}
        qa_type = str(item.qa.get("qa_type", "")).strip()
        item.append_journey_event(
            stage="transform",
            event_type="transformed",
            qa_type_before=qa_type,
            qa_type_after=qa_type,
            details={
                "changed": bool(latest_step.get("changed", False)),
                "validation_reason": str(
                    latest_step.get("validation_reason", "")
                ).strip(),
                "target_style": str(latest_step.get("target_style", "")).strip(),
            },
        )


def _accept_items_under_type_quota(
    items: list[GeneratedQA],
    *,
    accepted_type_counts: dict[str, int],
    target_type_counts: dict[str, int],
) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
    accepted: list[GeneratedQA] = []
    rejected: list[GeneratedQA] = []

    # Sort by composite score descending so higher-quality items fill quota first.
    items = sorted(
        items,
        key=lambda i: i.qa.get("eval_scores", {}).get("composite", 0),
        reverse=True,
    )

    for item in items:
        original_type = str(item.qa.get("qa_type", "")).strip() or (
            str(item.generation_metadata.get("qa_type_target", "")).strip() or "lookup"
        )
        effective_type, was_relabeled, relabel_direction = _resolve_effective_qa_type(
            item
        )
        item.qa["qa_type"] = effective_type

        if was_relabeled:
            item.append_journey_event(
                stage="type_balancer",
                event_type="relabelled",
                qa_type_before=original_type,
                qa_type_after=effective_type,
                reason_code=relabel_direction,
            )

        remaining = max(0, int(target_type_counts.get(effective_type, 0))) - max(
            0, int(accepted_type_counts.get(effective_type, 0))
        )
        if remaining > 0:
            accepted_type_counts[effective_type] = (
                int(accepted_type_counts.get(effective_type, 0)) + 1
            )
            item.append_journey_event(
                stage="type_balancer",
                event_type="accepted",
                qa_type_before=original_type,
                qa_type_after=effective_type,
                details={
                    "requested_qa_type": str(
                        item.generation_metadata.get("qa_type_target", "")
                    ).strip(),
                    "remaining_quota_after_accept": remaining - 1,
                },
            )
            accepted.append(item)
            continue

        _reject_item(
            item,
            reason="type_quota_exceeded",
            reason_code="type_quota_exceeded",
            metadata_extra={
                "requested_qa_type": str(
                    item.generation_metadata.get("qa_type_target", "")
                ).strip(),
                "effective_qa_type": effective_type,
                "was_relabelled": was_relabeled,
                "relabel_direction": relabel_direction,
                "quota_remaining_at_rejection": remaining,
            },
        )
        item.append_journey_event(
            stage="type_balancer",
            event_type="quota_rejected",
            qa_type_before=original_type,
            qa_type_after=effective_type,
            reason_code="type_quota_exceeded",
            details={
                "requested_qa_type": str(
                    item.generation_metadata.get("qa_type_target", "")
                ).strip(),
                "effective_qa_type": effective_type,
                "was_relabelled": was_relabeled,
                "relabel_direction": relabel_direction,
                "quota_remaining_at_rejection": remaining,
            },
        )
        rejected.append(item)

    return accepted, rejected


def _collect_journey_stats(
    *,
    passed_items: list[GeneratedQA],
    rejected_items: list[GeneratedQA],
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    relabel_counts: Counter[str] = Counter()
    quota_rejected_by_effective_type: Counter[str] = Counter()
    accepted_by_requested_type: Counter[str] = Counter()
    accepted_by_effective_type: Counter[str] = Counter()
    accepted_requested_to_effective: Counter[str] = Counter()

    for item in [*passed_items, *rejected_items]:
        for event in item.journey_events:
            event_type = str(event.get("event_type", "")).strip()
            if event_type:
                event_counts[event_type] += 1
            if event_type == "relabelled":
                relabel_key = str(event.get("reason_code", "")).strip() or (
                    f"{event.get('qa_type_before', '')}->{event.get('qa_type_after', '')}"
                )
                relabel_counts[relabel_key] += 1
            if event_type == "quota_rejected":
                effective_type = (
                    str(event.get("qa_type_after", "")).strip()
                    or str(
                        event.get("details", {}).get("effective_qa_type", "")
                    ).strip()
                )
                if effective_type:
                    quota_rejected_by_effective_type[effective_type] += 1

    for item in passed_items:
        requested_type = (
            str(item.generation_metadata.get("qa_type_target", "")).strip()
            or str(item.qa.get("qa_type", "")).strip()
            or "lookup"
        )
        effective_type = str(item.qa.get("qa_type", "")).strip() or requested_type
        accepted_by_requested_type[requested_type] += 1
        accepted_by_effective_type[effective_type] += 1
        accepted_requested_to_effective[f"{requested_type}->{effective_type}"] += 1

    return {
        "event_counts": dict(event_counts),
        "relabeled": dict(relabel_counts),
        "quota_rejected_by_effective_type": dict(quota_rejected_by_effective_type),
        "accepted_by_requested_type": dict(accepted_by_requested_type),
        "accepted_by_effective_type": dict(accepted_by_effective_type),
        "accepted_requested_to_effective": dict(accepted_requested_to_effective),
    }


def _relabel_qa_types(items: list[GeneratedQA]) -> dict[str, int]:
    """Correct qa_type labels based on actual reference chunk structure.

    A ``lookup`` with chunks from multiple distinct files is actually multi-hop.
    A ``multi_hop`` with only one reference chunk (or all chunks from the same
    file) is effectively a lookup.
    """
    stats = {"relabeled": 0, "lookup_to_multi_hop": 0, "multi_hop_to_lookup": 0}
    for item in items:
        effective_type, was_relabeled, relabel_direction = _resolve_effective_qa_type(
            item
        )
        item.qa["qa_type"] = effective_type
        if was_relabeled:
            stats["relabeled"] += 1
            if relabel_direction:
                stats[relabel_direction] += 1
    return stats


def _normalize_failure_type_for_regeneration(
    *,
    metadata: dict[str, Any],
    reason_code: str = "",
    verdict_reason: str = "",
    verdict_reasoning: str = "",
) -> str:
    raw = str(metadata.get("failure_type", "")).strip().lower()
    if raw in {"too_easy", "unsupported", "redundant_chunk"}:
        return raw

    text = " ".join(
        part.lower()
        for part in (reason_code, verdict_reason, verdict_reasoning)
        if str(part).strip()
    )
    if "unsupported" in text or "unanswerable" in text:
        return "unsupported"
    if "too_easy" in text or "naive_retrieval_sufficient" in text:
        return "too_easy"
    return "unknown"


def _expected_action_for_failure_type(failure_type: str) -> str:
    if failure_type == "too_easy":
        return "increase_difficulty_keep_answer_locked"
    if failure_type == "unsupported":
        return "reanchor_and_revise_answer_using_new_evidence"
    return "address_feedback_and_remain_grounded"


def _select_regeneration_seed(
    *,
    current_seed_chunk_id: str,
    seed_ids: list[str],
    item_index: int,
    current_refinement_count: int,
    same_seed_failures: int,
    max_same_seed_attempts_before_reanchor: int,
    force_reanchor: bool = False,
    context: Any = None,
) -> tuple[str, bool]:
    should_reanchor = force_reanchor
    if not should_reanchor and max_same_seed_attempts_before_reanchor > 0:
        should_reanchor = same_seed_failures >= max_same_seed_attempts_before_reanchor

    if not should_reanchor:
        return current_seed_chunk_id, False

    # Try to sample a fresh chunk from the corpus instead of rotating
    # within the stale seed pool.
    if context is not None:
        source = getattr(context, "source", None)
        min_chars = (
            getattr(context.config.corpus, "min_chunk_chars", 0) if context else 0
        )
        if source is not None:
            try:
                fresh = source.sample_chunks(1, min_chars=min_chars)
                if fresh:
                    chunk = fresh[0]
                    # Register in the seed lookup so the linker can find it.
                    seed_lookup = context.get("seed_chunk_lookup", {})
                    if isinstance(seed_lookup, dict):
                        seed_lookup[chunk.hash] = chunk
                    return chunk.hash, True
            except Exception:
                logger.debug("Fresh seed sampling failed; falling back to pool.")

    # Fallback: pick from the existing pool.
    candidates = [sid for sid in seed_ids if sid and sid != current_seed_chunk_id]
    if not candidates:
        return current_seed_chunk_id, False
    next_seed = candidates[(item_index + current_refinement_count) % len(candidates)]
    return next_seed, True


def _build_regeneration_prompt(
    item: GeneratedQA,
    *,
    regeneration_attempt: int,
    source_task_id: str,
    old_seed_chunk_id: str,
    new_seed_chunk_id: str,
    switched_seed: bool,
) -> str:
    verdict = item.filter_verdict
    metadata = dict(verdict.metadata) if verdict is not None else {}
    reason_code = str(metadata.get("reason_code", "")).strip() or (
        verdict.reason if verdict else "unknown"
    )
    reasoning = str(verdict.reasoning if verdict is not None else "").strip()
    hint = str(metadata.get("refinement_hint", "")).strip()
    judge_reasoning = str(metadata.get("judge_reasoning", "")).strip()
    lexical_evidence = str(metadata.get("judge_lexical_anchor_evidence", "")).strip()
    failure_type = _normalize_failure_type_for_regeneration(
        metadata=metadata,
        reason_code=reason_code,
        verdict_reason=verdict.reason if verdict is not None else "",
        verdict_reasoning=reasoning,
    )
    judge_reason_tag = str(metadata.get("judge_reason_tag", "")).strip() or "unknown"
    overlap_triggered = bool(metadata.get("overlap_triggered", False))
    question = _short_text(item.qa.get("question", ""), max_chars=420)
    answer = _short_text(item.qa.get("answer", ""), max_chars=520)
    expected_action = _expected_action_for_failure_type(failure_type)

    lines = [
        "Regeneration instructions:",
        f"- source_task_id: {source_task_id}",
        f"- attempt: {regeneration_attempt}",
        f"- previous_filter_reason_code: {reason_code or 'unknown'}",
        f"- previous_failure_type: {failure_type}",
        f"- previous_judge_reason_tag: {judge_reason_tag}",
        f"- overlap_triggered: {str(overlap_triggered).lower()}",
        f"- expected_action: {expected_action}",
    ]
    if reasoning:
        lines.append(f"- previous_filter_reasoning: {reasoning}")
    if hint:
        lines.append(f"- previous_refinement_hint: {hint}")
    if judge_reasoning:
        lines.append(f"- previous_judge_reasoning: {judge_reasoning}")
    if lexical_evidence:
        lines.append(f"- overlapping_terms_that_caused_rejection: {lexical_evidence}")
    if switched_seed:
        lines.append(
            "- reanchor_action: switched seed chunk after repeated failures "
            f"({old_seed_chunk_id} -> {new_seed_chunk_id})"
        )
    else:
        lines.append(f"- reanchor_action: keep same seed chunk ({new_seed_chunk_id})")
    lines.extend(
        [
            "Previous QA to improve:",
            f"- question: {question}",
            f"- answer: {answer}",
            (
                "Rewrite the QA to address the failure reason while staying grounded in the provided "  # noqa: E501
                "primary/secondary chunks for this attempt."
            ),
        ]
    )
    return "\n".join(lines)


def _create_generation_task(task_kwargs: dict[str, Any]) -> GenerationTask:
    """Instantiate GenerationTask while tolerating older class signatures."""
    try:
        return GenerationTask(**task_kwargs)
    except TypeError:
        try:
            supported_keys = set(inspect.signature(GenerationTask).parameters)
        except (TypeError, ValueError):
            supported_keys = set()
        if not supported_keys:
            raise

        compatible_kwargs = {
            key: value for key, value in task_kwargs.items() if key in supported_keys
        }
        if len(compatible_kwargs) == len(task_kwargs):
            raise

        task = GenerationTask(**compatible_kwargs)

        # Backfill dropped fields for downstream template rendering when possible.
        for key, value in task_kwargs.items():
            if key in compatible_kwargs or hasattr(task, key):
                continue
            try:
                setattr(task, key, value)
            except Exception:
                continue
        return task


def _build_regeneration_tasks(
    items: list[GeneratedQA],
    *,
    context: PipelineContext,
) -> tuple[list[GenerationTask], dict[str, GeneratedQA]]:
    seed_lookup = context.get("seed_chunk_lookup", {}) or {}
    fallback_seed_ids = [str(seed_id) for seed_id in seed_lookup if str(seed_id)]

    tasks: list[GenerationTask] = []
    item_by_task_id: dict[str, GeneratedQA] = {}
    seen_task_ids: set[str] = set()

    max_same_seed_attempts_before_reanchor = max(
        0,
        _int_or(
            context.config.refinement.max_same_seed_attempts_before_reanchor,
            0,
        ),
    )

    for idx, item in enumerate(items):
        meta = dict(item.generation_metadata)
        verdict_meta = (
            dict(item.filter_verdict.metadata)
            if item.filter_verdict is not None
            else {}
        )
        verdict_reason = str(
            item.filter_verdict.reason if item.filter_verdict is not None else ""
        ).strip()
        verdict_reasoning = str(
            item.filter_verdict.reasoning if item.filter_verdict is not None else ""
        ).strip()
        reason_code = str(verdict_meta.get("reason_code", "")).strip() or verdict_reason
        failure_type = _normalize_failure_type_for_regeneration(
            metadata=verdict_meta,
            reason_code=reason_code,
            verdict_reason=verdict_reason,
            verdict_reasoning=verdict_reasoning,
        )
        judge_reason_tag = (
            str(verdict_meta.get("judge_reason_tag", "")).strip() or "unknown"
        )
        overlap_triggered = bool(verdict_meta.get("overlap_triggered", False))
        expected_action = _expected_action_for_failure_type(failure_type)
        current_count = max(0, _int_or(meta.get("refinement_count", 0), 0))
        same_seed_failures = max(
            0, _int_or(meta.get("same_seed_refinement_count", 0), 0)
        )
        force_reanchor = bool(verdict_meta.get("force_reanchor", False))

        qa_type = (
            str(meta.get("qa_type_target", "")).strip()
            or str(item.qa.get("qa_type", "")).strip()
        )
        if not qa_type:
            qa_type = "lookup"

        target_hop_count = max(
            1,
            _int_or(meta.get("target_hop_count", item.qa.get("min_hop_count", 1)), 1),
        )
        if failure_type == "too_easy" and qa_type != "lookup":
            # Too-easy retries for multi-hop should increase composition depth.
            # Lookup items stay at hop_count=1 — the prompt already instructs
            # the LLM to rephrase for reduced lexical overlap.
            target_hop_count = min(max(target_hop_count + 1, 2), 6)

        seed_chunk_id = str(meta.get("seed_chunk_id", "")).strip()
        if not seed_chunk_id:
            reference_chunks = list(item.qa.get("reference_chunks", []) or [])
            if reference_chunks:
                seed_chunk_id = str((reference_chunks[0] or {}).get("id", "")).strip()
        if not seed_chunk_id and fallback_seed_ids:
            seed_chunk_id = fallback_seed_ids[idx % len(fallback_seed_ids)]
        old_seed_chunk_id = seed_chunk_id
        seed_chunk_id, switched_seed = _select_regeneration_seed(
            current_seed_chunk_id=seed_chunk_id,
            seed_ids=fallback_seed_ids,
            item_index=idx,
            current_refinement_count=current_count,
            same_seed_failures=same_seed_failures,
            max_same_seed_attempts_before_reanchor=max_same_seed_attempts_before_reanchor,
            force_reanchor=force_reanchor,
            context=context,
        )

        source_task_id = (
            str(meta.get("task_id", "")).strip() or f"task_refine_{idx:05d}"
        )
        regeneration_attempt = current_count + 1
        task_id = f"{source_task_id}__regen_{current_count + 1:02d}"
        suffix = 1
        while task_id in seen_task_ids:
            suffix += 1
            task_id = f"{source_task_id}__regen_{current_count + 1:02d}_{suffix}"
        seen_task_ids.add(task_id)

        task_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "qa_type": qa_type,
            "target_hop_count": target_hop_count,
            "seed_chunk_id": seed_chunk_id,
            "regeneration_prompt": _build_regeneration_prompt(
                item,
                regeneration_attempt=regeneration_attempt,
                source_task_id=source_task_id,
                old_seed_chunk_id=old_seed_chunk_id,
                new_seed_chunk_id=seed_chunk_id,
                switched_seed=switched_seed,
            ),
            "regeneration_attempt": regeneration_attempt,
            "source_task_id": source_task_id,
            "previous_failure_type": failure_type,
            "previous_judge_reason_tag": judge_reason_tag,
            "overlap_triggered": overlap_triggered,
            "expected_action": expected_action,
            "failed_question": str(item.qa.get("question", "")).strip(),
            "failed_answer": str(item.qa.get("answer", "")).strip(),
        }
        task = _create_generation_task(task_kwargs)
        tasks.append(task)
        item_by_task_id[task.task_id] = item

    return tasks, item_by_task_id


async def _run_stage(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Await *fn* if it's a coroutine function, else offload it to a worker thread.

    Stages flip to ``async def`` one increment at a time during the async
    migration; this adapter lets the async batch body call sync and async stages
    uniformly. The ``to_thread`` branch goes away once every stage is async.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _regenerate_with_generator(
    items: list[GeneratedQA],
    *,
    generator: QuestionGenerator,
    context: PipelineContext,
) -> tuple[list[GeneratedQA], list[GeneratedQA]]:
    tasks, originals_by_task_id = _build_regeneration_tasks(items, context=context)
    if not tasks:
        return [], []

    regenerated_items = await _run_stage(generator.generate, tasks, context)
    regenerated_by_task_id: dict[str, GeneratedQA] = {}
    for regenerated in regenerated_items:
        task_id = str(regenerated.generation_metadata.get("task_id", "")).strip()
        if not task_id:
            logger.warning("Dropping regenerated item with missing task_id metadata.")
            continue
        if task_id not in originals_by_task_id:
            logger.warning(
                "Dropping regenerated item for unknown task_id '%s'.", task_id
            )
            continue
        if task_id in regenerated_by_task_id:
            logger.warning(
                "Dropping duplicate regenerated item for task_id '%s'.", task_id
            )
            continue
        regenerated_by_task_id[task_id] = regenerated

    regenerated_for_retry: list[GeneratedQA] = []
    failed_to_regenerate: list[GeneratedQA] = []
    for task in tasks:
        source_item = originals_by_task_id[task.task_id]
        regenerated = regenerated_by_task_id.get(task.task_id)
        if regenerated is None:
            failed_to_regenerate.append(source_item)
            continue

        source_meta = dict(source_item.generation_metadata)
        current_count = max(0, _int_or(source_meta.get("refinement_count", 0), 0))
        next_count = current_count + 1
        previous_seed_chunk_id = (
            str(source_meta.get("seed_chunk_id", "")).strip() or task.seed_chunk_id
        )
        same_seed_failures = max(
            0, _int_or(source_meta.get("same_seed_refinement_count", 0), 0)
        )
        next_same_seed_failures = same_seed_failures + 1
        if task.seed_chunk_id != previous_seed_chunk_id:
            next_same_seed_failures = 1

        merged_meta = {**source_meta, **dict(regenerated.generation_metadata)}
        merged_meta["task_id"] = task.task_id
        merged_meta["seed_chunk_id"] = task.seed_chunk_id
        merged_meta["refinement_count"] = next_count
        merged_meta["same_seed_refinement_count"] = next_same_seed_failures
        merged_meta["source_task_id"] = (
            task.source_task_id or str(source_meta.get("task_id", "")).strip()
        )
        merged_meta["regeneration_attempt"] = task.regeneration_attempt

        regenerated.generation_metadata = merged_meta
        regenerated.filter_verdict = None
        regenerated.journey_events = list(source_item.journey_events) + list(
            regenerated.journey_events
        )
        regenerated.regeneration_history = source_item.regeneration_history + [
            {
                "type": "generator_retry",
                "round": next_count,
                "source_task_id": str(source_meta.get("task_id", "")).strip()
                or task.task_id,
                "seed_chunk_id": task.seed_chunk_id,
                "reanchored": task.seed_chunk_id != previous_seed_chunk_id,
            }
        ]
        qa_type = str(regenerated.qa.get("qa_type", "")).strip()
        regenerated.append_journey_event(
            stage="regeneration",
            event_type="regenerated",
            task_id=task.task_id,
            refinement_count=next_count,
            qa_type_before=qa_type,
            qa_type_after=qa_type,
            reason_code=str(task.previous_failure_type).strip(),
            details={
                "source_task_id": str(
                    task.source_task_id or source_meta.get("task_id", "")
                ).strip(),
                "seed_chunk_id": task.seed_chunk_id,
                "reanchored": task.seed_chunk_id != previous_seed_chunk_id,
            },
        )
        regenerated_for_retry.append(regenerated)

    return regenerated_for_retry, failed_to_regenerate


def _build_corpus_profile(
    cfg: PipelineConfig, source: Any, context: PipelineContext
) -> None:
    """Generate corpus summary/example queries from description and samples."""
    profile_cfg = cfg.corpus_context
    default_summary = profile_cfg.description
    default_queries = list(profile_cfg.example_queries)
    default_queries_bulleted = "\n".join(f"- {query}" for query in default_queries)

    # Always set context keys so generators can safely reference these placeholders.
    context["corpus_summary"] = default_summary
    context["corpus_queries"] = default_queries_bulleted
    context["corpus_example_queries"] = default_queries
    context["corpus_description"] = profile_cfg.description
    context["corpus_language"] = profile_cfg.language

    has_user_context = bool(profile_cfg.description.strip() or default_queries)
    if not profile_cfg.enabled or not has_user_context:
        context["corpus_profile"] = {
            "enabled": profile_cfg.enabled,
            "used": False,
            "summary": default_summary,
            "example_queries": default_queries,
            "raw_response": "",
        }
        return

    top_level_chunks = []
    if hasattr(source, "get_top_level_chunks"):
        try:
            top_level_chunks = list(source.get_top_level_chunks() or [])
        except Exception:
            logger.exception("Failed to fetch top-level chunks for corpus profiling.")
            top_level_chunks = []

    corpus_pool: list[Any] = context.get("corpus_pool", []) or []

    sampled_top_level = []
    if top_level_chunks and profile_cfg.num_top_level_samples > 0:
        sample_count = min(profile_cfg.num_top_level_samples, len(top_level_chunks))
        sampled_top_level = select_diverse(
            top_level_chunks, sample_count, rng=context.rng, stratify_key=_get_doc_id
        )

    sampled_random = []
    if profile_cfg.num_random_samples > 0:
        pool_for_random = corpus_pool or []
        if pool_for_random:
            sampled_random = select_diverse(
                pool_for_random,
                profile_cfg.num_random_samples,
                rng=context.rng,
                stratify_key=_get_doc_id,
            )
        else:
            try:
                sampled_random = source.sample_chunks(
                    profile_cfg.num_random_samples,
                    min_chars=profile_cfg.min_chunk_chars,
                )
            except Exception:
                logger.exception("Failed to fetch random chunks for corpus profiling.")
                sampled_random = []

    variables = {
        "user_context": (
            f"Description: {profile_cfg.description}\n"
            f"Example queries provided by user: {', '.join(default_queries)}"
        ),
        "top_level_content": "\n\n".join(
            (chunk.chunk_str() if hasattr(chunk, "chunk_str") else str(chunk))
            for chunk in sampled_top_level
        ),
        "random_content": "\n\n".join(
            (chunk.chunk_str() if hasattr(chunk, "chunk_str") else str(chunk))
            for chunk in sampled_random
        ),
    }

    try:
        user_prompt = render_template(profile_cfg.user_template, variables)
        client = _build_openai_client(
            api_key=profile_cfg.api_key,
            base_url=profile_cfg.base_url,
        )
        completion = _chat_completion_with_retry(
            client=client,
            model=profile_cfg.model,
            messages=[
                {"role": "system", "content": profile_cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        summary, queries = parse_corpus_summary_response(raw)
    except Exception as exc:
        logger.warning(
            "Corpus profiling failed; using user-provided description/example queries. reason=%s",
            exc,
        )
        summary, queries, raw, user_prompt = "", [], "", ""

    summary = summary.strip() or default_summary
    queries = [
        str(query).strip() for query in queries if str(query).strip()
    ] or default_queries

    context["corpus_summary"] = summary
    context["corpus_queries"] = "\n".join(f"- {query}" for query in queries)
    context["corpus_example_queries"] = queries
    context["corpus_profile"] = {
        "enabled": profile_cfg.enabled,
        "used": bool(raw),
        "summary": summary,
        "example_queries": queries,
        "user_prompt": user_prompt,
        "raw_response": raw,
    }


def _print_progress(message: str, *, verbose: bool) -> None:
    if verbose:
        _tqdm.write(message)


class Pipeline:
    """Orchestrates generation, filtering, refinement, and formatting."""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        source_factory: Callable[[PipelineConfig], Any] | None = None,
        rollout_client_factory: Callable[[PipelineConfig], RolloutClient] | None = None,
    ) -> None:
        self.cfg = cfg
        self.source_factory = source_factory or _load_source
        self.rollout_client_factory = rollout_client_factory

    @property
    def checkpoint_dir(self) -> Path:
        """Resolve the checkpoint directory path from config (mirrors _run_work_queue logic)."""
        cfg = self.cfg
        ckpt_dir_str = cfg.micro_batch.checkpoint_dir.strip()
        if not ckpt_dir_str:
            return Path(cfg.output.dir) / ".checkpoints"
        return Path(ckpt_dir_str)

    def run(self) -> dict[str, Any]:
        t_prep = time.monotonic()
        context = self._prepare_context()
        context["prepare_context_seconds"] = time.monotonic() - t_prep
        return self._run_from_context(context)

    # Public aliases — let external callers (e.g. platform-service codegen
    # running on a Modal worker) compose stages out-of-process: prepare
    # context here, do GPU entity extraction in a sidecar, inject results
    # into context, then call run_from_context. The leading-underscore
    # names stay so internal call sites and existing benchmax tests don't
    # have to flip.
    def prepare_context(self) -> PipelineContext:
        return self._prepare_context()

    def run_from_context(self, context: PipelineContext) -> dict[str, Any]:
        return self._run_from_context(context)

    def _prepare_context(self) -> PipelineContext:
        """Run stages 1-3: load source, build corpus profile, extract entities.

        Returns a fully populated PipelineContext with ``profile`` set, ready for
        ``_run_from_context``.  Resolves API keys as a side-effect.
        """
        cfg = self.cfg
        cfg.resolve_api_keys()

        source = self.source_factory(cfg)
        rng = random.Random(cfg.random_seed)
        context = PipelineContext(config=cfg, source=source, rng=rng)

        _print_progress("[1/6] Loading chunks from corpus...", verbose=cfg.verbose)
        try:
            chunk_count = source.get_chunk_count()
        except AttributeError:
            chunk_count = cfg.targets.total_samples * 10

        # Ensure all chunks are accessible in memory.  PostgresChunkSource
        # already materialises via populate; API-only backends (Turbopuffer)
        # need an explicit fetch so the entity-chunk graph and linker can
        # resolve any chunk by hash.
        max_materialize = 50_000
        if getattr(source, "collection", None) is None and chunk_count > 0:
            from castform.rag.chunkers.models import ChunkCollection  # noqa: PLC0415

            materialize_count = min(chunk_count, max_materialize)
            if chunk_count > max_materialize:
                logger.warning(
                    "Corpus has %d chunks (limit %d). Materialising a capped "
                    "sample so entity extraction and the chunk graph still work.",
                    chunk_count,
                    max_materialize,
                )
            else:
                logger.info(
                    "Materialising %d chunks from API backend into memory...",
                    chunk_count,
                )

            # Use sequential scan when available — cursor pagination avoids
            # the ID-collision overhead of random sampling at high fill rates.
            # ~1.9x faster for 50k chunks from a 65k namespace.
            if hasattr(source, "scan_chunks"):
                all_chunks = source.scan_chunks(
                    limit=materialize_count,
                    min_chars=cfg.corpus.min_chunk_chars,
                )
            else:
                all_chunks = source.sample_chunks(
                    materialize_count,
                    min_chars=cfg.corpus.min_chunk_chars,
                )

            if all_chunks:
                source.collection = ChunkCollection(chunks=all_chunks)  # type: ignore[attr-defined]
                logger.info(
                    "Cached %d/%d chunks on source.collection",
                    len(all_chunks),
                    chunk_count,
                )

        profile_sample = diverse_profile_sample(
            source,
            corpus_size=chunk_count,
            min_chars=cfg.corpus.min_chunk_chars,
            rng=rng,
        )
        if not profile_sample:
            raise RuntimeError("No eligible chunks were found for Pipeline generation.")
        _print_progress(
            f"[1/6] Loaded {len(profile_sample)} profile chunks from corpus",
            verbose=cfg.verbose,
        )
        header_prevalence = compute_header_prevalence(profile_sample)
        logger.debug(
            "Header prevalence: %.2f (pool size %d)",
            header_prevalence,
            len(profile_sample),
        )

        context["seed_chunk_lookup"] = {chunk.hash: chunk for chunk in profile_sample}
        context["corpus_pool"] = profile_sample
        context["corpus_language"] = str(cfg.corpus_context.language or "").strip()
        # Detect search capabilities from source.
        search_modes, best_search_mode = detect_search_capabilities(source)

        _print_progress("[2/6] Building corpus profile...", verbose=cfg.verbose)
        _build_corpus_profile(cfg, source, context)
        _print_progress("[2/6] Built corpus profile", verbose=cfg.verbose)

        kb_entities: list = []
        kb_entity_idx: dict[str, set[str]] = {}
        kb_chunk_idx: dict[str, list[str]] = {}
        kb_cooccurrence: dict[tuple[str, str], int] = {}

        if cfg.corpus_context.generate_entity_patterns:
            _print_progress("[3/6] Extracting entity patterns...", verbose=cfg.verbose)

            # Phase 1a: KeyBERT + metadata entity extraction.
            # Handles its own sampling internally; scans full corpus for graph.
            from castform.rag.qa_generation.corpus_profile import extract_entities

            all_chunks = (
                list(source.collection.chunks)
                if getattr(source, "collection", None)
                else profile_sample
            )
            kb_entities, kb_entity_idx, kb_chunk_idx = extract_entities(all_chunks)
            # Compute co-occurrence from the KeyBERT graph
            kb_entity_names = sorted(
                name for name, hashes in kb_entity_idx.items() if hashes
            )
            kb_cooccurrence: dict[tuple[str, str], int] = {}
            for i, e1 in enumerate(kb_entity_names):
                for e2 in kb_entity_names[i + 1 :]:
                    overlap = len(kb_entity_idx[e1] & kb_entity_idx[e2])
                    if overlap > 0:
                        kb_cooccurrence[(e1, e2)] = overlap

            logger.info(
                "KeyBERT extraction: %d entities from %d chunks",
                len(kb_entities),
                len(all_chunks),
            )

        else:
            _print_progress(
                "[3/6] Skipped entity pattern generation", verbose=cfg.verbose
            )

        # --- Build CorpusProfile from stages 2-3 outputs ---
        from castform.rag.qa_generation.corpus_capabilities import CorpusCapabilities

        # Use KeyBERT graph directly — already built during extract_entities().
        entity_patterns = (
            kb_entities if cfg.corpus_context.generate_entity_patterns else []
        )
        entity_chunk_idx = kb_entity_idx if entity_patterns else {}
        chunk_entity_idx = kb_chunk_idx if entity_patterns else {}
        cooccurrence = kb_cooccurrence if entity_patterns else {}

        token_df, token_df_n = compute_token_document_frequency(profile_sample)

        profile = CorpusProfile(
            corpus_summary=context.get("corpus_summary", ""),
            corpus_queries=context.get("corpus_example_queries", []),
            corpus_description=context.get("corpus_description", ""),
            entity_patterns=entity_patterns,
            capabilities=CorpusCapabilities.detect(profile_sample),
            search_modes=search_modes,
            best_search_mode=best_search_mode,
            token_document_frequency=token_df,
            token_df_sample_size=token_df_n,
            entity_chunk_index=entity_chunk_idx,
            chunk_entity_index=chunk_entity_idx,
            entity_cooccurrence=cooccurrence,
        )
        context.profile = profile

        # --- Optional wiki preprocessing ---
        if cfg.wiki_preprocessing.enabled:
            _print_progress(
                "[3.5/6] Building wiki from entity clusters...", verbose=cfg.verbose
            )
            from openai import OpenAI as _OpenAI  # noqa: PLC0415

            from castform.rag.qa_generation.wiki_builder import WikiBuilder  # noqa: PLC0415

            wiki_client = _OpenAI(
                # Empty key → credential seam (this block only runs when enabled).
                api_key=resolve_judge_key(
                    cfg.wiki_preprocessing.api_key, cfg.wiki_preprocessing.base_url
                ),
                base_url=cfg.wiki_preprocessing.base_url or None,
            )
            wiki_builder = WikiBuilder(cfg.wiki_preprocessing, wiki_client)

            # Use all corpus chunks for clustering, not just the profile sample.
            wiki_chunks = list(getattr(source, "collection", None) or profile_sample)
            clusters = wiki_builder.cluster_chunks(
                wiki_chunks,
                profile.entity_patterns,
                profile=profile,
            )

            if clusters:
                wiki_index = wiki_builder.generate_pages(
                    clusters,
                    corpus_summary=context.get("corpus_summary", ""),
                    corpus_description=context.get("corpus_description", ""),
                    corpus_language=str(
                        context.get("corpus_language", "") or ""
                    ).strip(),
                )
                context["wiki_index"] = wiki_index
                context["wiki_builder"] = wiki_builder
                _print_progress(
                    f"[3.5/6] Built {len(wiki_index.pages)} wiki pages from "
                    f"{len(clusters)} entity clusters",
                    verbose=cfg.verbose,
                )
            else:
                _print_progress(
                    "[3.5/6] No entity clusters large enough for wiki pages",
                    verbose=cfg.verbose,
                )

        # --- Corpus-aware auto-tuning and warnings ---
        _entity_names = profile.get_entity_names(discriminative_only=False)
        census = compute_metadata_census(profile_sample, _entity_names, chunk_count)
        profile.census = census
        for _chunk in profile_sample:
            _hash = getattr(_chunk, "hash", str(id(_chunk)))
            profile.chunk_suitability_scores[_hash] = compute_chunk_suitability(
                _chunk, census, profile
            )
        context["census"] = census

        warnings = emit_corpus_warnings(census, profile, cfg)
        for w in warnings:
            logger.warning(w)

        tuned = auto_tune(census, profile, cfg)
        if tuned:
            if "total_samples" in tuned:
                original = cfg.targets.total_samples
                cfg.targets.total_samples = tuned["total_samples"]
                logger.info(
                    "Auto-tune: total_samples %d -> %d (corpus has %d chunks)",
                    original,
                    tuned["total_samples"],
                    census.chunk_count,
                )
            if "primary_type_distribution" in tuned:
                cfg.targets.primary_type_distribution = tuned[
                    "primary_type_distribution"
                ]
                logger.info(
                    "Auto-tune: primary_type_distribution -> %s",
                    tuned["primary_type_distribution"],
                )
            if "hop_distribution" in tuned:
                cfg.targets.hop_distribution = tuned["hop_distribution"]
                logger.info(
                    "Auto-tune: hop_distribution -> %s",
                    tuned["hop_distribution"],
                )
            if "reasoning_mode_distribution" in tuned:
                cfg.targets.reasoning_mode_distribution = tuned[
                    "reasoning_mode_distribution"
                ]
                logger.info(
                    "Auto-tune: reasoning_mode_distribution -> %s",
                    tuned["reasoning_mode_distribution"],
                )

        return context

    def _run_from_context(self, context: PipelineContext) -> dict[str, Any]:
        """Run stages 4-8 given a context already populated by ``_prepare_context``.

        This is the compute-heavy phase: task creation, generation, filtering,
        refinement, dedup, and formatting.  The method also drives the micro-batch
        work queue and writes checkpoint files.
        """
        cfg = self.cfg
        source = context.source
        profile = context.profile

        pipeline_metrics = PipelineMetrics(
            target_samples=cfg.targets.total_samples,
        )
        context.metrics = pipeline_metrics
        t0 = time.monotonic()

        _print_progress("[4/6] Preparing generation...", verbose=cfg.verbose)

        linker = _build_linker(
            cfg, source, profile=profile, wiki_index=context.get("wiki_index")
        )
        generator = _build_generator(cfg, linker=linker)
        guard_filter = DeterministicGuardsFilter(cfg.filtering.deterministic_guards)
        filter_stage_names, filter_chain = _build_filter_chain(
            cfg,
            source=source,
            rollout_client_factory=self.rollout_client_factory,
        )
        context["filter_chain"] = list(filter_stage_names)
        transformer = _build_transformer(cfg)
        formatter = TrainEvalFormatter(output_cfg=cfg.output, split_cfg=cfg.split)

        # --- Micro-batch work queue (generate → filter → transform) ---
        _batch_size_display = cfg.micro_batch.batch_size or "auto"
        _print_progress(
            f"[5/6] Starting work queue (batch_size={_batch_size_display})...",
            verbose=cfg.verbose,
        )
        all_passed, all_rejected, total_regens, raw_items = self._run_work_queue(
            source=source,
            generator=generator,
            guard_filter=guard_filter,
            filter_stage_names=filter_stage_names,
            filter_chain=filter_chain,
            transformer=transformer,
            context=context,
        )

        # --- Global post-processing ---

        # Dedup: remove near-duplicate questions.
        from castform.rag.qa_generation.transformers.dedup import (  # noqa: I001
            DedupConfig as _DedupCfg,
            QuestionDeduplicator,
        )

        dedup_cfg = cfg.dedup
        deduplicator = QuestionDeduplicator(
            cfg=_DedupCfg(
                enabled=dedup_cfg.enabled,
                similarity_threshold=dedup_cfg.similarity_threshold,
                ngram_size=dedup_cfg.ngram_size,
                stats_key=dedup_cfg.stats_key,
            )
        )
        pre_dedup = len(all_passed)
        all_passed, all_rejected = deduplicator.deduplicate(
            all_passed, all_rejected, context
        )
        n_deduped = pre_dedup - len(all_passed)
        if n_deduped > 0:
            _print_progress(
                f"    Dedup: removed {n_deduped} near-duplicates",
                verbose=cfg.verbose,
            )

        relabel_stats = _relabel_qa_types(all_passed)
        if relabel_stats["relabeled"] > 0:
            _print_progress(
                f"    Warning: {relabel_stats['relabeled']} accepted items were still misaligned"
                f" ({relabel_stats['lookup_to_multi_hop']} lookup→multi_hop,"
                f" {relabel_stats['multi_hop_to_lookup']} multi_hop→lookup)",
                verbose=cfg.verbose,
            )

        context["rejected_items"] = [
            _serialize_qa_with_filter_details(item) for item in all_rejected
        ]
        context["passed_items"] = [
            _serialize_qa_with_filter_details(item) for item in all_passed
        ]
        context["total_regenerations"] = total_regens

        _print_progress("[6/6] Formatting output...", verbose=cfg.verbose)
        result = formatter.format(all_passed, context)
        _train = result.get("stats", {}).get("train", 0)
        _eval = result.get("stats", {}).get("eval", 0)
        _print_progress(
            f"[6/6] Formatted output: {_train} train, {_eval} eval",
            verbose=cfg.verbose,
        )
        _print_progress(
            f"\nPipeline complete: {len(all_passed)} passed,"
            f" {len(all_rejected)} rejected"
            f" ({total_regens} regenerations)",
            verbose=cfg.verbose,
        )
        result["raw_candidates"] = [item.qa for item in raw_items]
        result["filtered_dataset"] = [
            _serialize_qa_with_filter_details(item) for item in all_passed
        ]
        result["rejected_dataset"] = [
            _serialize_qa_with_filter_details(item) for item in all_rejected
        ]
        run_stats = RunStats(
            raw_candidates_total=len(raw_items),
            passed_total=len(all_passed),
            rejected_total=len(all_rejected),
            regenerated_total=total_regens,
            round_limit=cfg.refinement.max_rounds,
        )
        result["stats"] = {
            **result.get("stats", {}),
            **run_stats.__dict__,
            "filter_chain": list(filter_stage_names),
        }
        guard_stats = context.get("deterministic_guard_stats")
        if isinstance(guard_stats, dict) and guard_stats:
            result["stats"]["deterministic_guards"] = dict(guard_stats)

        retrieval_too_easy_stats = context.get("retrieval_too_easy_filter_stats")
        if isinstance(retrieval_too_easy_stats, dict) and retrieval_too_easy_stats:
            result["stats"]["retrieval_too_easy_filter"] = dict(
                retrieval_too_easy_stats
            )

        grounding_stats = context.get("grounding_filter_stats")
        if isinstance(grounding_stats, dict) and grounding_stats:
            result["stats"]["grounding_filter"] = dict(grounding_stats)

        transformation_stats = context.get("transformation_stats")
        if isinstance(transformation_stats, dict) and transformation_stats:
            result["stats"]["transformation"] = dict(transformation_stats)

        dedup_stats = context.get("dedup_stats")
        if isinstance(dedup_stats, dict) and dedup_stats:
            result["stats"]["dedup"] = dict(dedup_stats)
        hop_count_stats = context.get("hop_count_validity_stats")
        if isinstance(hop_count_stats, dict) and hop_count_stats:
            result["stats"]["hop_count_validity"] = dict(hop_count_stats)

        journey_stats = _collect_journey_stats(
            passed_items=all_passed,
            rejected_items=all_rejected,
        )
        if journey_stats:
            context["journey_stats"] = journey_stats
            result["stats"]["journey"] = journey_stats

        pipeline_metrics.wall_time_seconds = time.monotonic() - t0
        pipeline_metrics.total_generated = len(raw_items)
        pipeline_metrics.total_accepted = len(all_passed)
        pipeline_metrics.total_rejected = len(all_rejected)
        pipeline_metrics.total_regenerations = total_regens
        if pipeline_metrics.total_generated > 0:
            pipeline_metrics.overall_acceptance_rate = (
                pipeline_metrics.total_accepted / pipeline_metrics.total_generated
            )
        if pipeline_metrics.target_samples > 0:
            pipeline_metrics.fill_rate = (
                pipeline_metrics.total_accepted / pipeline_metrics.target_samples
            )
        result["stats"]["pipeline_metrics"] = pipeline_metrics.to_dict()
        result["stats"]["prepare_context_seconds"] = context.get(
            "prepare_context_seconds",
            0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Micro-batch work queue
    # ------------------------------------------------------------------

    def _run_work_queue(
        self,
        *,
        source: Any,
        generator: QuestionGenerator,
        guard_filter: DeterministicGuardsFilter,
        filter_stage_names: list[str],
        filter_chain: list[EvaluatorFilter],
        transformer: QuestionTransformer,
        context: PipelineContext,
    ) -> tuple[list[GeneratedQA], list[GeneratedQA], int, list[GeneratedQA]]:
        """Sync bridge into the async work queue (one event loop per run).

        The work queue is the pipeline's single async region; ``_run_from_context``
        (prepare) and the post-processing (dedup/format/stats) stay sync.
        """
        return self._run_coro(
            self._arun_work_queue_with_cleanup(
                source=source,
                generator=generator,
                guard_filter=guard_filter,
                filter_stage_names=filter_stage_names,
                filter_chain=filter_chain,
                transformer=transformer,
                context=context,
            )
        )

    async def _arun_work_queue_with_cleanup(
        self,
        *,
        source: Any,
        generator: QuestionGenerator,
        guard_filter: DeterministicGuardsFilter,
        filter_stage_names: list[str],
        filter_chain: list[EvaluatorFilter],
        transformer: QuestionTransformer,
        context: PipelineContext,
    ) -> tuple[list[GeneratedQA], list[GeneratedQA], int, list[GeneratedQA]]:
        """Run the async work queue, then close every per-loop async client built
        during the run (corpus ``AsyncClient`` + ``AsyncOpenAI`` generator/judge
        clients). ``asyncio.run`` mints a fresh loop per ``Pipeline.run`` and nothing
        else closes these, so without this a long-lived host (wizard, ``castform
        validate``, notebook) leaks one transport set per run."""
        try:
            return await self._arun_work_queue(
                source=source,
                generator=generator,
                guard_filter=guard_filter,
                filter_stage_names=filter_stage_names,
                filter_chain=filter_chain,
                transformer=transformer,
                context=context,
            )
        finally:
            for comp in (generator, transformer, guard_filter, *filter_chain, source):
                aclose = getattr(comp, "aclose", None)
                if aclose is None:
                    continue
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    logger.debug(
                        "Error closing async client during cleanup", exc_info=True
                    )

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        """Run *coro* to completion from sync code.

        ``asyncio.run`` normally; falls back to ``nest_asyncio`` when a loop is
        already running (e.g. a Jupyter kernel).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(coro)
        import nest_asyncio

        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)

    async def _aprocess_batch(
        self,
        tasks: list[GenerationTask],
        *,
        generator: QuestionGenerator,
        guard_filter: DeterministicGuardsFilter,
        filter_stage_names: list[str],
        filter_chain: list[EvaluatorFilter],
        transformer: QuestionTransformer,
        context: PipelineContext,
        incremental_dedup: IncrementalDeduplicator | None = None,
    ) -> tuple[list[GeneratedQA], list[GeneratedQA], list[GeneratedQA], int]:
        """Run stages 5-7 on a single micro-batch, on the event loop.

        Each stage runs through ``_run_stage``: awaited directly once it is
        ``async def`` (later increments), or offloaded to a worker thread while
        still sync. Behavior matches the former sync ``_process_batch`` until the
        stages themselves flip to async.

        Returns:
            (passed, rejected, raw_items, regens_count)
        """
        cfg = self.cfg
        metrics = context.metrics

        # Stage 5: Generate QA candidates.
        if metrics is not None:
            with stage_timer(metrics, "generation", len(tasks)):
                raw_items = await _run_stage(generator.generate, tasks, context)
        else:
            raw_items = await _run_stage(generator.generate, tasks, context)
        _annotate_generated_items(raw_items)

        # Stage 6: Filter + regeneration loop.
        final_passed: list[GeneratedQA] = []
        final_rejected: list[GeneratedQA] = []

        # Early dedup: remove near-duplicates before expensive filtering.
        if incremental_dedup is not None:
            active_items, early_dups = incremental_dedup.check_batch(
                raw_items,
            )
            if early_dups:
                final_rejected.extend(early_dups)
        else:
            active_items = list(raw_items)
        total_regens = 0

        for round_idx in range(cfg.refinement.max_rounds + 1):
            if not active_items:
                break
            for item in active_items:
                item.filter_verdict = None

            if metrics is not None:
                with stage_timer(
                    metrics, "deterministic_guards", len(active_items)
                ) as guard_stage:
                    active_items = await _run_stage(
                        guard_filter.evaluate, active_items, context
                    )
                    guard_stage.items_out_passed += sum(
                        1 for i in active_items if i.is_passed
                    )
                    guard_stage.items_out_rejected += sum(
                        1 for i in active_items if i.is_rejected
                    )
                    guard_stage.items_out_needs_refinement += sum(
                        1 for i in active_items if i.needs_refinement
                    )
            else:
                active_items = await _run_stage(
                    guard_filter.evaluate, active_items, context
                )

            for _stage_name, stage_filter in zip(filter_stage_names, filter_chain):
                if metrics is not None:
                    with stage_timer(metrics, _stage_name, len(active_items)) as sm:
                        active_items = await _run_stage(
                            stage_filter.evaluate, active_items, context
                        )
                        sm.items_out_passed += sum(
                            1 for i in active_items if i.is_passed
                        )
                        sm.items_out_rejected += sum(
                            1 for i in active_items if i.is_rejected
                        )
                        sm.items_out_needs_refinement += sum(
                            1 for i in active_items if i.needs_refinement
                        )
                else:
                    active_items = await _run_stage(
                        stage_filter.evaluate, active_items, context
                    )
                extract_filter_scores(active_items, _stage_name)

            passed = [item for item in active_items if item.is_passed]
            for item in passed:
                item.qa["eval_scores"] = compute_eval_scores(item, cfg.scoring)
            needs_refinement = [item for item in active_items if item.needs_refinement]
            rejected = [item for item in active_items if item.is_rejected]
            _record_filter_events(passed, event_type="filter_passed")
            _record_filter_events(
                needs_refinement, event_type="filter_needs_refinement"
            )
            _record_filter_events(rejected, event_type="filter_rejected")
            final_passed.extend(passed)
            final_rejected.extend(rejected)

            if not needs_refinement:
                active_items = []
                break

            if not cfg.refinement.enabled or round_idx >= cfg.refinement.max_rounds:
                final_rejected.extend(
                    _mark_rejected(
                        needs_refinement,
                        reason="max_rounds_reached",
                        reason_code="max_rounds_reached",
                    )
                )
                active_items = []
                break

            if metrics is not None:
                with stage_timer(metrics, "regeneration", len(needs_refinement)):
                    refined_items, regen_failures = await _regenerate_with_generator(
                        needs_refinement,
                        generator=generator,
                        context=context,
                    )
            else:
                refined_items, regen_failures = await _regenerate_with_generator(
                    needs_refinement,
                    generator=generator,
                    context=context,
                )
            if regen_failures:
                final_rejected.extend(
                    _mark_rejected(
                        regen_failures,
                        reason="generator_regeneration_failed",
                        reason_code="generator_regeneration_failed",
                    )
                )
            total_regens += len(needs_refinement)

            next_active: list[GeneratedQA] = []
            for item in refined_items:
                if item.filter_verdict is None:
                    next_active.append(item)
                elif item.is_rejected:
                    final_rejected.append(item)
                elif item.is_passed:
                    final_passed.append(item)
                else:
                    item.filter_verdict = None
                    next_active.append(item)
            active_items = next_active

        if active_items:
            final_rejected.extend(
                _mark_rejected(
                    active_items,
                    reason="pipeline_terminated_with_unresolved_items",
                    reason_code="unresolved_items",
                )
            )

        # Stage 7: Transform passed items.
        final_passed = await _run_stage(transformer.transform, final_passed, context)
        _annotate_transformed_items(final_passed)

        return final_passed, final_rejected, raw_items, total_regens

    async def _arun_work_queue(
        self,
        *,
        source: Any,
        generator: QuestionGenerator,
        guard_filter: DeterministicGuardsFilter,
        filter_stage_names: list[str],
        filter_chain: list[EvaluatorFilter],
        transformer: QuestionTransformer,
        context: PipelineContext,
    ) -> tuple[list[GeneratedQA], list[GeneratedQA], int, list[GeneratedQA]]:
        """Run stages 5-7 in micro-batches with dynamic distribution.

        Returns:
            (all_passed, all_rejected, total_regens, all_raw_items)
        """
        from pathlib import Path

        from castform.rag.qa_generation.pipeline_config import (
            allocate_largest_remainder_generic,
        )
        from castform.rag.qa_generation.checkpoint import (
            CheckpointManager,
            compute_config_hash,
        )

        cfg = self.cfg
        target = cfg.targets.total_samples
        profile = context.profile

        # Auto-compute batch_size and max_parallel_batches when set to 0.
        batch_size = cfg.micro_batch.batch_size
        if batch_size <= 0:
            batch_size, _ = compute_batch_heuristics(target)
        max_parallel_cfg = cfg.micro_batch.max_parallel_batches
        if max_parallel_cfg <= 0:
            _, max_parallel_cfg = compute_batch_heuristics(target)

        max_iterations = cfg.micro_batch.max_iterations

        # Compute target counts for type, mode, hop distributions.
        target_type_counts = _compute_target_type_counts(cfg)
        target_mode_counts = allocate_largest_remainder_generic(
            target_type_counts.get("multi_hop", 0),
            cfg.targets.reasoning_mode_distribution,
        )
        target_hop_counts = allocate_largest_remainder_generic(
            target_type_counts.get("multi_hop", 0),
            {str(k): v for k, v in cfg.targets.hop_distribution.items()},
        )

        # Initialize checkpoint manager.
        ckpt_dir_str = cfg.micro_batch.checkpoint_dir.strip()
        if not ckpt_dir_str:
            ckpt_dir = Path(cfg.output.dir) / ".checkpoints"
        else:
            ckpt_dir = Path(ckpt_dir_str)

        config_hash = compute_config_hash(
            total_samples=target,
            corpus_id=cfg.corpus.corpus_id,
            primary_type_distribution=cfg.targets.primary_type_distribution,
            reasoning_mode_distribution=(cfg.targets.reasoning_mode_distribution),
            hop_distribution={
                str(k): v for k, v in cfg.targets.hop_distribution.items()
            },
            acceptance_policy="quota_aware_v1",
        )
        ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, config_hash=config_hash)

        # Resume from checkpoint if available.
        resume = ckpt_mgr.resume_state() if cfg.micro_batch.resume else None
        if resume and resume.passed_items:
            all_passed = resume.passed_items
            accepted_type_counts = dict(resume.accepted_by_type)
            accepted_mode_counts = dict(resume.accepted_by_reasoning_mode)
            accepted_hop_counts = dict(resume.accepted_by_hop_count)
            batch_idx = resume.completed_batch_count
            iteration_count = resume.iteration_count
            _print_progress(
                f"  Resumed from checkpoint: {len(all_passed)} passed,"
                f" {batch_idx} batches completed,"
                f" iteration {iteration_count}",
                verbose=cfg.verbose,
            )
        else:
            all_passed: list[GeneratedQA] = []
            accepted_type_counts = {qa_type: 0 for qa_type in target_type_counts}
            accepted_mode_counts: dict[str, int] = {}
            accepted_hop_counts: dict[str, int] = {}
            batch_idx = 0
            iteration_count = 0

        all_rejected: list[GeneratedQA] = []
        all_raw: list[GeneratedQA] = []
        total_regens = 0
        consecutive_empty = 0

        # Early dedup: catch near-duplicates before the expensive filter chain.
        incremental_dedup = IncrementalDeduplicator(
            similarity_threshold=cfg.dedup.similarity_threshold,
            ngram_size=cfg.dedup.ngram_size,
        )
        if all_passed:
            incremental_dedup.register_accepted(all_passed)

        pbar = _tqdm(
            total=target,
            initial=len(all_passed),
            desc="[5/6] Work queue",
            unit="samples",
        )

        max_parallel = max_parallel_cfg

        if max_parallel <= 1:
            # Sequential processing.
            while True:
                tasks = compute_next_batch(
                    target_type_counts=target_type_counts,
                    accepted_type_counts=accepted_type_counts,
                    target_mode_counts=target_mode_counts,
                    accepted_mode_counts=accepted_mode_counts,
                    target_hop_counts=target_hop_counts,
                    accepted_hop_counts=accepted_hop_counts,
                    batch_size=batch_size,
                    source=source,
                    cfg=cfg,
                    iteration_count=iteration_count,
                    profile=profile,
                )
                if not tasks or iteration_count >= max_iterations:
                    break

                passed, rejected, raw, regens = await self._aprocess_batch(
                    tasks,
                    generator=generator,
                    guard_filter=guard_filter,
                    filter_stage_names=filter_stage_names,
                    filter_chain=filter_chain,
                    transformer=transformer,
                    context=context,
                    incremental_dedup=incremental_dedup,
                )

                accepted, quota_rejected = _accept_items_under_type_quota(
                    passed,
                    accepted_type_counts=accepted_type_counts,
                    target_type_counts=target_type_counts,
                )
                batch_rejected = [*rejected, *quota_rejected]

                if len(accepted) == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0

                if consecutive_empty >= 5:
                    logger.warning(
                        "No items accepted in 5 consecutive batches, stopping early"
                    )
                    all_rejected.extend(batch_rejected)
                    all_raw.extend(raw)
                    total_regens += regens
                    break

                _update_subdistribution_counts(
                    accepted,
                    accepted_mode_counts=accepted_mode_counts,
                    accepted_hop_counts=accepted_hop_counts,
                )

                all_passed.extend(accepted)
                incremental_dedup.register_accepted(accepted)
                all_rejected.extend(batch_rejected)
                all_raw.extend(raw)
                total_regens += regens
                pbar.update(len(accepted))

                ckpt_mgr.save_batch(
                    batch_idx,
                    accepted,
                    batch_rejected,
                    regens,
                    accepted_by_type=accepted_type_counts,
                    accepted_by_reasoning_mode=accepted_mode_counts,
                    accepted_by_hop_count=accepted_hop_counts,
                    iteration_count=iteration_count,
                )
                batch_idx += 1
                iteration_count += 1

                _print_progress(
                    f"  Batch {batch_idx}: {len(raw)} generated"
                    f" → {len(accepted)} accepted,"
                    f" {len(batch_rejected)} rejected,"
                    f" {regens} regenerated",
                    verbose=cfg.verbose,
                )

                pipeline_metrics = context.metrics
                if pipeline_metrics is not None:
                    cum_gen = len(all_raw)
                    cum_acc = len(all_passed)
                    batch_m = BatchMetrics(
                        batch_index=batch_idx - 1,
                        generated_count=len(raw),
                        accepted_count=len(accepted),
                        rejected_count=len(batch_rejected),
                        regeneration_count=regens,
                        acceptance_rate=(len(accepted) / len(raw) if raw else 0.0),
                        cumulative_acceptance_rate=(
                            cum_acc / cum_gen if cum_gen > 0 else 0.0
                        ),
                        cumulative_fill_rate=(cum_acc / target if target > 0 else 0.0),
                    )
                    pipeline_metrics.batch_history.append(batch_m)

                    remaining = target - cum_acc
                    if remaining > batch_size and should_early_stop(
                        pipeline_metrics.batch_history
                    ):
                        logger.warning(
                            "Acceptance rate below threshold for recent batches, stopping early"
                        )
                        break
        else:
            # Parallel processing with asyncio tasks on one event loop.
            import copy
            import threading

            # The collect/submit critical sections below contain no awaits, so on
            # a single event loop the lock never contends; kept to keep the
            # shared-state mutations explicit (and to survive a future refactor).
            lock = threading.Lock()

            # futures[task] = (batch_idx, batch_context, in_flight_types)
            # in_flight_types lets us decrement the shared in-flight counters
            # on collection without recomputing from the task list.
            futures: dict[Any, tuple[int, PipelineContext, dict[str, int]]] = {}
            in_flight_type_counts: dict[str, int] = {t: 0 for t in target_type_counts}

            def _submit_next() -> None:
                nonlocal batch_idx, iteration_count
                while len(futures) < max_parallel and iteration_count < max_iterations:
                    # Subtract in-flight work from remaining quota so concurrent
                    # batches don't collectively overshoot a type's target.
                    effective_accepted_types = {
                        t: accepted_type_counts.get(t, 0)
                        + in_flight_type_counts.get(t, 0)
                        for t in target_type_counts
                    }
                    tasks = compute_next_batch(
                        target_type_counts=target_type_counts,
                        accepted_type_counts=effective_accepted_types,
                        target_mode_counts=target_mode_counts,
                        accepted_mode_counts=accepted_mode_counts,
                        target_hop_counts=target_hop_counts,
                        accepted_hop_counts=accepted_hop_counts,
                        batch_size=batch_size,
                        source=source,
                        cfg=cfg,
                        iteration_count=iteration_count,
                        profile=profile,
                    )
                    if not tasks:
                        break
                    batch_in_flight: dict[str, int] = {}
                    for t in tasks:
                        batch_in_flight[t.qa_type] = (
                            batch_in_flight.get(t.qa_type, 0) + 1
                        )
                    for k, v in batch_in_flight.items():
                        in_flight_type_counts[k] = in_flight_type_counts.get(k, 0) + v
                    batch_context = copy.copy(context)
                    batch_context.state = dict(context.state)
                    if context.metrics is not None:
                        batch_context.metrics = PipelineMetrics()
                    fut = asyncio.create_task(
                        self._aprocess_batch(
                            tasks,
                            generator=generator,
                            guard_filter=guard_filter,
                            filter_stage_names=filter_stage_names,
                            filter_chain=filter_chain,
                            transformer=transformer,
                            context=batch_context,
                            incremental_dedup=incremental_dedup,
                        )
                    )
                    futures[fut] = (batch_idx, batch_context, batch_in_flight)
                    batch_idx += 1
                    iteration_count += 1

            _submit_next()

            while futures:
                done = [f for f in futures if f.done()]
                if not done:
                    done_set, _ = await asyncio.wait(
                        futures.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    done = list(done_set)

                def _collect_result(
                    fut: Any, b_idx: int, batch_ctx: PipelineContext
                ) -> None:
                    nonlocal consecutive_empty, total_regens
                    passed, rejected, raw, regens = fut.result()

                    with lock:
                        accepted, quota_rejected = _accept_items_under_type_quota(
                            passed,
                            accepted_type_counts=(accepted_type_counts),
                            target_type_counts=target_type_counts,
                        )
                        batch_rejected = [
                            *rejected,
                            *quota_rejected,
                        ]

                        if len(accepted) == 0:
                            consecutive_empty += 1
                        else:
                            consecutive_empty = 0

                        _update_subdistribution_counts(
                            accepted,
                            accepted_mode_counts=accepted_mode_counts,
                            accepted_hop_counts=accepted_hop_counts,
                        )

                        all_passed.extend(accepted)
                        incremental_dedup.register_accepted(accepted)
                        all_rejected.extend(batch_rejected)
                        all_raw.extend(raw)
                        total_regens += regens
                        pbar.update(len(accepted))
                        ckpt_mgr.save_batch(
                            b_idx,
                            accepted,
                            batch_rejected,
                            regens,
                            accepted_by_type=accepted_type_counts,
                            accepted_by_reasoning_mode=(accepted_mode_counts),
                            accepted_by_hop_count=(accepted_hop_counts),
                            iteration_count=iteration_count,
                        )

                        p_metrics = context.metrics
                        batch_metrics_obj = batch_ctx.metrics
                        if p_metrics is not None and batch_metrics_obj is not None:
                            p_metrics.merge_stage_metrics(batch_metrics_obj)
                            c_gen = len(all_raw)
                            c_acc = len(all_passed)
                            batch_m = BatchMetrics(
                                batch_index=b_idx,
                                generated_count=len(raw),
                                accepted_count=len(accepted),
                                rejected_count=len(batch_rejected),
                                regeneration_count=regens,
                                acceptance_rate=(
                                    len(accepted) / len(raw) if raw else 0.0
                                ),
                                cumulative_acceptance_rate=(
                                    c_acc / c_gen if c_gen > 0 else 0.0
                                ),
                                cumulative_fill_rate=(
                                    c_acc / target if target > 0 else 0.0
                                ),
                            )
                            p_metrics.batch_history.append(batch_m)

                    _print_progress(
                        f"  Batch {b_idx + 1}: {len(raw)} generated"
                        f" → {len(accepted)} accepted,"
                        f" {len(batch_rejected)} rejected,"
                        f" {regens} regenerated",
                        verbose=cfg.verbose,
                    )

                for fut in done:
                    b_idx, batch_ctx, in_flight_types = futures.pop(fut)
                    with lock:
                        for k, v in in_flight_types.items():
                            in_flight_type_counts[k] = max(
                                0, in_flight_type_counts.get(k, 0) - v
                            )
                    _collect_result(fut, b_idx, batch_ctx)

                with lock:
                    _should_stop = consecutive_empty >= 5
                    remaining_count = target - len(all_passed)
                    if (
                        not _should_stop
                        and remaining_count > batch_size
                        and context.metrics is not None
                        and should_early_stop(context.metrics.batch_history)
                    ):
                        _should_stop = True

                if _should_stop:
                    # Drain remaining in-flight tasks.
                    for fut, (b_idx, batch_ctx, _in_flight) in list(futures.items()):
                        await fut  # wait
                        _collect_result(fut, b_idx, batch_ctx)
                    futures.clear()
                    logger.warning(
                        "Acceptance rate below threshold or no items "
                        "accepted in consecutive batches, stopping early"
                    )
                    break

                with lock:
                    _submit_next()

        pbar.close()

        # Clean up checkpoints on success (unless keep_checkpoints).
        if not cfg.micro_batch.keep_checkpoints:
            ckpt_mgr.cleanup()

        context["raw_candidates"] = [item.qa for item in all_raw]
        context["raw_count"] = len(all_raw)

        return all_passed, all_rejected, total_regens, all_raw


def _filter_and_sample_seeds(
    source: Any,
    n: int,
    min_chars: int,
    profile: CorpusProfile | None,
    rng: random.Random,
) -> list[Any]:
    """Sample seeds from the full corpus, filtering out bottom-quartile chunks.

    Scores chunks on-the-fly (using cached scores when available) and
    excludes those below the p25 threshold. Falls back to unfiltered
    sampling when profile is unavailable or the eligible pool is too small.
    """
    if (
        profile is None
        or not profile.chunk_suitability_scores
        or profile.census is None
    ):
        return source.sample_chunks(n, min_chars=min_chars) or []

    # Adaptive threshold: bottom 25% of profile pool scores
    scores = sorted(profile.chunk_suitability_scores.values())
    threshold = scores[len(scores) // 4] if len(scores) >= 4 else 0.0

    collection = getattr(source, "collection", None)
    if collection is not None:
        # In-memory: score all chunks on-the-fly, filter by threshold
        candidates = [c for c in collection.chunks if len(c.content) >= min_chars]
        eligible = [
            c
            for c in candidates
            if (
                profile.chunk_suitability_scores.get(c.hash)
                or compute_chunk_suitability(c, profile.census, profile)
            )
            > threshold
        ]
        if len(eligible) < n:
            return rng.sample(candidates, min(n, len(candidates))) if candidates else []
        return rng.sample(eligible, n)

    # API backend: fetch all chunks once, score, and cache eligible pool
    if not hasattr(profile, "_api_eligible_cache"):
        chunk_count = 0
        try:
            chunk_count = source.get_chunk_count()
        except (AttributeError, Exception):
            pass
        if chunk_count > 0:
            logger.info(
                "Fetching all %d chunks from API backend for scoring...",
                chunk_count,
            )
            all_chunks = source.sample_chunks(chunk_count, min_chars=min_chars)
        else:
            all_chunks = source.sample_chunks(n * 2, min_chars=min_chars)

        if not all_chunks:
            return []

        eligible = [
            c
            for c in all_chunks
            if compute_chunk_suitability(c, profile.census, profile) > threshold
        ]
        profile._api_eligible_cache = eligible if eligible else all_chunks
        logger.info(
            "Cached %d eligible chunks (of %d total, threshold=%.3f)",
            len(profile._api_eligible_cache),
            len(all_chunks),
            threshold,
        )

    cached = profile._api_eligible_cache
    if not cached:
        return source.sample_chunks(n, min_chars=min_chars) or []
    return rng.sample(cached, min(n, len(cached)))


def compute_next_batch(
    *,
    target_type_counts: dict[str, int],
    accepted_type_counts: dict[str, int],
    target_mode_counts: dict[str, int],
    accepted_mode_counts: dict[str, int],
    target_hop_counts: dict[str, int],
    accepted_hop_counts: dict[str, int],
    batch_size: int,
    source: Any,
    cfg: PipelineConfig,
    iteration_count: int,
    profile: CorpusProfile | None = None,
) -> list[GenerationTask]:
    """Compute the next batch of generation tasks based on remaining quotas."""
    from castform.rag.qa_generation.pipeline_config import (
        allocate_largest_remainder_generic,
    )

    remaining = {
        t: max(0, target - accepted_type_counts.get(t, 0))
        for t, target in target_type_counts.items()
    }
    total_remaining = sum(remaining.values())
    if total_remaining == 0:
        return []

    n = min(batch_size, total_remaining)

    # Proportional allocation by target type distribution. A greedy "fill
    # multi_hop first" policy overshoots under parallel submission because
    # in-flight batches all compute against a stale accepted_type_counts — each
    # wave of k parallel batches would launch k*batch_size multi_hop tasks and
    # starve lookup entirely until multi_hop is saturated.
    type_dist = {
        t: w
        for t, w in cfg.targets.primary_type_distribution.items()
        if t in target_type_counts
    }
    if not type_dist:
        type_dist = {t: 1.0 for t in target_type_counts}

    raw_alloc = allocate_largest_remainder_generic(n, type_dist)

    # Cap each type by its remaining quota; redistribute any overflow to types
    # that still have room so the batch stays at size n when possible.
    allocated: dict[str, int] = {}
    overflow = 0
    for t, count in raw_alloc.items():
        cap = remaining.get(t, 0)
        if count > cap:
            overflow += count - cap
            allocated[t] = cap
        else:
            allocated[t] = count

    while overflow > 0:
        room = {t: max(0, remaining.get(t, 0) - allocated.get(t, 0)) for t in type_dist}
        available = {t: float(v) for t, v in room.items() if v > 0}
        if not available:
            break
        give = min(overflow, int(sum(available.values())))
        if give <= 0:
            break
        extra = allocate_largest_remainder_generic(give, available)
        given = 0
        for t, c in extra.items():
            if c > 0:
                allocated[t] = allocated.get(t, 0) + c
                given += c
        if given == 0:
            break
        overflow -= given

    n_lookup = allocated.get("lookup", 0)
    n_multi_hop = allocated.get("multi_hop", 0)

    # For multi_hop: allocate reasoning_mode and hop_count proportionally
    # to remaining sub-distribution counts.
    remaining_mode = {
        m: max(0, target_mode_counts.get(m, 0) - accepted_mode_counts.get(m, 0))
        for m in target_mode_counts
    }
    remaining_hop = {
        h: max(0, target_hop_counts.get(h, 0) - accepted_hop_counts.get(h, 0))
        for h in target_hop_counts
    }

    mode_total = sum(remaining_mode.values())
    if mode_total > 0:
        mode_dist = {m: c / mode_total for m, c in remaining_mode.items() if c > 0}
    else:
        mode_dist = dict(cfg.targets.reasoning_mode_distribution)
    mode_counts = allocate_largest_remainder_generic(n_multi_hop, mode_dist)

    hop_total = sum(remaining_hop.values())
    if hop_total > 0:
        hop_dist = {h: c / hop_total for h, c in remaining_hop.items() if c > 0}
    else:
        hop_dist = {str(k): v for k, v in cfg.targets.hop_distribution.items()}
    hop_counts = allocate_largest_remainder_generic(n_multi_hop, hop_dist)

    # Build mode and hop pools.
    rng = random.Random(cfg.random_seed + iteration_count + 1)

    mode_pool: list[str] = []
    for mode, count in sorted(mode_counts.items()):
        mode_pool.extend([mode] * count)
    rng.shuffle(mode_pool)

    hop_pool: list[int] = []
    for hop_str, count in sorted(hop_counts.items()):
        hop_pool.extend([int(hop_str)] * count)
    rng.shuffle(hop_pool)

    # Sample fresh seed chunks.
    total_tasks = n_lookup + n_multi_hop
    new_seeds = _filter_and_sample_seeds(
        source, total_tasks, cfg.corpus.min_chunk_chars, profile, rng
    )
    if not new_seeds:
        seed_ids = [f"fallback_{i}" for i in range(total_tasks)]
    else:
        seed_ids = [c.hash for c in new_seeds]

    tasks: list[GenerationTask] = []
    idx = 0
    multi_hop_idx = 0

    for _ in range(n_lookup):
        seed_id = seed_ids[idx % len(seed_ids)] if seed_ids else "fallback_0"
        idx += 1
        tasks.append(
            GenerationTask(
                task_id=f"iter_{iteration_count}_{len(tasks):05d}",
                qa_type="lookup",
                target_hop_count=1,
                seed_chunk_id=seed_id,
                reasoning_mode="",
            )
        )

    for _ in range(n_multi_hop):
        seed_id = seed_ids[idx % len(seed_ids)] if seed_ids else "fallback_0"
        idx += 1
        mode = mode_pool[multi_hop_idx] if multi_hop_idx < len(mode_pool) else "factual"
        hop = hop_pool[multi_hop_idx] if multi_hop_idx < len(hop_pool) else 2
        multi_hop_idx += 1
        tasks.append(
            GenerationTask(
                task_id=f"iter_{iteration_count}_{len(tasks):05d}",
                qa_type="multi_hop",
                target_hop_count=hop,
                seed_chunk_id=seed_id,
                reasoning_mode=mode,
            )
        )

    rng.shuffle(tasks)
    return tasks


def _update_subdistribution_counts(
    accepted_items: list[GeneratedQA],
    *,
    accepted_mode_counts: dict[str, int],
    accepted_hop_counts: dict[str, int],
) -> None:
    """Update reasoning_mode and hop_count counts for accepted items."""
    for item in accepted_items:
        effective_type, _, _ = _resolve_effective_qa_type(item)
        if effective_type != "multi_hop":
            continue
        mode = (
            str(item.generation_metadata.get("reasoning_mode", "")).strip()
            or str(item.qa.get("reasoning_mode", "")).strip()
        )
        if mode:
            accepted_mode_counts[mode] = accepted_mode_counts.get(mode, 0) + 1
        ref_chunks = list(item.qa.get("reference_chunks", []) or [])
        hop_key = str(len(ref_chunks))
        accepted_hop_counts[hop_key] = accepted_hop_counts.get(hop_key, 0) + 1


def run_pipeline(
    cfg: PipelineConfig,
    *,
    source_factory: Callable[[PipelineConfig], Any] | None = None,
    rollout_client_factory: Callable[[PipelineConfig], RolloutClient] | None = None,
) -> dict[str, Any]:
    """Run Pipeline with a fully constructed config."""
    pipeline = Pipeline(
        cfg,
        source_factory=source_factory,
        rollout_client_factory=rollout_client_factory,
    )
    return pipeline.run()


def run_pipeline_from_config(
    config_path: str | Path,
    *,
    source_factory: Callable[[PipelineConfig], Any] | None = None,
    rollout_client_factory: Callable[[PipelineConfig], RolloutClient] | None = None,
) -> dict[str, Any]:
    """Load config YAML and run Pipeline."""
    cfg = load_pipeline_config(config_path)
    return run_pipeline(
        cfg,
        source_factory=source_factory,
        rollout_client_factory=rollout_client_factory,
    )
