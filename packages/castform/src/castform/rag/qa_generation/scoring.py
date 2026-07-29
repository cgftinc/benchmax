"""Quality scoring for generated QA items.

Extracts scoring signals from filter verdicts during the filter pipeline,
computes composite quality scores, and provides quality-ranked selection
for quota acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from castform.rag.qa_generation.generated_qa import GeneratedQA

# Keys to harvest from each filter stage's FilterVerdict.metadata.
# These must match the actual keys written by each filter implementation.
_SCORE_EXTRACTION_KEYS: dict[str, list[str]] = {
    "grounding_llm": [
        "confidence",
        "judge_answerable",
        "supporting_chunk_ids",
        "reference_chunk_count",
    ],
    "retrieval_too_easy_llm": [
        "ref_overlap_ratio",
        "confidence",
        "judge_reason_tag",
        "too_easy_overlap_triggered",
        "top_score",
    ],
    "hop_count_validity": [
        "hop_count_validated",
        "redundant_chunks",
        "difficulty_score",
    ],
}


@dataclass
class ScoringConfig:
    """Weights and thresholds for quality scoring."""

    min_quality_score: float = 0.0  # 0.0 = no threshold
    grounding_weight: float = 0.4
    retrieval_difficulty_weight: float = 0.3
    hop_validity_weight: float = 0.3


def extract_filter_scores(items: list[GeneratedQA], stage_name: str) -> None:
    """Harvest scoring signals from filter_verdict.metadata before
    the next filter overwrites it."""
    keys = _SCORE_EXTRACTION_KEYS.get(stage_name)
    if not keys:
        return
    for item in items:
        if item.filter_verdict is None:
            continue
        if not item.is_passed:
            continue
        meta = (
            item.filter_verdict.metadata if isinstance(item.filter_verdict.metadata, dict) else {}
        )
        scores = {k: meta[k] for k in keys if k in meta}
        if scores:
            filter_scores: dict[str, Any] = item.generation_metadata.setdefault("filter_scores", {})
            filter_scores[stage_name] = scores


def compute_eval_scores(item: GeneratedQA, cfg: ScoringConfig) -> dict[str, float]:
    """Compute quality scores from extracted filter signals."""
    filter_scores = item.generation_metadata.get("filter_scores", {})
    scores: dict[str, float] = {}

    grounding = filter_scores.get("grounding_llm", {})
    if "confidence" in grounding:
        scores["grounding"] = float(grounding["confidence"])

    retrieval = filter_scores.get("retrieval_too_easy_llm", {})
    if "ref_overlap_ratio" in retrieval:
        scores["retrieval_difficulty"] = 1.0 - float(retrieval["ref_overlap_ratio"])

    hop = filter_scores.get("hop_count_validity", {})
    if "hop_count_validated" in hop:
        if hop["hop_count_validated"]:
            scores["hop_validity"] = 1.0
        elif hop.get("demoted"):
            # Demoted: redundant chunks were removed but ≥2 essential remain.
            # Score higher than unvalidated (0.5) since the remaining chunks
            # were confirmed essential by the filter.
            scores["hop_validity"] = 0.75
        else:
            scores["hop_validity"] = 0.5
    if "difficulty_score" in hop:
        scores["hop_difficulty"] = float(hop["difficulty_score"])

    if scores:
        weights = {
            "grounding": cfg.grounding_weight,
            "retrieval_difficulty": cfg.retrieval_difficulty_weight,
            "hop_validity": cfg.hop_validity_weight,
        }
        available = {k: v for k, v in weights.items() if k in scores}
        total_weight = sum(available.values())
        if total_weight > 0:
            scores["composite"] = sum(scores[k] * available[k] for k in available) / total_weight

    return scores
