"""Corpus-aware parameter auto-tuning and warnings for Pipeline."""

from __future__ import annotations

import logging
from typing import Any

from benchmax.rag.qa_generation.corpus_profile import CorpusMetadataCensus, CorpusProfile

logger = logging.getLogger(__name__)


def emit_corpus_warnings(
    census: CorpusMetadataCensus,
    profile: CorpusProfile,
    cfg: Any,
) -> list[str]:
    """Return warnings when config looks infeasible for the corpus."""
    warnings: list[str] = []

    if census.chunk_count == 0:
        warnings.append("Corpus has 0 chunks — auto-tune cannot assess feasibility.")
        return warnings

    suitability_scores: dict[str, float] = getattr(profile, "chunk_suitability_scores", {})
    suitable_fraction = sum(1 for s in suitability_scores.values() if s > 0.4) / max(
        len(suitability_scores), 1
    )

    # Low date prevalence + temporal mode
    mode_dist = dict(cfg.targets.reasoning_mode_distribution)
    temporal_weight = mode_dist.get("temporal", 0)
    if temporal_weight > 0 and census.date_prevalence < 0.3:
        date_pct = int(census.date_prevalence * 100)
        if census.date_prevalence < 0.05:
            new_temporal_pct = 0
        else:
            new_temporal_pct = int(min(temporal_weight, census.date_prevalence) * 100)
        warnings.append(
            f"Only {date_pct}% of chunks have date metadata — "
            f"temporal reasoning mode reduced to {new_temporal_pct}%."
        )

    # Low linkability + multi-hop ratio
    linkability = _compute_linkability(census)
    multi_hop_ratio = dict(cfg.targets.primary_type_distribution).get("multi_hop", 0.7)
    adjusted_mh = max(0.3, multi_hop_ratio * linkability)

    if adjusted_mh < multi_hop_ratio - 0.001:
        scores = sorted(suitability_scores.values())
        p50 = scores[len(scores) // 2] if scores else 0.0
        warnings.append(
            f"Corpus has low linkability (suitability p50={p50:.2f}) — "
            f"multi-hop ratio reduced to {adjusted_mh:.0%}."
        )

    # Effective pool vs total_samples
    effective_pool = int(census.chunk_count * suitable_fraction)
    total_samples = cfg.targets.total_samples
    if effective_pool > 0 and total_samples > effective_pool * 2:
        suitable_pct = int(suitable_fraction * 100)
        cap = effective_pool * 2
        warnings.append(
            f"Effective pool is {effective_pool} chunks ({suitable_pct}% suitable) — "
            f"total_samples capped at {cap}."
        )

    # Single-file corpus
    if not census.multi_file_corpus:
        warnings.append(
            "Corpus appears to be from a single source file. "
            "Multi-hop cross-document questions may have limited diversity."
        )

    return warnings


def auto_tune(
    census: CorpusMetadataCensus,
    profile: CorpusProfile,
    cfg: Any,
) -> dict[str, Any]:
    """Derive sensible defaults from corpus characteristics.

    Returns dict of adjusted values. Does NOT mutate cfg — caller applies.

    Adjustments:
    - 7a: Multi-hop ratio — graduated by metadata regime + entity density
    - 7b: Hop distribution — suitability-aware, with short-chunk gate preserved
    - 7c: Total samples — capped at 2× effective pool (suitable chunks)
    - 7d: Reasoning modes — temporal/sequential removed or capped by prevalence
    """
    adjustments: dict[str, Any] = {}

    if census.chunk_count == 0:
        return adjustments

    suitability_scores: dict[str, float] = getattr(profile, "chunk_suitability_scores", {})
    suitable_fraction = sum(1 for s in suitability_scores.values() if s > 0.4) / max(
        len(suitability_scores), 1
    )

    # --- 7a. Multi-hop ratio (graduated by regime + entity density) ---
    linkability = _compute_linkability(census)
    current_dist = dict(cfg.targets.primary_type_distribution)
    target_mh = current_dist.get("multi_hop", 0.7)
    adjusted_mh = max(0.3, target_mh * linkability)

    if abs(adjusted_mh - target_mh) > 0.001:
        adjustments["primary_type_distribution"] = {
            "lookup": round(1.0 - adjusted_mh, 2),
            "multi_hop": round(adjusted_mh, 2),
        }

    # --- 7b. Hop distribution (suitability-aware + short-chunk gate) ---
    hop_dist = {int(k): float(v) for k, v in cfg.targets.hop_distribution.items()}
    high_hop_weight = sum(v for k, v in hop_dist.items() if k >= 3)

    if suitability_scores and suitable_fraction < 0.3:
        # Very low suitability: cap at 2-hop
        adjustments["hop_distribution"] = {2: 1.0}
    elif (suitability_scores and suitable_fraction < 0.5 and high_hop_weight > 0.2) or (
        census.content_length_p50 < 600 and high_hop_weight > 0.2
    ):
        # Moderate suitability or short chunks: halve 3+ hop, redistribute to 2-hop
        adjustments["hop_distribution"] = _halve_high_hops(hop_dist)

    # --- 7c. Total samples — effective pool cap ---
    effective_pool = int(census.chunk_count * suitable_fraction)
    total_samples = cfg.targets.total_samples
    if effective_pool > 0 and total_samples > effective_pool * 2:
        adjustments["total_samples"] = effective_pool * 2

    # --- 7d. Reasoning mode distribution — prevalence-proportional ---
    mode_dist = dict(cfg.targets.reasoning_mode_distribution)
    mode_changed = False

    if "temporal" in mode_dist:
        if census.date_prevalence < 0.05:
            temporal_weight = mode_dist.pop("temporal")
            _redistribute_weight(mode_dist, temporal_weight)
            mode_changed = True
        elif census.date_prevalence < 0.3:
            current_temporal = mode_dist["temporal"]
            capped = min(current_temporal, census.date_prevalence)
            if capped < current_temporal:
                mode_dist["temporal"] = capped
                _redistribute_weight(mode_dist, current_temporal - capped, exclude="temporal")
                mode_changed = True

    if "sequential" in mode_dist:
        if census.sequential_prevalence < 0.05:
            seq_weight = mode_dist.pop("sequential")
            _redistribute_weight(mode_dist, seq_weight)
            mode_changed = True
        elif census.sequential_prevalence < 0.3:
            current_seq = mode_dist["sequential"]
            capped = min(current_seq, census.sequential_prevalence)
            if capped < current_seq:
                mode_dist["sequential"] = capped
                _redistribute_weight(mode_dist, current_seq - capped, exclude="sequential")
                mode_changed = True

    if mode_changed:
        total_weight = sum(mode_dist.values())
        if total_weight > 0:
            mode_dist = {k: round(v / total_weight, 4) for k, v in mode_dist.items()}
        adjustments["reasoning_mode_distribution"] = mode_dist

    return adjustments


def _redistribute_weight(
    mode_dist: dict[str, float],
    weight: float,
    exclude: str | None = None,
) -> None:
    """Distribute *weight* proportionally across remaining modes in-place."""
    remaining = {k: v for k, v in mode_dist.items() if k != exclude}
    total = sum(remaining.values())
    if total > 0:
        for k in remaining:
            mode_dist[k] += weight * (remaining[k] / total)
    else:
        mode_dist["factual"] = mode_dist.get("factual", 0) + weight


def _compute_linkability(census: CorpusMetadataCensus) -> float:
    """Derive linkability score from metadata regime and entity density."""
    regime_linkability = {"structured": 1.0, "mixed": 0.6, "unstructured": 0.3}
    linkability = regime_linkability.get(census.metadata_regime, 0.3)
    if census.entity_density_p25 > 0.0:
        linkability = min(1.0, linkability + 0.15)
    return linkability


def _halve_high_hops(hop_dist: dict[int, float]) -> dict[int, float]:
    """Halve the weight of 3+ hop types and redistribute to 2-hop."""
    adjusted: dict[int, float] = {}
    redistributed = 0.0
    for k, v in hop_dist.items():
        if k >= 3:
            new_v = v * 0.5
            redistributed += v - new_v
            adjusted[k] = new_v
        else:
            adjusted[k] = v
    adjusted[2] = adjusted.get(2, 0.0) + redistributed
    return adjusted


def should_early_stop(
    batch_history: list[Any],
    early_stop_rejection_rate: float = 0.85,
    early_stop_window: int = 3,
    min_generated_mass: int = 20,
) -> bool:
    """Check if recent batches indicate the pipeline is spinning.

    Uses sample-weighted acceptance over the window so tiny tail batches and
    infrastructure-failure batches (0 generated) don't dominate the signal.
    """
    if len(batch_history) < early_stop_window:
        return False
    recent = batch_history[-early_stop_window:]
    total_generated = sum(b.generated_count for b in recent)
    if total_generated < min_generated_mass:
        return False
    total_accepted = sum(b.accepted_count for b in recent)
    return (total_accepted / total_generated) < (1.0 - early_stop_rejection_rate)


def compute_batch_heuristics(
    total_samples: int,
) -> tuple[int, int]:
    """Return (batch_size, max_parallel_batches) from total_samples.

    Lookup table (tuned via parallelism probing — Corpora API ceiling ~20):
    total_samples  | batch_size | max_parallel
    <= 25          | 5          | 5
    26-50          | 10         | 5
    51-100         | 10         | 10
    101-500        | 50         | 10
    501-1000       | 50         | 20
    > 1000         | 50         | 20
    """
    if total_samples <= 25:
        return 5, 5
    if total_samples <= 50:
        return 10, 5
    if total_samples <= 100:
        return 10, 10
    if total_samples <= 500:
        return 50, 10
    return 50, 20
