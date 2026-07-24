"""Retrieval-quality metrics for the Neon golden eval (hit@k, MRR, ablation).

Pure ranking metrics — every function takes an already-ranked list of chunk ids
(best-first) and the gold/decoy id sets, so the module has no database or network
dependency and is unit-testable in isolation. The live integration gate supplies
the ranked ids from real Neon retrieval; these functions turn them into the
per-mode hit@k / MRR and the lexical-ablation delta that the frozen golden set is
judged against (:mod:`castform.rag.corpus.neon.eval_schema`).

Conventions
-----------
* ``hit@k`` per record is a hit if ANY gold id appears in the top ``k`` — the
  standard "was a relevant chunk retrieved" definition; aggregated hit@k is the
  mean over records.
* ``MRR@k`` uses the reciprocal of the 1-based rank of the FIRST gold id in the
  top ``k`` (0 if none), aggregated as the mean.
* A ``decoy violation`` is any decoy id appearing in the top ``k``; the gate
  requires zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from castform.rag.corpus.neon.eval_schema import (
    DEFAULT_THRESHOLDS,
    LEXICAL_ABLATION_MIN_DELTA,
    NeonEvalRecord,
    NeonEvalThresholds,
)
from castform.rag.corpus.search_schema.search_types import SearchMode


def reciprocal_rank(gold: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Reciprocal of the 1-based rank of the first gold id within the top ``k``.

    Returns ``0.0`` when no gold id is in the top ``k``.

    Args:
        gold: Chunk-hash ids that count as correct.
        ranked_ids: Retrieved chunk-hash ids, best-first.
        k: Retrieval depth to score at.
    """
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def is_hit(gold: set[str], ranked_ids: Sequence[str], k: int) -> bool:
    """Whether any gold id appears within the top ``k`` retrieved ids."""
    return any(chunk_id in gold for chunk_id in ranked_ids[:k])


def has_decoy(decoys: set[str], ranked_ids: Sequence[str], k: int) -> bool:
    """Whether any decoy id appears within the top ``k`` (a gate violation)."""
    return any(chunk_id in decoys for chunk_id in ranked_ids[:k])


@dataclass(frozen=True)
class RecordEval:
    """The scored outcome of one eval record against one ranked result list.

    Args:
        capability: The record's capability label (carried through for reporting).
        search_mode: The mode the record exercises.
        hit: Whether a gold id was in the top ``k``.
        reciprocal_rank: Reciprocal rank of the first gold id (0 if missed).
        decoy_violation: Whether a decoy id was in the top ``k``.
    """

    capability: str
    search_mode: SearchMode
    hit: bool
    reciprocal_rank: float
    decoy_violation: bool


def evaluate_record(
    record: NeonEvalRecord, ranked_ids: Sequence[str], k: int
) -> RecordEval:
    """Score one :class:`NeonEvalRecord` against a ranked id list at depth ``k``."""
    gold = set(record.gold_chunk_hashes)
    decoys = set(record.decoy_chunk_hashes)
    return RecordEval(
        capability=record.capability,
        search_mode=record.search_mode,
        hit=is_hit(gold, ranked_ids, k),
        reciprocal_rank=reciprocal_rank(gold, ranked_ids, k),
        decoy_violation=has_decoy(decoys, ranked_ids, k),
    )


@dataclass(frozen=True)
class ModeMetrics:
    """Aggregate metrics for one mode over its eval records.

    Args:
        mode: The retrieval mode these metrics cover.
        n: Number of records scored.
        hit_at_k: Mean hit@k over the records.
        mrr_at_k: Mean reciprocal rank over the records.
        decoy_violations: Count of records whose top ``k`` surfaced a decoy.
        k: The retrieval depth these metrics were measured at.
    """

    mode: SearchMode
    n: int
    hit_at_k: float
    mrr_at_k: float
    decoy_violations: int
    k: int

    def passes(self, thresholds: NeonEvalThresholds) -> bool:
        """Whether hit@k and MRR clear ``thresholds`` and no decoy was surfaced."""
        return (
            self.hit_at_k >= thresholds.hit_at_k
            and self.mrr_at_k >= thresholds.mrr_at_k
            and self.decoy_violations == 0
        )


def aggregate_by_mode(evals: Sequence[RecordEval], k: int) -> dict[SearchMode, ModeMetrics]:
    """Aggregate per-record evals into per-mode :class:`ModeMetrics`.

    Args:
        evals: Scored records (mixed modes).
        k: Retrieval depth the evals were scored at (recorded on the metrics).
    """
    modes: dict[SearchMode, list[RecordEval]] = {}
    for ev in evals:
        modes.setdefault(ev.search_mode, []).append(ev)
    out: dict[SearchMode, ModeMetrics] = {}
    for mode, items in modes.items():
        n = len(items)
        out[mode] = ModeMetrics(
            mode=mode,
            n=n,
            hit_at_k=sum(e.hit for e in items) / n,
            mrr_at_k=sum(e.reciprocal_rank for e in items) / n,
            decoy_violations=sum(e.decoy_violation for e in items),
            k=k,
        )
    return out


def hit_at_k(evals: Sequence[RecordEval]) -> float:
    """Mean hit@k over a set of already-scored records (0 for an empty set)."""
    return sum(e.hit for e in evals) / len(evals) if evals else 0.0


@dataclass(frozen=True)
class AblationResult:
    """Lexical-ablation comparison for a set of records.

    Confirms the records are not keyword-solvable: their gold is recalled under
    the primary (semantic) mode but MISSED under a lexical-only (BM25) retrieval,
    so ``delta`` (primary minus lexical hit@k) must clear
    :data:`LEXICAL_ABLATION_MIN_DELTA`.

    Args:
        n: Number of records compared.
        primary_hit_at_k: hit@k under the records' own mode.
        lexical_hit_at_k: hit@k under lexical-only retrieval of the same queries.
        delta: ``primary_hit_at_k - lexical_hit_at_k``.
    """

    n: int
    primary_hit_at_k: float
    lexical_hit_at_k: float
    delta: float

    @property
    def passes(self) -> bool:
        """Whether the ablation delta clears the frozen minimum."""
        return self.delta >= LEXICAL_ABLATION_MIN_DELTA


def lexical_ablation(
    primary_evals: Sequence[RecordEval], lexical_evals: Sequence[RecordEval]
) -> AblationResult:
    """Compare primary-mode recall against lexical-only recall of the same records.

    Both sequences must score the SAME records in the SAME order (one under the
    record's mode, one under a forced lexical retrieval of the record's query).
    """
    if len(primary_evals) != len(lexical_evals):
        raise ValueError("primary and lexical eval sets must align 1:1")
    primary = hit_at_k(primary_evals)
    lexical = hit_at_k(lexical_evals)
    return AblationResult(
        n=len(primary_evals),
        primary_hit_at_k=primary,
        lexical_hit_at_k=lexical,
        delta=primary - lexical,
    )


def thresholds_for(mode: SearchMode) -> NeonEvalThresholds:
    """Return the frozen default thresholds for ``mode``."""
    return DEFAULT_THRESHOLDS[mode]
