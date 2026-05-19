"""Tests for auto_tune.py — graduated corpus-aware parameter adjustment."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from benchmax.rag.qa_generation.auto_tune import (
    auto_tune,
    compute_batch_heuristics,
    emit_corpus_warnings,
)
from benchmax.rag.qa_generation.corpus_profile import CorpusMetadataCensus, CorpusProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_census(
    *,
    metadata_regime: str = "structured",
    header_prevalence: float = 0.8,
    doc_id_prevalence: float = 0.9,
    date_prevalence: float = 0.5,
    sequential_prevalence: float = 0.5,
    content_length_p25: int = 500,
    content_length_p50: int = 800,
    content_length_p75: int = 1200,
    entity_density_p25: float = 1.0,
    entity_density_p50: float = 2.0,
    entity_density_p75: float = 3.0,
    lexical_diversity_p25: float = 0.4,
    lexical_diversity_p50: float = 0.6,
    lexical_diversity_p75: float = 0.8,
    chunk_count: int = 1000,
) -> CorpusMetadataCensus:
    return CorpusMetadataCensus(
        metadata_regime=metadata_regime,
        header_prevalence=header_prevalence,
        doc_id_prevalence=doc_id_prevalence,
        date_prevalence=date_prevalence,
        sequential_prevalence=sequential_prevalence,
        content_length_p25=content_length_p25,
        content_length_p50=content_length_p50,
        content_length_p75=content_length_p75,
        entity_density_p25=entity_density_p25,
        entity_density_p50=entity_density_p50,
        entity_density_p75=entity_density_p75,
        lexical_diversity_p25=lexical_diversity_p25,
        lexical_diversity_p50=lexical_diversity_p50,
        lexical_diversity_p75=lexical_diversity_p75,
        chunk_count=chunk_count,
    )


_DEFAULT_MODES: dict[str, float] = {"factual": 0.5, "temporal": 0.3, "sequential": 0.2}


def _make_profile(suitability_scores: dict[str, float] | None = None) -> CorpusProfile:
    profile = CorpusProfile()
    profile.chunk_suitability_scores = (  # type: ignore[attr-defined]
        suitability_scores if suitability_scores is not None else {}
    )
    return profile


def _high_profile() -> CorpusProfile:
    """Profile with 10 high-suitability chunks."""
    return _make_profile({f"c_{i}": 0.8 for i in range(10)})


def _make_cfg(
    *,
    total_samples: int = 100,
    multi_hop: float = 0.7,
    lookup: float = 0.3,
    hop_distribution: dict[Any, Any] | None = None,
    reasoning_mode_distribution: dict[str, float] | None = None,
) -> Any:
    if hop_distribution is None:
        hop_distribution = {2: 0.5, 3: 0.3, 4: 0.2}
    if reasoning_mode_distribution is None:
        reasoning_mode_distribution = {"factual": 0.5, "temporal": 0.3, "sequential": 0.2}
    return SimpleNamespace(
        targets=SimpleNamespace(
            total_samples=total_samples,
            primary_type_distribution={"multi_hop": multi_hop, "lookup": lookup},
            hop_distribution=hop_distribution,
            reasoning_mode_distribution=reasoning_mode_distribution,
        )
    )


def _mixed_scores(high: int, low: int) -> dict[str, float]:
    """Create suitability scores with `high` above 0.4 and `low` below."""
    scores: dict[str, float] = {}
    for i in range(high):
        scores[f"high_{i}"] = 0.8
    for i in range(low):
        scores[f"low_{i}"] = 0.2
    return scores


# ---------------------------------------------------------------------------
# 7a: Multi-hop ratio — graduated by regime × entity density
# ---------------------------------------------------------------------------


class TestAutoTuneGraduatedMultihop:
    def test_structured_no_entity_density_no_change(self):
        # linkability = 1.0, adjusted_mh = 0.7 * 1.0 = 0.7 → no change
        census = _make_census(metadata_regime="structured", entity_density_p25=0.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        assert "primary_type_distribution" not in result

    def test_structured_with_entity_density_still_no_change(self):
        # linkability = min(1.0, 1.0 + 0.15) = 1.0 → no change
        census = _make_census(metadata_regime="structured", entity_density_p25=1.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        assert "primary_type_distribution" not in result

    def test_mixed_no_entity_density(self):
        # linkability = 0.6, adjusted_mh = max(0.3, 0.7 * 0.6) = 0.42
        census = _make_census(metadata_regime="mixed", entity_density_p25=0.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        dist = result["primary_type_distribution"]
        assert abs(dist["multi_hop"] - 0.42) < 0.01
        assert abs(dist["lookup"] - 0.58) < 0.01

    def test_mixed_with_entity_density(self):
        # linkability = min(1.0, 0.6 + 0.15) = 0.75, adjusted_mh = max(0.3, 0.7 * 0.75) = 0.525
        census = _make_census(metadata_regime="mixed", entity_density_p25=1.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        dist = result["primary_type_distribution"]
        assert abs(dist["multi_hop"] - 0.525) < 0.01

    def test_unstructured_no_entity_density_hits_floor(self):
        # linkability = 0.3, adjusted_mh = max(0.3, 0.7 * 0.3) = max(0.3, 0.21) = 0.30
        census = _make_census(metadata_regime="unstructured", entity_density_p25=0.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        dist = result["primary_type_distribution"]
        assert abs(dist["multi_hop"] - 0.30) < 0.01

    def test_unstructured_with_entity_density(self):
        # linkability = min(1.0, 0.3 + 0.15) = 0.45, adjusted_mh = max(0.3, 0.7 * 0.45) = 0.315
        census = _make_census(metadata_regime="unstructured", entity_density_p25=1.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.7))
        dist = result["primary_type_distribution"]
        assert abs(dist["multi_hop"] - 0.315) < 0.01

    def test_floor_never_goes_below_0_3(self):
        # Even with a low target, floor holds
        census = _make_census(metadata_regime="unstructured", entity_density_p25=0.0)
        result = auto_tune(census, _make_profile(), _make_cfg(multi_hop=0.4))
        dist = result.get("primary_type_distribution", {})
        actual_mh = dist.get("multi_hop", 0.4)
        assert actual_mh >= 0.30

    def test_zero_chunk_count_returns_empty(self):
        census = _make_census(chunk_count=0)
        result = auto_tune(census, _make_profile(), _make_cfg())
        assert result == {}


# ---------------------------------------------------------------------------
# 7b: Hop distribution — suitability-aware thresholds
# ---------------------------------------------------------------------------


class TestAutoTuneHopDistribution:
    def test_very_low_suitable_fraction_caps_at_2hop(self):
        # suitable_fraction = 0 / 10 = 0.0 < 0.3 → cap at 2-hop
        scores = {f"c_{i}": 0.2 for i in range(10)}
        census = _make_census(content_length_p50=800)
        result = auto_tune(census, _make_profile(scores), _make_cfg())
        assert result["hop_distribution"] == {2: 1.0}

    def test_moderate_suitable_fraction_halves_high_hops(self):
        # suitable_fraction = 4/10 = 0.4 → 0.3 <= 0.4 < 0.5 → halve 3+ hops
        scores = _mixed_scores(high=4, low=6)
        census = _make_census(content_length_p50=800)
        cfg = _make_cfg(hop_distribution={2: 0.5, 3: 0.3, 4: 0.2})
        result = auto_tune(census, _make_profile(scores), cfg)
        hop = result["hop_distribution"]
        # 3-hop: 0.3 * 0.5 = 0.15; 4-hop: 0.2 * 0.5 = 0.10; 2-hop: 0.5 + 0.25 = 0.75
        assert abs(hop[2] - 0.75) < 0.01
        assert abs(hop[3] - 0.15) < 0.01
        assert abs(hop[4] - 0.10) < 0.01

    def test_high_suitable_fraction_no_hop_change(self):
        # suitable_fraction = 1.0 >= 0.5, content_length_p50 = 800 → no change
        scores = {f"c_{i}": 0.8 for i in range(10)}
        census = _make_census(content_length_p50=800)
        result = auto_tune(census, _make_profile(scores), _make_cfg())
        assert "hop_distribution" not in result

    def test_short_chunk_gate_triggers_independently(self):
        # suitable_fraction = 1.0 (high) but content_length_p50 < 600 → short-chunk gate
        scores = {f"c_{i}": 0.8 for i in range(10)}
        census = _make_census(content_length_p50=400)
        cfg = _make_cfg(hop_distribution={2: 0.5, 3: 0.3, 4: 0.2})
        result = auto_tune(census, _make_profile(scores), cfg)
        hop = result["hop_distribution"]
        assert abs(hop[3] - 0.15) < 0.01
        assert abs(hop[2] - 0.75) < 0.01

    def test_high_hop_weight_threshold_no_adjustment(self):
        # suitable_fraction = 0.4 but high_hop_weight = 0.2 (not > 0.2) → no adjustment
        scores = _mixed_scores(high=4, low=6)
        census = _make_census(content_length_p50=800)
        cfg = _make_cfg(hop_distribution={2: 0.8, 3: 0.1, 4: 0.1})
        result = auto_tune(census, _make_profile(scores), cfg)
        assert "hop_distribution" not in result

    def test_no_suitability_scores_skips_suitability_gate(self):
        # Empty scores → suitability gate skipped; content_length_p50 = 800 → no short-chunk gate
        census = _make_census(content_length_p50=800)
        result = auto_tune(census, _make_profile({}), _make_cfg())
        assert "hop_distribution" not in result

    def test_no_suitability_scores_short_chunk_still_triggers(self):
        # Empty scores, short chunks → short-chunk gate fires
        census = _make_census(content_length_p50=300)
        cfg = _make_cfg(hop_distribution={2: 0.5, 3: 0.3, 4: 0.2})
        result = auto_tune(census, _make_profile({}), cfg)
        hop = result["hop_distribution"]
        assert abs(hop[3] - 0.15) < 0.01


# ---------------------------------------------------------------------------
# 7c: Total samples — effective pool cap
# ---------------------------------------------------------------------------


class TestAutoTuneTotalSamplesEffectivePool:
    def test_caps_when_exceeds_2x_effective_pool(self):
        # 500 suitable of 1000 → effective_pool = 500; total_samples = 1500 > 1000 → cap at 1000
        scores = {f"c_{i}": (0.8 if i < 500 else 0.2) for i in range(1000)}
        census = _make_census(chunk_count=1000)
        result = auto_tune(census, _make_profile(scores), _make_cfg(total_samples=1500))
        assert result["total_samples"] == 1000  # effective_pool * 2

    def test_no_cap_when_within_pool(self):
        # 100 suitable of 200 → effective_pool = 100; total_samples = 100 <= 200 → no cap
        scores = {f"c_{i}": 0.8 for i in range(100)}
        census = _make_census(chunk_count=200)
        result = auto_tune(census, _make_profile(scores), _make_cfg(total_samples=100))
        assert "total_samples" not in result

    def test_zero_effective_pool_no_cap(self):
        # No suitable chunks → effective_pool = 0 → guard prevents capping to 0
        scores = {f"c_{i}": 0.1 for i in range(100)}
        census = _make_census(chunk_count=100)
        result = auto_tune(census, _make_profile(scores), _make_cfg(total_samples=50))
        assert "total_samples" not in result

    def test_empty_suitability_scores_no_cap(self):
        # Empty scores → suitable_fraction = 0 → effective_pool = 0 → no cap
        census = _make_census(chunk_count=100)
        result = auto_tune(census, _make_profile({}), _make_cfg(total_samples=50))
        assert "total_samples" not in result

    def test_partial_suitability(self):
        # 30 of 100 suitable → effective_pool = 30; total_samples = 100 > 60 → cap at 60
        scores = {f"c_{i}": (0.8 if i < 30 else 0.1) for i in range(100)}
        census = _make_census(chunk_count=100)
        result = auto_tune(census, _make_profile(scores), _make_cfg(total_samples=100))
        assert result["total_samples"] == 60


# ---------------------------------------------------------------------------
# 7d: Reasoning mode distribution — prevalence-proportional
# ---------------------------------------------------------------------------


class TestAutoTuneReasoningModes:
    def _high_suit_profile(self) -> CorpusProfile:
        return _high_profile()

    def test_temporal_removed_when_date_prevalence_very_low(self):
        census = _make_census(date_prevalence=0.02)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert "temporal" not in mode
        # temporal weight redistributed to factual, then renorm
        assert mode.get("factual", 0) > 0

    def test_temporal_capped_when_date_prevalence_moderate(self):
        # date_prevalence = 0.1 → cap temporal at 0.1 (was 0.3)
        census = _make_census(date_prevalence=0.1)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert "temporal" in mode
        assert mode["temporal"] < 0.3

    def test_sequential_removed_when_sequential_prevalence_very_low(self):
        census = _make_census(sequential_prevalence=0.01)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert "sequential" not in mode

    def test_sequential_capped_when_sequential_prevalence_moderate(self):
        # sequential_prevalence = 0.15 → cap sequential at 0.15 (was 0.2)
        census = _make_census(sequential_prevalence=0.15)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert "sequential" in mode
        assert mode["sequential"] < 0.2

    def test_both_removed_renormalization_sums_to_one(self):
        census = _make_census(date_prevalence=0.01, sequential_prevalence=0.01)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert abs(sum(mode.values()) - 1.0) < 0.001
        assert "temporal" not in mode
        assert "sequential" not in mode

    def test_no_change_when_prevalence_high(self):
        # Both prevalences >= 0.3 → no temporal/sequential adjustment
        census = _make_census(date_prevalence=0.5, sequential_prevalence=0.5)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        assert "reasoning_mode_distribution" not in result

    def test_missing_temporal_key_no_error(self):
        # No "temporal" in config → no adjustment for temporal
        census = _make_census(date_prevalence=0.02, sequential_prevalence=0.5)
        cfg = _make_cfg(reasoning_mode_distribution={"factual": 0.8, "sequential": 0.2})
        result = auto_tune(census, self._high_suit_profile(), cfg)
        # sequential_prevalence = 0.5 >= 0.3 → no sequential change; no temporal key → no change
        assert "reasoning_mode_distribution" not in result

    def test_capped_mode_renormalizes_correctly(self):
        # date_prevalence = 0.1: temporal 0.3 → capped at 0.1; excess 0.2 → factual
        # Before renorm: factual = 0.7, temporal = 0.1, sequential = 0.2; total = 1.0
        # After renorm: same (already sums to 1.0)
        census = _make_census(date_prevalence=0.1, sequential_prevalence=0.5)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        result = auto_tune(census, self._high_suit_profile(), cfg)
        mode = result["reasoning_mode_distribution"]
        assert abs(sum(mode.values()) - 1.0) < 0.001
        assert abs(mode["temporal"] - 0.1) < 0.01


# ---------------------------------------------------------------------------
# emit_corpus_warnings
# ---------------------------------------------------------------------------


class TestEmitCorpusWarnings:
    def test_zero_chunk_count_returns_early(self):
        census = _make_census(chunk_count=0)
        warnings = emit_corpus_warnings(census, _make_profile(), _make_cfg())
        assert len(warnings) == 1
        assert "0 chunks" in warnings[0]

    def test_low_date_prevalence_warns(self):
        census = _make_census(date_prevalence=0.02, chunk_count=100)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        warnings = emit_corpus_warnings(census, _high_profile(), cfg)
        assert any("date metadata" in w for w in warnings)
        assert any("temporal" in w for w in warnings)

    def test_no_date_warning_when_no_temporal_mode(self):
        # Low prevalence but no "temporal" mode in config → no warning
        census = _make_census(date_prevalence=0.02, chunk_count=100)
        cfg = _make_cfg(reasoning_mode_distribution={"factual": 1.0})
        warnings = emit_corpus_warnings(census, _high_profile(), cfg)
        assert not any("date metadata" in w for w in warnings)

    def test_no_date_warning_when_high_prevalence(self):
        census = _make_census(date_prevalence=0.8, chunk_count=100)
        cfg = _make_cfg(reasoning_mode_distribution=_DEFAULT_MODES)
        warnings = emit_corpus_warnings(census, _high_profile(), cfg)
        assert not any("date metadata" in w for w in warnings)

    def test_low_linkability_warns(self):
        # unstructured + no entity density → linkability = 0.3 → reduced multi-hop
        census = _make_census(
            metadata_regime="unstructured", entity_density_p25=0.0, chunk_count=100
        )
        warnings = emit_corpus_warnings(census, _high_profile(), _make_cfg(multi_hop=0.7))
        assert any("linkability" in w for w in warnings)

    def test_no_linkability_warning_when_structured(self):
        # structured → linkability = 1.0 → no change
        census = _make_census(metadata_regime="structured", chunk_count=100)
        cfg = _make_cfg(multi_hop=0.7)
        warnings = emit_corpus_warnings(census, _high_profile(), cfg)
        assert not any("linkability" in w for w in warnings)

    def test_effective_pool_warning(self):
        # 10 suitable of 100 → effective_pool = 10; total_samples = 300 > 20 → warning
        scores = {f"c_{i}": (0.8 if i < 10 else 0.1) for i in range(100)}
        census = _make_census(chunk_count=100)
        warnings = emit_corpus_warnings(census, _make_profile(scores), _make_cfg(total_samples=300))
        assert any("Effective pool" in w for w in warnings)
        assert any("capped at" in w for w in warnings)

    def test_no_effective_pool_warning_when_within_bounds(self):
        scores = {f"c_{i}": 0.8 for i in range(100)}
        census = _make_census(chunk_count=100)
        warnings = emit_corpus_warnings(census, _make_profile(scores), _make_cfg(total_samples=50))
        assert not any("Effective pool" in w for w in warnings)

    def test_single_file_corpus_warns(self):
        # doc_id_prevalence = 0.0 → multi_file_corpus = False
        census = _make_census(doc_id_prevalence=0.0, chunk_count=100)
        ws = emit_corpus_warnings(census, _high_profile(), _make_cfg())
        assert any("single source file" in w for w in ws)

    def test_multi_file_corpus_no_single_file_warning(self):
        # doc_id_prevalence = 0.9 → multi_file_corpus = True
        census = _make_census(doc_id_prevalence=0.9, chunk_count=100)
        ws = emit_corpus_warnings(census, _high_profile(), _make_cfg())
        assert not any("single source file" in w for w in ws)


# ---------------------------------------------------------------------------
# compute_batch_heuristics (existing logic, now tested)
# ---------------------------------------------------------------------------


class TestComputeBatchHeuristics:
    @pytest.mark.parametrize(
        "total, expected_batch, expected_parallel",
        [
            (1, 5, 5),
            (25, 5, 5),
            (26, 10, 5),
            (50, 10, 5),
            (51, 10, 10),
            (100, 10, 10),
            (101, 50, 10),
            (200, 50, 10),
            (500, 50, 10),
            (501, 50, 20),
            (1000, 50, 20),
            (1001, 50, 20),
            (2000, 50, 20),
        ],
    )
    def test_lookup_table(self, total: int, expected_batch: int, expected_parallel: int) -> None:
        batch, parallel = compute_batch_heuristics(total)
        assert batch == expected_batch
        assert parallel == expected_parallel
