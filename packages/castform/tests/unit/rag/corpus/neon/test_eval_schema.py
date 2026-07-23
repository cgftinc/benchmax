"""Contract #6: eval JSONL schema (frozen; validates today)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from castform.rag.corpus.neon.eval_schema import (
    DEFAULT_THRESHOLDS,
    LEXICAL_ABLATION_MIN_DELTA,
    NeonEvalRecord,
)


def test_record_carries_gold_hashes_explicitly() -> None:
    rec = NeonEvalRecord(
        capability="filter_eq",
        search_mode="hybrid",
        query="who wrote it",
        filter_dsl={"field": "year", "op": "eq", "value": 2026},
        gold_chunk_hashes=["abc123"],
        decoy_chunk_hashes=["def456"],
    )
    assert rec.gold_chunk_hashes == ["abc123"]
    assert rec.decoy_chunk_hashes == ["def456"]


def test_record_requires_at_least_one_gold() -> None:
    with pytest.raises(ValidationError):
        NeonEvalRecord(
            capability="c",
            search_mode="lexical",
            query="q",
            gold_chunk_hashes=[],
        )


def test_filter_dsl_optional() -> None:
    rec = NeonEvalRecord(
        capability="c",
        search_mode="vector",
        query="q",
        gold_chunk_hashes=["h"],
    )
    assert rec.filter_dsl is None


def test_default_thresholds_encode_gate_contract() -> None:
    """Threshold contract: lexical + vector are real gates (vector beats lexical since
    it must beat keyword retrieval); hybrid is a SMOKE floor, deliberately NOT tied to
    the ordering (RRF is real + unit-tested in Slice 4, but a fusion-necessity gate is
    unbuildable on this lexical- and vector-strong corpus; deferred to Path X). The old
    "hybrid is the strongest gate" ordering was overturned.
    """
    # every threshold is a usable fraction
    for mode, th in DEFAULT_THRESHOLDS.items():
        assert 0.0 < th.hit_at_k <= 1.0, (mode, th.hit_at_k)
        assert 0.0 < th.mrr_at_k <= 1.0, (mode, th.mrr_at_k)
        assert th.k == 5, (mode, th.k)
    # vector is the harder real gate; it must sit above lexical
    assert (
        DEFAULT_THRESHOLDS["vector"].hit_at_k > DEFAULT_THRESHOLDS["lexical"].hit_at_k
    )
    # hybrid is a smoke floor, NOT required to exceed the real gates
    assert DEFAULT_THRESHOLDS["hybrid"].hit_at_k <= DEFAULT_THRESHOLDS["vector"].hit_at_k
    assert LEXICAL_ABLATION_MIN_DELTA == 0.05
