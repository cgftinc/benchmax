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


def test_default_thresholds_ordered_by_mode() -> None:
    assert DEFAULT_THRESHOLDS["hybrid"].hit_at_k > DEFAULT_THRESHOLDS["vector"].hit_at_k
    assert (
        DEFAULT_THRESHOLDS["vector"].hit_at_k > DEFAULT_THRESHOLDS["lexical"].hit_at_k
    )
    assert LEXICAL_ABLATION_MIN_DELTA == 0.05
