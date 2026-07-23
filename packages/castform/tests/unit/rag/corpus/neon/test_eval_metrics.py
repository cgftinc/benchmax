"""Unit tests for the pure retrieval-quality metrics (no DB)."""

from __future__ import annotations

from castform.rag.corpus.neon.eval_metrics import (
    AblationResult,
    aggregate_by_mode,
    evaluate_record,
    has_decoy,
    is_hit,
    lexical_ablation,
    reciprocal_rank,
)
from castform.rag.corpus.neon.eval_schema import (
    LEXICAL_ABLATION_MIN_DELTA,
    NeonEvalRecord,
)


def test_reciprocal_rank_first_gold_wins() -> None:
    assert reciprocal_rank({"b"}, ["a", "b", "c"], k=5) == 0.5
    assert reciprocal_rank({"a"}, ["a", "b"], k=5) == 1.0


def test_reciprocal_rank_zero_when_outside_k() -> None:
    assert reciprocal_rank({"d"}, ["a", "b", "c", "d"], k=3) == 0.0
    assert reciprocal_rank({"z"}, ["a", "b"], k=5) == 0.0


def test_is_hit_and_decoy_respect_k() -> None:
    ranked = ["a", "b", "c", "d"]
    assert is_hit({"c"}, ranked, k=3) is True
    assert is_hit({"d"}, ranked, k=3) is False
    assert has_decoy({"b"}, ranked, k=3) is True
    assert has_decoy({"d"}, ranked, k=3) is False


def _rec(mode: str, gold: list[str], decoys: list[str] | None = None) -> NeonEvalRecord:
    return NeonEvalRecord(
        capability=f"{mode}_x",
        search_mode=mode,  # type: ignore[arg-type]
        query="q",
        gold_chunk_hashes=gold,
        decoy_chunk_hashes=decoys or [],
    )


def test_evaluate_record_captures_hit_rr_and_decoy() -> None:
    rec = _rec("vector", ["g"], ["bad"])
    ev = evaluate_record(rec, ["x", "g", "bad"], k=5)
    assert ev.hit is True
    assert ev.reciprocal_rank == 0.5
    assert ev.decoy_violation is True
    assert ev.search_mode == "vector"


def test_aggregate_by_mode_means_and_threshold_gate() -> None:
    recs = [_rec("lexical", ["g1"]), _rec("lexical", ["g2"])]
    # first hits at rank 1, second misses entirely
    evals = [
        evaluate_record(recs[0], ["g1"], k=5),
        evaluate_record(recs[1], ["nope"], k=5),
    ]
    by_mode = aggregate_by_mode(evals, k=5)
    m = by_mode["lexical"]
    assert m.n == 2
    assert m.hit_at_k == 0.5
    assert m.mrr_at_k == 0.5
    assert m.decoy_violations == 0
    # 0.5 hit@k is below the lexical 0.80 bar
    from castform.rag.corpus.neon.eval_metrics import thresholds_for

    assert m.passes(thresholds_for("lexical")) is False


def test_lexical_ablation_delta_and_pass() -> None:
    recs = [_rec("vector", ["g1"]), _rec("vector", ["g2"])]
    # vector retrieval finds both; lexical finds neither -> delta 1.0
    primary = [
        evaluate_record(recs[0], ["g1"], k=5),
        evaluate_record(recs[1], ["g2"], k=5),
    ]
    lexical = [
        evaluate_record(recs[0], ["junk"], k=5),
        evaluate_record(recs[1], ["junk"], k=5),
    ]
    res = lexical_ablation(primary, lexical)
    assert isinstance(res, AblationResult)
    assert res.primary_hit_at_k == 1.0
    assert res.lexical_hit_at_k == 0.0
    assert res.delta == 1.0
    assert res.passes is True


def test_lexical_ablation_fails_when_keyword_solvable() -> None:
    recs = [_rec("vector", ["g1"])]
    primary = [evaluate_record(recs[0], ["g1"], k=5)]
    lexical = [evaluate_record(recs[0], ["g1"], k=5)]  # BM25 also finds it
    res = lexical_ablation(primary, lexical)
    assert res.delta == 0.0
    assert res.delta < LEXICAL_ABLATION_MIN_DELTA
    assert res.passes is False
