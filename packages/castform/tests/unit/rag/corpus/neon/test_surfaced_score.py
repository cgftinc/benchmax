"""Contract #4: public surfaced-score formula + dedup/order.

The formula itself is frozen in Slice A (these assertions pass); the dedup and
3-tuple ordering are xfail skeletons filled by Slice 1.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.search import (
    SURFACED_RANK_K,
    NeonSearch,
    surfaced_score,
)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, 1 / 60),
        (1, 1 / 61),
        (2, 1 / 62),
    ],
)
def test_surfaced_score_exact_values(rank: int, expected: float) -> None:
    assert surfaced_score(rank) == expected


def test_surfaced_score_constant() -> None:
    assert SURFACED_RANK_K == 60
    assert surfaced_score(0) == 1 / 60


def test_surfaced_score_strictly_decreasing() -> None:
    scores = [surfaced_score(r) for r in range(10)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_surfaced_score_rejects_negative_rank() -> None:
    with pytest.raises(ValueError):
        surfaced_score(-1)


@pytest.mark.xfail(reason="multi-query dedup built in Slice 1", strict=False)
def test_multi_query_dedup_keeps_max_rank() -> None:
    # chunk C: rank 0 in query a, rank 2 in query b -> max_score = 1/60.
    search = NeonSearch("corpus")
    results = search.query(...)  # type: ignore[arg-type]
    by_id = {r[0]: r[1] for r in results}
    assert by_id["C"] == pytest.approx(1 / 60)


@pytest.mark.xfail(reason="3-tuple ordering built in Slice 1", strict=False)
def test_results_sorted_by_query_count_then_score() -> None:
    # (len(queries), not same_file, max_score) all descending.
    search = NeonSearch("corpus")
    results = search.query(...)  # type: ignore[arg-type]
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
