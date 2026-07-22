"""Contract #4 / NB1: the surfaced-score formula and the native-score split.

The formula and QueryHit shape are frozen in Slice A (these pass). Multi-query
dedup + ordering live in test_search_related.py (they belong to search_related).
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.search import SURFACED_RANK_K, QueryHit, surfaced_score


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


def test_surfaced_score_strictly_decreasing() -> None:
    scores = [surfaced_score(r) for r in range(10)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_surfaced_score_rejects_negative_rank() -> None:
    with pytest.raises(ValueError):
        surfaced_score(-1)


def test_query_hit_keeps_native_score_separate() -> None:
    # NB1: raw native score is a distinct field, not overloaded onto the ordinal.
    hit = QueryHit(chunk_id="C", surfaced_score=1 / 60, native_score=-3.14, rank=0)
    assert hit.surfaced_score == surfaced_score(0)
    assert hit.native_score == -3.14
