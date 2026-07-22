"""Contract #4: multi-query search_related dedup + full 3-tuple ordering.

Belongs to the multi-query ``NeonChunkSource.search_related`` surface. Skeleton:
must raise NotImplementedError until Slice 1. Controlled hits are injected (chunk
C hit by BOTH queries at ranks 0 and 2, chunk D by one query) so the filled-in
test asserts the full ``(query_count, cross_file, max_score)`` tuple order and the
max-reciprocal-rank dedup unconditionally — no vacuous ``if``.
"""

from __future__ import annotations

import pytest

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.neon.search import QueryHit
from castform.rag.corpus.neon.source import NeonChunkSource


def _chunk(content: str, source_file: str, chunk_index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata=(("source_file", source_file), ("chunk_index", chunk_index)),
    )


class _FakeSearch:
    """Returns canned per-query hits keyed by query text."""

    def __init__(self, by_query: dict[str, list[QueryHit]]) -> None:
        self.by_query = by_query

    def query_hits(self, query: str) -> list[QueryHit]:
        return self.by_query.get(query, [])


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_search_related_dedup_and_full_tuple_order() -> None:
    origin = _chunk("origin", "a.md", 0)
    c = _chunk("cross-file hit", "b.md", 4)  # hit by q1 (rank 0) and q2 (rank 2)
    d = _chunk("same-file hit", "a.md", 9)  # hit by q1 only (rank 1)

    source = NeonChunkSource("mycorpus")
    source._search = _FakeSearch(  # type: ignore[attr-defined]
        {
            "q1": [
                QueryHit(c.hash, 1 / 60, -2.0, 0),
                QueryHit(d.hash, 1 / 61, -2.5, 1),
            ],
            "q2": [QueryHit(c.hash, 1 / 62, -3.0, 2)],
        }
    )

    results = source.search_related(origin, ["q1", "q2"], top_k=5)

    for r in results:
        assert set(r) >= {"chunk", "queries", "same_file", "max_score", "native_score"}

    keys = [(len(r["queries"]), not r["same_file"], r["max_score"]) for r in results]
    assert keys == sorted(keys, reverse=True)

    # C: hit by both queries, dedups to max reciprocal rank 1/60, ranks first.
    top = results[0]
    assert top["chunk"].hash == c.hash
    assert top["max_score"] == 1 / 60
    assert len(top["queries"]) == 2
    # native_score comes from the winning hit (rank 0 in q1).
    assert top["native_score"] == -2.0
