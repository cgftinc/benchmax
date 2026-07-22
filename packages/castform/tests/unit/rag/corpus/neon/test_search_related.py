"""Contract #4: multi-query search_related dedup + full 3-tuple ordering.

These belong to the multi-query ``NeonChunkSource.search_related`` surface (not
single-query ``NeonSearch.query``). Skeleton: must raise NotImplementedError
until Slice 1 fills it. The scenario + assertions encode the frozen contract so
Slice 1 has an exact target: dedup keeps the max reciprocal rank, and results
sort by ``(query_count, cross_file, max_score)`` all descending.
"""

from __future__ import annotations

import pytest

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.neon.source import NeonChunkSource


def _chunk(content: str, source_file: str, chunk_index: int) -> Chunk:
    return Chunk(
        content=content,
        metadata=(("source_file", source_file), ("chunk_index", chunk_index)),
    )


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_search_related_dedup_and_full_tuple_order() -> None:
    source = NeonChunkSource.__new__(NeonChunkSource)
    origin = _chunk("origin", "a.md", 0)

    results = source.search_related(origin, ["q1", "q2"], top_k=5)

    # Every result exposes both scores (NB1) and the dedup/ordering keys.
    for r in results:
        assert set(r) >= {"chunk", "queries", "same_file", "max_score", "native_score"}

    # Full 3-tuple sort: (len(queries) desc, cross-file-first, max_score desc).
    keys = [(len(r["queries"]), not r["same_file"], r["max_score"]) for r in results]
    assert keys == sorted(keys, reverse=True)

    # A chunk hit by both queries dedups to the MAX reciprocal rank.
    multi = [r for r in results if len(r["queries"]) > 1]
    if multi:
        assert multi[0]["max_score"] == max(r["max_score"] for r in results)
