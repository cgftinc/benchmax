"""Contract #5: scan_chunks total, pageable, deterministic ordering.

The typed order key is frozen here (pass). The iterator is an xfail skeleton that
must raise NotImplementedError until Slice 1. It is backed by a fake reader with
THREE seeded rows and ``batch_size=2`` so the filled-in test genuinely exercises
multi-page ordering — not an empty/single-page read.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.source import SCAN_ORDER_BY, NeonChunkSource


class _FakeReader:
    """Minimal paging backend: rows pre-sorted by (source_file, chunk_index, id)."""

    def __init__(self, rows: list[tuple[str, int, str]]) -> None:
        self.rows = sorted(rows, key=lambda r: (r[0], r[1], r[2]))

    def page(self, after: tuple[str, int, str] | None, limit: int) -> list[tuple]:
        start = 0
        if after is not None:
            start = next(
                (i for i, r in enumerate(self.rows) if r > after), len(self.rows)
            )
        return self.rows[start : start + limit]


def test_scan_order_key_frozen() -> None:
    # Typed non-null columns, not JSONB extraction.
    assert SCAN_ORDER_BY == ("source_file", "chunk_index", "id")


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_scan_chunks_deterministic_across_pages() -> None:
    source = NeonChunkSource("mycorpus")
    # Seed three rows spanning >1 page at batch_size=2, out of natural order.
    source._reader = _FakeReader(  # type: ignore[attr-defined]
        [("b.md", 2, "h3"), ("a.md", 10, "h2"), ("a.md", 2, "h1")]
    )
    first = [c.hash for c in source.scan_chunks(batch_size=2)]
    second = [c.hash for c in source.scan_chunks(batch_size=2)]
    assert first == second
    assert first == ["h1", "h2", "h3"]  # (a.md,2) < (a.md,10) < (b.md,2)
