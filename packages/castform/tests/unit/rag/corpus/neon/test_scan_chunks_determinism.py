"""Contract #5: scan_chunks total, pageable, deterministic ordering.

The typed order key is frozen here (pass); the iterator is an xfail skeleton
that must raise NotImplementedError until Slice 1. The skeleton forces a
multi-page scan (batch_size < corpus size) and asserts run-to-run determinism,
so Slice 1 cannot satisfy it with a single-page or empty read.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.source import SCAN_ORDER_BY, NeonChunkSource


def test_scan_order_key_frozen() -> None:
    # Typed non-null columns, not JSONB extraction.
    assert SCAN_ORDER_BY == ("source_file", "chunk_index", "id")


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 1")
def test_scan_chunks_deterministic_across_pages() -> None:
    source = NeonChunkSource.__new__(NeonChunkSource)
    first = [c.hash for c in source.scan_chunks(batch_size=2)]
    second = [c.hash for c in source.scan_chunks(batch_size=2)]
    assert first == second
    assert len(first) > 2  # more than one page materialized
