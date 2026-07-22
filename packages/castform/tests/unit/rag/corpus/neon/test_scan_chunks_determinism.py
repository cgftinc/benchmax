"""Contract #5: scan_chunks stable ordering.

The order key is frozen here (pass); the iterator itself is an xfail skeleton
filled by Slice 1.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.source import SCAN_ORDER_BY, NeonChunkSource


def test_scan_order_key_frozen() -> None:
    assert SCAN_ORDER_BY == ("source_file", "chunk_index", "id")


@pytest.mark.xfail(reason="scan_chunks built in Slice 1", strict=False)
def test_scan_chunks_is_deterministic() -> None:
    source = NeonChunkSource("mycorpus")
    first = [c.hash for c in source.scan_chunks()]
    second = [c.hash for c in source.scan_chunks()]
    assert first == second
