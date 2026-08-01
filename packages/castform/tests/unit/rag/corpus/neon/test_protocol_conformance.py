"""NeonChunkSource conforms to the ChunkSource protocol (structural check)."""

from __future__ import annotations

from castform.rag.corpus.neon.source import NeonChunkSource
from castform.rag.corpus.source import ChunkSource


def test_neon_source_is_chunk_source() -> None:
    source = NeonChunkSource.__new__(NeonChunkSource)
    assert isinstance(source, ChunkSource)
