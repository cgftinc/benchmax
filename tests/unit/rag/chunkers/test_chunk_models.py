"""Tests for Chunk and ChunkCollection data models."""

from __future__ import annotations

import pickle

from benchmax.rag.chunkers.models import Chunk, ChunkCollection

# ---------------------------------------------------------------------------
# Test data: 2 files × 3 chunks each, plus a deeper-nested file
# ---------------------------------------------------------------------------


def _make_chunk(file: str, index: int, content: str) -> Chunk:
    return Chunk(
        content=content,
        metadata=(("file", file), ("index", index), ("h1", f"Section {index}")),
    )


def _make_chunks() -> list[Chunk]:
    return [
        _make_chunk("doc.md", 0, "First chunk of doc about installation."),
        _make_chunk("doc.md", 1, "Second chunk of doc about configuration."),
        _make_chunk("doc.md", 2, "Third chunk of doc about usage."),
        _make_chunk("ref.md", 0, "First chunk of ref about API endpoints."),
        _make_chunk("ref.md", 1, "Second chunk of ref about auth."),
        _make_chunk("ref.md", 2, "Third chunk of ref about errors."),
        _make_chunk("subdir/deep.md", 0, "Deep file first chunk."),
    ]


def _make_collection() -> ChunkCollection:
    return ChunkCollection(chunks=_make_chunks())


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_auto_hash(self):
        c = Chunk(content="hello", metadata=(("k", "v"),))
        assert c.hash
        assert len(c.hash) == 64  # SHA256 hex

    def test_deterministic_hash(self):
        a = Chunk(content="same", metadata=(("k", "v"),))
        b = Chunk(content="same", metadata=(("k", "v"),))
        assert a.hash == b.hash

    def test_different_content_different_hash(self):
        a = Chunk(content="aaa", metadata=())
        b = Chunk(content="bbb", metadata=())
        assert a.hash != b.hash

    def test_get_metadata_present(self):
        c = _make_chunk("doc.md", 0, "text")
        assert c.get_metadata("file") == "doc.md"
        assert c.get_metadata("index") == 0

    def test_get_metadata_missing(self):
        c = Chunk(content="text", metadata=())
        assert c.get_metadata("nope") is None

    def test_get_metadata_default(self):
        c = Chunk(content="text", metadata=())
        assert c.get_metadata("nope", "fallback") == "fallback"

    def test_metadata_dict(self):
        c = _make_chunk("doc.md", 0, "text")
        d = c.metadata_dict
        assert isinstance(d, dict)
        assert d["file"] == "doc.md"

    def test_len(self):
        c = Chunk(content="hello", metadata=())
        assert len(c) == 5

    def test_chunk_str_includes_content(self):
        c = Chunk(content="hello world", metadata=(("k", "v"),))
        s = c.chunk_str()
        assert "hello world" in s

    def test_chunk_str_truncates(self):
        c = Chunk(content="a" * 200, metadata=())
        truncated = c.chunk_str(max_chars=50)
        full = c.chunk_str()
        assert len(truncated) < len(full)

    def test_pickle_roundtrip(self):
        c = _make_chunk("doc.md", 0, "pickle me")
        restored = pickle.loads(pickle.dumps(c))
        assert restored.content == c.content
        assert restored.metadata == c.metadata
        assert restored.hash == c.hash


# ---------------------------------------------------------------------------
# ChunkCollection
# ---------------------------------------------------------------------------


class TestChunkCollection:
    def test_len(self):
        coll = _make_collection()
        assert len(coll) == 7

    def test_files(self):
        coll = _make_collection()
        assert set(coll.files) == {"doc.md", "ref.md", "subdir/deep.md"}

    def test_get_file_chunks(self):
        coll = _make_collection()
        doc_chunks = coll.get_file_chunks("doc.md")
        assert len(doc_chunks) == 3
        assert all(c.get_metadata("file") == "doc.md" for c in doc_chunks)

    def test_get_file_chunks_preserves_order(self):
        coll = _make_collection()
        doc_chunks = coll.get_file_chunks("doc.md")
        indices = [c.get_metadata("index") for c in doc_chunks]
        assert indices == [0, 1, 2]

    def test_neighboring_middle(self):
        coll = _make_collection()
        middle = coll.get_file_chunks("doc.md")[1]  # index=1
        before, after = coll.get_neighboring_chunks(middle)
        assert len(before) == 1
        assert len(after) == 1
        assert before[0].get_metadata("index") == 0
        assert after[0].get_metadata("index") == 2

    def test_neighboring_first(self):
        coll = _make_collection()
        first = coll.get_file_chunks("doc.md")[0]
        before, after = coll.get_neighboring_chunks(first)
        assert before == []
        assert len(after) == 1

    def test_neighboring_unknown_chunk(self):
        coll = _make_collection()
        foreign = Chunk(content="unknown", metadata=())
        before, after = coll.get_neighboring_chunks(foreign)
        assert before == []
        assert after == []

    def test_get_chunk_by_hash_found(self):
        coll = _make_collection()
        target = coll.chunks[0]
        found = coll.get_chunk_by_hash(target.hash)
        assert found is not None
        assert found.content == target.content

    def test_get_chunk_by_hash_not_found(self):
        coll = _make_collection()
        assert coll.get_chunk_by_hash("0" * 64) is None

    def test_top_level_chunks(self):
        coll = _make_collection()
        top = coll.get_top_level_chunks()
        # doc.md and ref.md are at depth 0, subdir/deep.md at depth 1
        files = {c.get_metadata("file") for c in top}
        assert "doc.md" in files
        assert "ref.md" in files
        assert "subdir/deep.md" not in files

    def test_top_level_empty(self):
        coll = ChunkCollection(chunks=[])
        assert coll.get_top_level_chunks() == []

    def test_get_chunk_with_context_keys(self):
        coll = _make_collection()
        chunk = coll.get_file_chunks("doc.md")[1]
        ctx = coll.get_chunk_with_context(chunk)
        assert "chunk_content" in ctx
        assert "prev_chunk_preview" in ctx
        assert "next_chunk_preview" in ctx

    def test_get_chunk_with_context_first(self):
        coll = _make_collection()
        first = coll.get_file_chunks("doc.md")[0]
        ctx = coll.get_chunk_with_context(first)
        assert ctx["prev_chunk_preview"] == "(No previous chunk)"
        assert ctx["next_chunk_preview"] != "(No next chunk)"

    def test_get_chunk_with_context_last(self):
        coll = _make_collection()
        last = coll.get_file_chunks("doc.md")[2]
        ctx = coll.get_chunk_with_context(last)
        assert ctx["prev_chunk_preview"] != "(No previous chunk)"
        assert ctx["next_chunk_preview"] == "(No next chunk)"
