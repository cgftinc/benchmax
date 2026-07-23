"""NeonChunkSource — reads, search surface, and versioned-replace ingest.

Fake-backed unit tests (no live Neon): the read/search collaborators are injected,
and ``FakeWriteClient`` simulates the versioned ledger so the atomic-swap /
no-stale-rows / rollback contract (F10) and the deterministic re-chunk (B9) are
exercised without a database.
"""

from __future__ import annotations

import pytest
from fakes.neon import (
    FakeQueryRunner,
    FakeReadClient,
    FakeWriteClient,
    constant_embed_fn,
    make_neon_source,
    make_query_row,
    make_read_row,
)

from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.provision import CORPUS_SCHEMA, RO_ROLE
from castform.rag.corpus.neon.schema import view_name
from castform.rag.corpus.neon.source import NeonChunkSource, NeonIngestError
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidSearchSpecError,
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import SearchSpec


def _chunk(content: str, file: str = "a.md", index: int = 0) -> Chunk:
    return Chunk(content=content, metadata=(("file", file), ("index", index)))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_get_chunk_count(self):
        source = make_neon_source(read_client=FakeReadClient(count=42))
        assert source.get_chunk_count() == 42

    def test_sample_chunks_maps_rows_and_preserves_hash(self):
        read = FakeReadClient(
            sample_rows=[make_read_row("h1", "c1", {"file": "a.md", "index": 0})]
        )
        source = make_neon_source(read_client=read)
        chunks = source.sample_chunks(5, min_chars=0)
        assert [c.hash for c in chunks] == ["h1"]  # hash=id, not recomputed
        assert chunks[0].content == "c1"
        assert chunks[0].get_metadata("file") == "a.md"

    def test_scan_chunks_maps_rows(self):
        read = FakeReadClient(
            scan_rows=[make_read_row("h1", "c1"), make_read_row("h2", "c2")]
        )
        source = make_neon_source(read_client=read)
        assert [c.hash for c in source.scan_chunks(batch_size=2)] == ["h1", "h2"]

    def test_get_top_level_chunks(self):
        read = FakeReadClient(
            top_level_rows=[make_read_row("h1", "c1"), make_read_row("h2", "c2")]
        )
        source = make_neon_source(read_client=read)
        assert [c.hash for c in source.get_top_level_chunks()] == ["h1", "h2"]

    def test_get_chunk_with_context_with_neighbors(self):
        chunk = _chunk("mid", index=1)
        read = FakeReadClient(neighbor_rows=[(0, "previous text"), (2, "next text")])
        source = make_neon_source(read_client=read)
        ctx = source.get_chunk_with_context(chunk, max_chars=200)
        assert ctx["chunk_content"] == chunk.chunk_str()
        assert "previous text" in ctx["prev_chunk_preview"]
        assert "next text" in ctx["next_chunk_preview"]

    def test_get_chunk_with_context_truncates(self):
        chunk = _chunk("mid", index=1)
        read = FakeReadClient(neighbor_rows=[(2, "x" * 500)])
        source = make_neon_source(read_client=read)
        ctx = source.get_chunk_with_context(chunk, max_chars=50)
        assert len(ctx["next_chunk_preview"]) == 50
        assert ctx["next_chunk_preview"].endswith("...")
        assert ctx["prev_chunk_preview"] == "(no previous chunk)"

    def test_get_chunk_with_context_no_file_metadata(self):
        source = make_neon_source(read_client=FakeReadClient())
        ctx = source.get_chunk_with_context(Chunk(content="x", metadata=()))
        assert ctx["prev_chunk_preview"] == "(no previous chunk)"
        assert ctx["next_chunk_preview"] == "(no next chunk)"


# ---------------------------------------------------------------------------
# Search surface
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_returns_chunks(self):
        runner = FakeQueryRunner(
            rows=[make_query_row("h1", "alpha", metadata={"file": "a.md"})]
        )
        source = make_neon_source(embed_fn=constant_embed_fn(), search=runner)
        chunks = source.search(SearchSpec(mode="lexical", top_k=5, text_query="q"))
        assert [c.hash for c in chunks] == ["h1"]
        assert chunks[0].content == "alpha"

    def test_search_content_returns_strings(self):
        runner = FakeQueryRunner(rows=[make_query_row("h1", "alpha")])
        source = make_neon_source(embed_fn=constant_embed_fn(), search=runner)
        out = source.search_content(SearchSpec(mode="lexical", top_k=5, text_query="q"))
        assert out == ["alpha"]
        assert all(isinstance(s, str) for s in out)

    def test_search_text_is_lexical(self):
        runner = FakeQueryRunner(rows=[make_query_row("h1", "found")])
        source = make_neon_source(search=runner)  # no embed_fn -> lexical only
        chunks = source.search_text("find me", top_k=3)
        assert chunks[0].content == "found"
        assert runner.calls[0].mode == "lexical"
        assert runner.calls[0].text == "find me"

    def test_search_vector_mode_without_embed_raises(self):
        source = make_neon_source(embed_fn=None, search=FakeQueryRunner())
        with pytest.raises(UnsupportedSearchModeError):
            source.search(SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2]))

    def test_search_invalid_shape_raises(self):
        source = make_neon_source(embed_fn=None, search=FakeQueryRunner())
        with pytest.raises(InvalidSearchSpecError):
            source.search(SearchSpec(mode="lexical", top_k=5, text_query=""))

    def test_embed_query(self):
        source = make_neon_source(embed_fn=constant_embed_fn(dim=4))
        assert source.embed_query("hi") == [0.1, 0.1, 0.1, 0.1]

    def test_embed_query_none_without_embed_fn(self):
        assert make_neon_source(embed_fn=None).embed_query("hi") is None


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_lexical_only_without_embed(self):
        caps = make_neon_source(embed_fn=None).get_search_capabilities()
        assert caps["backend"] == "neon"
        assert caps["modes"] == {"lexical"}
        assert "cosine" not in caps["ranking"]

    def test_vector_hybrid_with_embed(self):
        caps = make_neon_source(embed_fn=constant_embed_fn()).get_search_capabilities()
        assert caps["modes"] == {"lexical", "vector", "hybrid"}
        assert {"cosine", "rrf"} <= caps["ranking"]

    def test_filter_ops_full_set(self):
        caps = make_neon_source().get_search_capabilities()
        assert {"ne", "gt", "lt", "contains_all"} <= caps["filter_ops"]["field"]
        assert caps["filter_ops"]["logical"] == {"and", "or", "not"}


# ---------------------------------------------------------------------------
# Ingest — next version, embed requirement, versioned replace
# ---------------------------------------------------------------------------


class TestIngestVersioning:
    def test_next_version_fresh_db_is_one(self):
        source = make_neon_source(write_client=FakeWriteClient())
        assert source._next_version() == 1

    def test_next_version_increments_from_ledger(self):
        fake = FakeWriteClient()
        fake.state = {1: "activated", 2: "ready"}
        fake.current = 1
        source = make_neon_source(write_client=fake)
        assert source._next_version() == 3

    def test_populate_requires_embed_fn(self):
        source = make_neon_source(embed_fn=None, write_client=FakeWriteClient())
        with pytest.raises(ValueError, match="embed_fn"):
            source.populate_from_chunks(ChunkCollection([_chunk("a")]))

    def test_short_embedding_batch_raises_before_publishing(self):
        """An embed_fn that returns fewer vectors than inputs must raise (data loss
        guard) — NOT silently truncate and publish a version missing chunks."""
        fake = FakeWriteClient()
        # Returns exactly one vector no matter how many chunks were passed.
        short_embed = lambda texts: [[0.1, 0.1, 0.1]]  # noqa: E731
        source = make_neon_source(embed_fn=short_embed, write_client=fake)
        collection = ChunkCollection([_chunk("a", index=0), _chunk("b", index=1)])
        with pytest.raises(ValueError, match="exactly one embedding"):
            source.populate_from_chunks(collection, show_summary=False)
        assert fake.versions == {}  # nothing built or published

    def test_versioned_replace_atomic_swap_no_stale_rows_and_rollback(self):
        """Proves the F10 contract end to end: each ingest builds a NEW version and
        atomically swaps the current pointer; the prior version's rows are retained
        (no stale-row deletion); rollback re-points to the prior version."""
        fake = FakeWriteClient()
        source = make_neon_source(embed_fn=constant_embed_fn(), write_client=fake)

        source.populate_from_chunks(
            ChunkCollection([_chunk("alpha", index=0), _chunk("beta", index=1)]),
            show_summary=False,
        )
        assert fake.current == 1
        assert fake.activations == [1]
        # activation carries the RO grant on the stable view identifier.
        assert fake.last_grant.view == view_name("mycorpus")
        assert fake.last_grant.schema == CORPUS_SCHEMA
        assert fake.last_grant.ro_role == RO_ROLE

        source.populate_from_chunks(
            ChunkCollection([_chunk("alpha v2", index=0)]), show_summary=False
        )
        # Atomic swap: the pointer moved to v2...
        assert fake.current == 2
        assert fake.activations == [1, 2]
        # ...no stale rows: BOTH version row-stores are retained (versioned replace,
        # not in-place upsert), and v1 stays 'activated' so rollback has a target.
        assert set(fake.versions) == {1, 2}
        assert len(fake.versions[1]) == 2 and len(fake.versions[2]) == 1
        assert fake.state[1] == "activated"

        source.rollback_version(1)
        assert fake.current == 1  # re-pointed, non-destructive
        assert fake.rollbacks == [1]
        assert set(fake.versions) == {1, 2}  # both physical stores still present

    def test_rows_carry_typed_columns_from_metadata(self):
        fake = FakeWriteClient()
        source = make_neon_source(embed_fn=constant_embed_fn(dim=3), write_client=fake)
        source.populate_from_chunks(
            ChunkCollection([_chunk("alpha", file="docs/a.md", index=7)]),
            show_summary=False,
        )
        (row,) = fake.versions[1]
        # INSERT_COLUMNS order: id, content, metadata, embedding, source_file, index.
        assert row[0]  # id == chunk hash
        assert row[1] == "alpha"
        assert row[3] == [0.1, 0.1, 0.1]  # embedding list
        assert row[4] == "docs/a.md"  # source_file from "file" metadata
        assert row[5] == 7  # chunk_index from "index" metadata


# ---------------------------------------------------------------------------
# populate_from_folder — deterministic re-chunk (B9)
# ---------------------------------------------------------------------------


class TestPopulateFromFolder:
    def _seed(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\nalpha section body\n\n## Sub\n\nmore body")
        (tmp_path / "b.md").write_text("# B\n\nbeta section body")

    def test_deterministic_chunk_count(self, tmp_path):
        """Re-chunking the same folder twice yields identical chunks in identical
        order, and every chunk is ingested (no silent drop)."""
        self._seed(tmp_path)
        src1 = make_neon_source(
            embed_fn=constant_embed_fn(), write_client=FakeWriteClient()
        )
        src1.populate_from_folder(str(tmp_path), min_chars=1, show_summary=False)
        src2 = make_neon_source(
            embed_fn=constant_embed_fn(), write_client=FakeWriteClient()
        )
        src2.populate_from_folder(str(tmp_path), min_chars=1, show_summary=False)

        hashes1 = [c.hash for c in src1.collection]
        hashes2 = [c.hash for c in src2.collection]
        assert hashes1 == hashes2
        assert len(hashes1) >= 2
        # built row count equals chunk count — no file silently dropped.
        assert len(src1._write_client.versions[1]) == len(hashes1)

    def test_surfaces_per_file_errors_instead_of_dropping(self, tmp_path):
        """A file that cannot be decoded surfaces as a NeonIngestError naming it,
        and NOTHING is ingested — unlike the base chunker's silent print-and-drop."""
        (tmp_path / "good.md").write_text("# Good\n\ncontent body")
        (tmp_path / "bad.md").write_bytes(b"\xff\xfe not valid utf-8")
        fake = FakeWriteClient()
        source = make_neon_source(embed_fn=constant_embed_fn(), write_client=fake)
        with pytest.raises(NeonIngestError, match="bad.md"):
            source.populate_from_folder(str(tmp_path), min_chars=1, show_summary=False)
        assert fake.versions == {}  # aborted before any DB write


def test_neon_source_structurally_implements_chunk_source():
    from castform.rag.corpus.source import ChunkSource

    assert isinstance(NeonChunkSource.__new__(NeonChunkSource), ChunkSource)
