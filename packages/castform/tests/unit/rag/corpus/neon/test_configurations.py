"""NeonChunkSource configuration variants.

Degraded/variant configs, mirroring the tpuf/chroma configuration suites:
- No embed_fn (lexical only) with vector/hybrid requested -> clean rejection.
- No file metadata -> graceful degradation (no neighbor skip, same_file False,
  context fallback).
- Auto mode preference and the search_text/search_content flows.
"""

from __future__ import annotations

import pytest
from fakes.neon import (
    FakeQueryRunner,
    FakeReadClient,
    constant_embed_fn,
    make_neon_source,
    make_query_row,
)

from castform.rag.chunkers.models import Chunk
from castform.rag.corpus.search_schema.search_exceptions import (
    UnsupportedSearchModeError,
)
from castform.rag.corpus.search_schema.search_types import SearchSpec


class TestNoEmbedFnModeValidation:
    def test_search_vector_spec_raises(self):
        source = make_neon_source(embed_fn=None, search=FakeQueryRunner())
        with pytest.raises(UnsupportedSearchModeError):
            source.search(SearchSpec(mode="vector", top_k=5, vector_query=[0.1, 0.2]))

    def test_search_related_vector_mode_raises(self):
        source = make_neon_source(embed_fn=None, search=FakeQueryRunner())
        with pytest.raises(UnsupportedSearchModeError):
            source.search_related(Chunk(content="seed"), ["q"], top_k=5, mode="vector")

    def test_search_related_hybrid_mode_raises(self):
        source = make_neon_source(embed_fn=None, search=FakeQueryRunner())
        with pytest.raises(UnsupportedSearchModeError):
            source.search_related(Chunk(content="seed"), ["q"], top_k=5, mode="hybrid")

    def test_search_related_lexical_still_works(self):
        runner = FakeQueryRunner(rows=[make_query_row("r1", "lexical result")])
        source = make_neon_source(embed_fn=None, search=runner)
        results = source.search_related(Chunk(content="seed"), ["q"], top_k=5)
        assert len(results) == 1
        assert results[0]["chunk"].content == "lexical result"


class TestNoFileMetadata:
    def test_no_neighbor_skip_without_file_metadata(self):
        rows = {
            "q": [
                make_query_row("a1", source_file="x.md", chunk_index=0),
                make_query_row("a2", source_file="x.md", chunk_index=1),
            ]
        }
        source = make_neon_source(search=FakeQueryRunner(rows_by_query=rows))
        results = source.search_related(Chunk(content="seed"), ["q"], top_k=5)
        assert len(results) == 2  # adjacency undefined without file/index -> no skip

    def test_same_file_always_false_without_file_metadata(self):
        rows = {"q": [make_query_row("r1", source_file="x.md", chunk_index=0)]}
        source = make_neon_source(search=FakeQueryRunner(rows_by_query=rows))
        results = source.search_related(Chunk(content="seed"), ["q"], top_k=5)
        assert results[0]["same_file"] is False

    def test_get_chunk_with_context_fallback(self):
        source = make_neon_source(read_client=FakeReadClient())
        ctx = source.get_chunk_with_context(Chunk(content="body"))
        assert ctx["chunk_content"]
        assert ctx["prev_chunk_preview"] == "(no previous chunk)"
        assert ctx["next_chunk_preview"] == "(no next chunk)"


class TestModeResolution:
    def test_auto_prefers_hybrid_with_embed(self):
        runner = FakeQueryRunner(rows=[make_query_row("r1", "r")])
        source = make_neon_source(embed_fn=constant_embed_fn(), search=runner)
        origin = Chunk(content="seed", metadata=(("file", "a.md"), ("index", 0)))
        source.search_related(origin, ["q"], top_k=5)
        assert runner.calls[0].mode == "hybrid"

    def test_auto_is_lexical_without_embed(self):
        runner = FakeQueryRunner(rows=[make_query_row("r1", "r")])
        source = make_neon_source(search=runner)
        source.search_related(Chunk(content="seed"), ["q"], top_k=5)
        assert runner.calls[0].mode == "lexical"


class TestSearchTextFlow:
    def test_search_text_delegates_to_lexical(self):
        runner = FakeQueryRunner(rows=[make_query_row("h1", "found it")])
        source = make_neon_source(search=runner)
        results = source.search_text("find me", top_k=5)
        assert len(results) == 1
        assert results[0].content == "found it"
        assert runner.calls[0].mode == "lexical"

    def test_search_content_returns_strings(self):
        runner = FakeQueryRunner(rows=[make_query_row("h1", "content text")])
        source = make_neon_source(embed_fn=constant_embed_fn(), search=runner)
        out = source.search_content(SearchSpec(mode="lexical", top_k=5, text_query="q"))
        assert out == ["content text"]
        assert all(isinstance(s, str) for s in out)
