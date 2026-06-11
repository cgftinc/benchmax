"""Shared fakes for Chroma tests.

One place to update when the real ChromaChunkSource interface changes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from benchmax.rag.chunkers.models import Chunk

# ---------------------------------------------------------------------------
# FakeCollection — simulates Chroma collection responses
# ---------------------------------------------------------------------------


class FakeCollection:
    """Returns canned results for get() and query() calls.

    Args:
        get_result: Result returned by get().
        query_results_per_call: List of query result dicts, one per
            query() call.  When exhausted, returns empty results.
        query_result: Single query result returned for every call.
            Ignored if query_results_per_call is provided.
        count: Value returned by count().
    """

    def __init__(
        self,
        get_result: dict[str, Any] | None = None,
        query_results_per_call: list[dict[str, Any]] | None = None,
        query_result: dict[str, Any] | None = None,
        count: int = 0,
    ):
        self._get_result = get_result or {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        if query_results_per_call is not None:
            self._query_results = list(query_results_per_call)
        elif query_result is not None:
            self._query_results = [query_result]
        else:
            self._query_results = []
        self._query_idx = 0
        self._count = count

    def get(self, **kwargs):
        return dict(self._get_result)

    def query(self, **kwargs):
        self._last_query_kwargs = kwargs
        if self._query_idx < len(self._query_results):
            result = self._query_results[self._query_idx]
        else:
            result = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        self._query_idx += 1
        return dict(result)

    def count(self):
        return self._count

    def upsert(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# FakeFiles — file-structure awareness fakes
# ---------------------------------------------------------------------------


class FileAwareFakeFiles:
    """File-aware fake that reads metadata from chunks."""

    def check(self) -> bool:
        return True

    @staticmethod
    def chunk_file_path(chunk: Chunk) -> str | None:
        return chunk.get_metadata("file_path") or chunk.get_metadata("file") or None

    @staticmethod
    def chunk_index(chunk: Chunk) -> int | None:
        val = chunk.get_metadata("chunk_index")
        if val is None:
            val = chunk.get_metadata("index")
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def get_file_chunks(self, _path: str) -> list[Chunk]:
        return []

    def get_all_file_paths(self) -> list[str]:
        return []

    def invalidate(self) -> None:
        pass


class NoFileFakeFiles:
    """Fake that simulates missing file-structure metadata."""

    def check(self) -> bool:
        return False

    def chunk_file_path(self, _chunk: Chunk) -> str | None:
        return None

    def chunk_index(self, _chunk: Chunk) -> int | None:
        return None

    def get_file_chunks(self, _path: str) -> list[Chunk]:
        return []

    def get_all_file_paths(self) -> list[str]:
        return []

    def invalidate(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_query_result(
    docs: list[str],
    metas: list[dict[str, Any]] | None = None,
    distances: list[float] | None = None,
) -> dict[str, Any]:
    """Build a mock Chroma query() result dict (nested lists)."""
    if metas is None:
        metas = [{"file_path": "a.md", "chunk_index": i} for i in range(len(docs))]
    if distances is None:
        distances = [0.1 * (i + 1) for i in range(len(docs))]
    return {
        "ids": [[f"id_{i}" for i in range(len(docs))]],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


def make_source(
    collection: FakeCollection,
    files=None,
    embed_fn=None,
    enable_bm25: bool = False,
):
    """Build a ChromaChunkSource with fakes injected -- no real Chroma.

    Constructs a ChromaClient with the fake collection pre-injected,
    then wires it into a ChromaChunkSource without calling __init__.
    """
    from unittest.mock import patch

    from benchmax.rag.corpus.chroma.client import ChromaClient
    from benchmax.rag.corpus.chroma.source import ChromaChunkSource

    # Build a ChromaClient with fakes -- patch has_search_api so it
    # doesn't try to import chromadb.
    with patch("benchmax.rag.corpus.chroma.client.has_search_api", return_value=False):
        chroma = ChromaClient(
            collection_name="test",
            host="localhost",
            port=8000,
            embed_fn=embed_fn,
            enable_bm25=enable_bm25,
        )
    # Inject the fake collection so no real Chroma connection is made
    chroma._collection = collection
    chroma._total_count = collection.count() if collection else None
    # Default fake collection embeds server-side (hosted EF) so dense queries are
    # "safe" by default. Tests covering the local-download / error policy override
    # collection._model.configuration_json with a default/third-party/no EF.
    if collection is not None and not hasattr(collection, "_model"):
        collection._model = SimpleNamespace(
            configuration_json={"embedding_function": {"name": "chroma-cloud-qwen"}}
        )

    source = ChromaChunkSource.__new__(ChromaChunkSource)
    source._chroma = chroma
    source._files = files or NoFileFakeFiles()
    source._search_capabilities = {
        "backend": "chroma",
        "modes": {"vector"},
        "filter_ops": {
            "field": {"eq", "in", "gte", "lte"},
            "logical": {"and", "or", "not"},
        },
        "ranking": {"cosine"},
        "constraints": {"max_top_k": 10000, "vector_dimensions": None},
        "graph_expansion": False,
    }
    return source
