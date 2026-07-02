"""Shared fakes for Pinecone tests.

One place to update when the real PineconeIndexClient interface changes.
"""

from __future__ import annotations

from types import SimpleNamespace

from benchmax.rag.chunkers.models import Chunk

# ---------------------------------------------------------------------------
# FakeIndex — simulates Pinecone Index responses
# ---------------------------------------------------------------------------


class FakeIndex:
    """Returns canned matches for each query call.

    Args:
        matches_per_call: List of match lists, one per query() call.
            When exhausted, returns empty matches.
        matches: Single list of matches returned for every call.
            Ignored if matches_per_call is provided.
        dimension: Index dimension returned by describe_index_stats.
    """

    def __init__(
        self,
        matches_per_call: list[list] | None = None,
        matches: list | None = None,
        dimension: int = 3,
    ):
        if matches_per_call is not None:
            self.matches_per_call = list(matches_per_call)
        elif matches is not None:
            self.matches_per_call = [list(matches)]
        else:
            self.matches_per_call = []
        self._call_idx = 0
        self._dimension = dimension

    def query(self, **kwargs):
        if self._call_idx < len(self.matches_per_call):
            matches = self.matches_per_call[self._call_idx]
        else:
            matches = []
        self._call_idx += 1
        return SimpleNamespace(matches=list(matches))

    def describe_index_stats(self):
        total = sum(len(m) for m in self.matches_per_call)
        return SimpleNamespace(dimension=self._dimension, total_vector_count=total)


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
# FakeClient — simulates PineconeIndexClient
# ---------------------------------------------------------------------------


class FakeClient:
    """Configurable fake PineconeIndexClient.

    Args:
        index: FakeIndex to delegate query() calls to.
        embed_fn: Embedding function. Defaults to a fixed 3-d vector.
        field_mapping: Maps Pinecone metadata keys → internal names.
    """

    def __init__(
        self,
        index: FakeIndex,
        embed_fn=None,
        field_mapping: dict[str, str] | None = None,
    ):
        self._index = index
        default_mapping = {"content": "content", "file_path": "file_path"}
        self._field_mapping = field_mapping or default_mapping
        self._reverse_mapping = {v: k for k, v in self._field_mapping.items()}
        self._vector_dim = index._dimension
        self._embed_model = "test-model"
        self.embed_fn = embed_fn or (lambda texts: [[0.1, 0.2, 0.3]] * len(texts))

    def query(self, **kwargs):
        return self._index.query(**kwargs)

    def _get_index(self):
        return self._index

    def _pc_field(self, internal_name: str) -> str:
        return self._reverse_mapping.get(internal_name, internal_name)

    def _internal_field(self, pc_name: str) -> str:
        return self._field_mapping.get(pc_name, pc_name)

    def zero_vector(self) -> list[float]:
        return [0.0] * self._vector_dim

    def vector_type(self) -> str:
        return "dense"

    def namespace_vector_count(self) -> int:
        return sum(len(m) for m in self._index.matches_per_call)

    def _all_matches(self) -> list:
        return [m for batch in self._index.matches_per_call for m in batch]

    def sample_ids(self, n: int) -> list[str]:
        ids = [m.id for m in self._all_matches()]
        return ids[:n]

    def fetch_by_ids_raw(self, ids: list[str]) -> list[dict]:
        wanted = set(ids)
        return [self.match_to_raw(m) for m in self._all_matches() if m.id in wanted]

    def match_content(self, match) -> str:
        metadata = getattr(match, "metadata", {}) or {}
        content_key = self._pc_field("content")
        return str(metadata.get(content_key, ""))

    def match_to_raw(self, match) -> dict:
        metadata = getattr(match, "metadata", {}) or {}
        content_key = self._pc_field("content")
        content = str(metadata.get(content_key, ""))
        attrs = {}
        for k, v in metadata.items():
            internal = self._internal_field(k)
            if internal != "content":
                attrs[internal] = v
        attrs["_pinecone_id"] = match.id
        return {
            "id": match.id,
            "content": content,
            "metadata": attrs,
            "score": getattr(match, "score", 0.0) or 0.0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_match(match_id: str, content: str, score: float = 0.9, **extra_meta):
    """Build a fake Pinecone query match."""
    metadata = {"content": content, **extra_meta}
    return SimpleNamespace(id=match_id, metadata=metadata, score=score)


def make_source(
    index: FakeIndex,
    files=None,
    embed_fn=None,
    field_mapping: dict[str, str] | None = None,
):
    """Build a PineconeChunkSource with fakes injected — no real Pinecone."""
    from benchmax.rag.corpus.pinecone.source import PineconeChunkSource

    source = PineconeChunkSource.__new__(PineconeChunkSource)
    source._client = FakeClient(index=index, embed_fn=embed_fn, field_mapping=field_mapping)
    source._files = files or NoFileFakeFiles()
    # Pinecone is always vector-only regardless of embed_fn — unlike
    # turbopuffer where embed_fn unlocks hybrid mode.
    source._search_capabilities = {
        "backend": "pinecone",
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
