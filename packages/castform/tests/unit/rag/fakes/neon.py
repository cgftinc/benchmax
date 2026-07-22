"""Shared fakes for Neon corpus source tests.

One place to update when the ``NeonChunkSource`` / ``NeonAsyncChunkSource``
collaborator seams change. The fakes stand in for the three collaborators:

- ``FakeQueryRunner`` / ``FakeAsyncQueryRunner`` — the ``_search`` seam over the
  query layer, returning canned ``QueryRow``s per query text (or a flat list).
- ``FakeReadClient`` — the read-only ``NeonClient`` used for the direct reads
  (count/sample/neighbors/top-level/scan). Its ``*_sql`` builders return a tagged
  sentinel that ``execute`` dispatches on, so no psycopg composable is needed.
- ``FakeWriteClient`` — a behavioral read-write ``NeonClient`` simulating the
  versioned-replace ledger (build -> activate pointer swap -> rollback) so the
  atomic-swap / no-stale-rows / rollback contract is testable without a live DB.
"""

from __future__ import annotations

from typing import Any

from castform.rag.corpus.neon.provision import CORPUS_SCHEMA, RO_ROLE
from castform.rag.corpus.neon.query import QueryRow
from castform.rag.corpus.neon.schema import DEFAULT_TEXT_SEARCH_CONFIG
from castform.rag.corpus.neon.source import NeonChunkSource

# ---------------------------------------------------------------------------
# Row / QueryRow builders
# ---------------------------------------------------------------------------


def make_query_row(
    chunk_id: str,
    content: str = "content",
    *,
    metadata: dict[str, Any] | None = None,
    source_file: str = "f.md",
    chunk_index: int = 0,
    surfaced_score: float = 1 / 60,
    native_score: float = -1.0,
    rank: int = 0,
) -> QueryRow:
    """Build a ranked ``QueryRow`` (the shape ``_search.query_rows`` returns)."""
    return QueryRow(
        chunk_id=chunk_id,
        content=content,
        metadata=metadata or {},
        source_file=source_file,
        chunk_index=chunk_index,
        surfaced_score=surfaced_score,
        native_score=native_score,
        rank=rank,
    )


def make_read_row(
    chunk_id: str,
    content: str = "content",
    metadata: dict[str, Any] | None = None,
    source_file: str = "f.md",
    chunk_index: int = 0,
) -> tuple[Any, ...]:
    """Build a ``READ_COLUMNS`` row ``(id, content, metadata, file, index)``."""
    return (chunk_id, content, metadata or {}, source_file, chunk_index)


# ---------------------------------------------------------------------------
# Search seam fakes
# ---------------------------------------------------------------------------


class FakeQueryRunner:
    """Canned ``_search`` seam: returns rows per query text, or a flat list."""

    def __init__(
        self,
        rows_by_query: dict[str, list[QueryRow]] | None = None,
        rows: list[QueryRow] | None = None,
    ) -> None:
        self.rows_by_query = rows_by_query or {}
        self.rows = rows
        self.calls: list[Any] = []

    def query_rows(self, request: Any) -> list[QueryRow]:
        self.calls.append(request)
        if self.rows is not None:
            return list(self.rows)
        return list(self.rows_by_query.get(request.text, []))


class FakeAsyncQueryRunner:
    """Async twin of :class:`FakeQueryRunner` (awaitable ``query_rows``)."""

    def __init__(
        self,
        rows_by_query: dict[str, list[QueryRow]] | None = None,
        rows: list[QueryRow] | None = None,
    ) -> None:
        self.rows_by_query = rows_by_query or {}
        self.rows = rows
        self.calls: list[Any] = []

    async def query_rows(self, request: Any) -> list[QueryRow]:
        self.calls.append(request)
        if self.rows is not None:
            return list(self.rows)
        return list(self.rows_by_query.get(request.text, []))


# ---------------------------------------------------------------------------
# Read-client fake
# ---------------------------------------------------------------------------


class FakeReadClient:
    """Canned read-only ``NeonClient``: ``*_sql`` return tags, ``execute`` dispatches."""

    def __init__(
        self,
        *,
        count: int = 0,
        sample_rows: list[tuple[Any, ...]] | None = None,
        top_level_rows: list[tuple[Any, ...]] | None = None,
        neighbor_rows: list[tuple[Any, ...]] | None = None,
        scan_rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self._count = count
        self._sample_rows = sample_rows or []
        self._top_level_rows = top_level_rows or []
        self._neighbor_rows = neighbor_rows or []
        self._scan_rows = scan_rows or []
        self.calls: list[tuple[Any, Any]] = []

    def count_sql(self, logical_name: str) -> tuple[str, str]:
        return ("count", logical_name)

    def sample_sql(self, logical_name: str) -> tuple[str, str]:
        return ("sample", logical_name)

    def top_level_sql(self, logical_name: str) -> tuple[str, str]:
        return ("top_level", logical_name)

    def neighbors_sql(self, logical_name: str) -> tuple[str, str]:
        return ("neighbors", logical_name)

    def execute(
        self, query: Any, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        self.calls.append((query, params))
        tag = query[0] if isinstance(query, tuple) else None
        if tag == "count":
            return [(self._count,)]
        if tag == "sample":
            return list(self._sample_rows)
        if tag == "top_level":
            return list(self._top_level_rows)
        if tag == "neighbors":
            return list(self._neighbor_rows)
        return []

    def scan_chunks(self, logical_name: str, batch_size: int = 1000) -> Any:
        yield from self._scan_rows


# ---------------------------------------------------------------------------
# Write-client fake (versioned-replace simulator)
# ---------------------------------------------------------------------------


class FakeWriteClient:
    """Behavioral read-write ``NeonClient``: simulates the versioned ledger.

    Tracks per-version row stores, ledger state, and the ``is_current`` pointer so
    a test can prove: activation swaps the pointer atomically, a prior version's
    rows are RETAINED (no stale-row deletion), and rollback re-points to a prior
    activated version. ``execute`` answers the two ingest reads the source issues
    (``to_regclass`` existence probe and the ledger read) off the simulated state.
    """

    def __init__(self) -> None:
        self.versions: dict[int, list[tuple[Any, ...]]] = {}
        self.state: dict[int, str] = {}
        self.current: int | None = None
        self.activations: list[int] = []
        self.rollbacks: list[int] = []

    def execute(
        self, query: Any, params: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        params = params or {}
        if "name" in params:  # to_regclass(...) existence probe
            return [("neon_corpus_versions",)] if self.state else [(None,)]
        if "logical" in params:  # ledger read
            return [
                (version, self.state[version], version == self.current)
                for version in sorted(self.state)
            ]
        return []

    def read_ledger_sql(self) -> tuple[str]:
        return ("read_ledger",)

    def build_version(self, spec: Any, rows: list[tuple[Any, ...]]) -> None:
        self.versions[spec.version] = list(rows)
        self.state[spec.version] = "ready"

    def activate(self, spec: Any, grant: Any) -> None:
        # Atomic pointer swap: prior current is retained, only the pointer moves.
        self.current = spec.version
        self.state[spec.version] = "activated"
        self.activations.append(spec.version)
        self.last_grant = grant

    def rollback(self, logical_name: str, target_version: int) -> None:
        assert self.state.get(target_version) == "activated", (
            "rollback target must be an activated version"
        )
        self.current = target_version
        self.rollbacks.append(target_version)


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------


def make_neon_source(
    *,
    logical_name: str = "mycorpus",
    embed_fn: Any = None,
    search: Any = None,
    read_client: Any = None,
    write_client: Any = None,
) -> NeonChunkSource:
    """Build a ``NeonChunkSource`` with fakes injected — no real Neon connection."""
    source = NeonChunkSource.__new__(NeonChunkSource)
    source._logical_name = logical_name
    source._embed_fn = embed_fn
    source._read_dsn_provider = None
    source._write_dsn_provider = None
    source._schema = CORPUS_SCHEMA
    source._text_search_config = DEFAULT_TEXT_SEARCH_CONFIG
    source._ro_role = RO_ROLE
    source.collection = None
    source._read_client = read_client
    source._write_client = write_client
    source._search = search
    source._active_version = None
    return source


def make_async_neon_source(
    *,
    logical_name: str = "mycorpus",
    embed_fn: Any = None,
    search: Any = None,
) -> Any:
    """Build a ``NeonAsyncChunkSource`` with a fake async search seam injected."""
    from castform.rag.corpus.neon.async_source import NeonAsyncChunkSource

    source = NeonAsyncChunkSource.__new__(NeonAsyncChunkSource)
    source._logical_name = logical_name
    source._embed_fn = embed_fn
    source._read_dsn_provider = None
    source._schema = CORPUS_SCHEMA
    source._text_search_config = DEFAULT_TEXT_SEARCH_CONFIG
    source._search = search
    return source


def constant_embed_fn(dim: int = 3) -> Any:
    """Return a deterministic ``embed_fn`` (a fixed vector per input text)."""
    return lambda texts: [[0.1] * dim for _ in texts]
