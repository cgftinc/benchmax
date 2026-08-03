"""Example-local, pickle-safe Neon SearchClient implementation.

Implements the minimal ``SearchClient`` protocol (``search`` and
``available_modes``) without inheriting it. No psycopg import occurs at module
load. The explicit read-only database URL is bundled with the environment, while
live connections are dropped when serialized.

The single hybrid-RRF fusion and the surfaced-score formula are owned by
``query.py``; this module re-exports their public names
(:data:`SURFACED_RANK_K`, :class:`QueryHit`, :func:`surfaced_score`,
:class:`NeonQueryRequest`, :func:`fuse_rrf`) so ``neon_backend.search``
stays a stable import path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from benchmax.rag.search import SearchResult

from neon_backend.config import CORPUS_SCHEMA
from neon_backend.query import (
    SURFACED_RANK_K,
    NeonQueryRequest,
    QueryHit,
    fuse_rrf,
    surfaced_score,
)
from neon_backend.reader import AsyncNeonReader
from neon_backend.schema import DEFAULT_TEXT_SEARCH_CONFIG
from neon_backend.types import SearchMode

if TYPE_CHECKING:
    from neon_backend.query import QueryRow

__all__ = [
    "SURFACED_RANK_K",
    "NeonQueryRequest",
    "NeonSearch",
    "QueryHit",
    "fuse_rrf",
    "surfaced_score",
]

# ``search(mode="auto")`` resolves to the richest available mode, best-first.
_AUTO_MODE_PREFERENCE: tuple[SearchMode, ...] = ("hybrid", "vector", "lexical")


class NeonSearch:
    """Pickle-safe Neon corpus search client for RL environments.

    Args:
        table: Logical corpus name to query (resolved to the active-version view).
        embed_fn: Embedding function for vector/hybrid modes. ``Callable[[list[str]],
            list[list[float]]]`` — the shape every provider expects. When absent,
            only lexical search is available.
        database_url: Explicit read-only Postgres connection URL.
        schema: Postgres schema the corpus objects live in (qualifies the BM25
            index regclass for the RO invoker). Not a credential.
        text_search_config: ``regconfig`` the corpus tsvector was built with; must
            match the version's baked config or bm25 scores drift.
    """

    def __init__(
        self,
        table: str,
        *,
        database_url: str,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        schema: str = CORPUS_SCHEMA,
        text_search_config: str = DEFAULT_TEXT_SEARCH_CONFIG,
    ) -> None:
        if not database_url:
            raise ValueError("database_url must be non-empty")
        self._table = table
        self._embed_fn = embed_fn
        self._schema = schema
        self._text_search_config = text_search_config
        self._database_url = database_url
        self._client: Any = None

    # --- lazy client + version resolution ------------------------------------

    def _get_client(self) -> AsyncNeonReader:
        """Build the concurrency-safe read transport lazily."""
        if self._client is None:
            self._client = AsyncNeonReader(
                self._database_url,
                logical_name=self._table,
                schema=self._schema,
                text_search_config=self._text_search_config,
            )
        return self._client

    async def _run(self, request: NeonQueryRequest) -> list[QueryRow]:
        return await self._get_client().query(request)

    # --- SearchClient protocol -----------------------------------------------

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search and return structured results best-first.

        Returns dicts keyed ``content`` / ``source`` / ``metadata`` / ``score``,
        where ``score`` is the surfaced reciprocal-rank (higher-better, uniform
        across modes — never the raw native score). ``mode="auto"`` picks the
        richest available mode (hybrid > vector > lexical). Vector/hybrid embed
        ``query`` via ``embed_fn``.
        """
        resolved = self._resolve_mode(mode)
        request = NeonQueryRequest(
            mode=resolved,
            top_k=top_k,
            text=query,
            vector=await self._embed_tuple(query) if resolved in ("vector", "hybrid") else None,
        )
        return [
            {
                "content": row.content,
                "source": row.source_file,
                "metadata": row.metadata,
                "score": row.surfaced_score,
            }
            for row in await self._run(request)
        ]

    async def embed(self, text: str) -> list[float] | None:
        """Return an embedding for *text*, or ``None`` if no embedder is set."""
        if self._embed_fn is None:
            return None
        return (await self._embed_fn([text]))[0]

    @property
    def available_modes(self) -> list[str]:
        """Modes gated on ``embed_fn``: lexical-only without, +vector/+hybrid with."""
        modes = ["lexical"]
        if self._embed_fn is not None:
            modes += ["vector", "hybrid"]
        return sorted(modes)

    def get_params(self) -> dict[str, Any]:
        """Serializable connection params for inspection — NO credential."""
        return {"backend": "neon", "table": self._table, "schema": self._schema}

    # --- richer internal API (QueryRequest-driven) ---------------------------

    async def query(self, request: NeonQueryRequest) -> list[QueryHit]:
        """Run one query, returning ``QueryHit`` rows best-first (id + scores)."""
        return [row.to_hit() for row in await self._run(request)]

    async def search_content(self, request: NeonQueryRequest) -> list[str]:
        """Return content strings only (cloudpickle-safe rollout path)."""
        return [row.content for row in await self._run(request)]

    # --- helpers -------------------------------------------------------------

    def _resolve_mode(self, mode: str) -> SearchMode:
        modes = self.available_modes
        if mode == "auto":
            return next(m for m in _AUTO_MODE_PREFERENCE if m in modes)
        if mode not in modes:
            hint = (
                " pass an embed_fn for vector/hybrid search."
                if mode in ("vector", "hybrid")
                else ""
            )
            raise ValueError(f"search mode {mode!r} not available; available modes: {modes}.{hint}")
        return mode  # type: ignore[return-value]

    async def _embed_tuple(self, query: str) -> tuple[float, ...]:
        if self._embed_fn is None:
            raise ValueError("vector/hybrid search requires embed_fn")
        return tuple((await self._embed_fn([query]))[0])

    # --- pickle safety -------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None  # never pickle a live psycopg connection
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
