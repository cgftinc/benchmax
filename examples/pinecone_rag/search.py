"""Example-local Pinecone adapter for Benchmax ``RagEnv``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from benchmax.rag.search import SearchResult


class PineconeSearch:
    """Lazy, pickle-safe dense retrieval through a Pinecone index host."""

    def __init__(
        self,
        index_host: str,
        *,
        api_key: str,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        namespace: str = "",
        content_field: str = "content",
    ) -> None:
        if not index_host.strip():
            raise ValueError("index_host must be non-empty")
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._index_host = index_host
        self._embed_fn = embed_fn
        self._namespace = namespace
        self._content_field = content_field
        self._api_key = api_key
        self._index: Any = None

    @property
    def available_modes(self) -> list[str]:
        return ["vector"]

    def _get_index(self) -> Any:
        if self._index is None:
            from pinecone import Pinecone

            self._index = Pinecone(api_key=self._api_key).Index(host=self._index_host)
        return self._index

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        if mode not in {"auto", "vector"}:
            raise ValueError("PineconeSearch supports only vector mode")
        vector = (await self._embed_fn([query]))[0]
        response = await asyncio.to_thread(
            self._get_index().query,
            namespace=self._namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )
        return [self._match_to_result(match) for match in (response.matches or [])]

    def _match_to_result(self, match: Any) -> SearchResult:
        metadata = dict(getattr(match, "metadata", None) or {})
        content = metadata.pop(self._content_field, "")
        source = metadata.get("file") or metadata.get("file_path") or ""
        return {
            "content": str(content),
            "source": str(source),
            "metadata": metadata,
            "score": float(getattr(match, "score", 0.0) or 0.0),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_index"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._index = None


__all__ = ["PineconeSearch"]
