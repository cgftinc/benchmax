"""Example-local Chroma adapter for Benchmax ``RagEnv``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from benchmax.rag.search import SearchResult


class ChromaSearch:
    """Lazy, pickle-safe dense retrieval from Chroma Cloud or an HTTP server."""

    def __init__(
        self,
        collection_name: str,
        *,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        api_key: str | None = None,
    ) -> None:
        cloud = bool(tenant and database)
        if cloud == bool(host):
            raise ValueError("configure either Chroma Cloud tenant/database or a host")
        if cloud and not api_key:
            raise ValueError("Chroma Cloud requires api_key")
        self._collection_name = collection_name
        self._embed_fn = embed_fn
        self._tenant = tenant
        self._database = database
        self._host = host
        self._port = port
        self._ssl = ssl
        self._api_key = api_key
        self._client: Any = None
        self._collection: Any = None

    @property
    def available_modes(self) -> list[str]:
        """Return configured modes without initializing Chroma or reading a key."""

        return ["vector"]

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb

        if self._tenant and self._database:
            self._client = chromadb.CloudClient(
                api_key=self._api_key,
                tenant=self._tenant,
                database=self._database,
            )
        else:
            self._client = chromadb.HttpClient(
                host=self._host or "localhost",
                port=self._port,
                ssl=self._ssl,
            )
        self._collection = self._client.get_collection(self._collection_name)
        return self._collection

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        if mode not in {"auto", "vector"}:
            raise ValueError("ChromaSearch supports only vector mode in this example")
        vector = (await self._embed_fn([query]))[0]
        result = await asyncio.to_thread(
            self._get_collection().query,
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        rows: list[SearchResult] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = dict(metadata or {})
            distance = distances[index] if index < len(distances) else None
            score = 1.0 / (1.0 + max(0.0, float(distance))) if distance is not None else 0.0
            rows.append(
                {
                    "content": str(document or ""),
                    "source": str(metadata.get("file") or metadata.get("file_path") or ""),
                    "metadata": metadata,
                    "score": score,
                }
            )
        return rows

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        state["_collection"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
        self._collection = None


__all__ = ["ChromaSearch"]
