"""PineconeSearch — pickle-safe search client for RL environments.

Implements :class:`SearchClient` using the shared
:class:`PineconeIndexClient`.  No Chunk or Pydantic dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PineconeSearch:
    """Pickle-safe Pinecone search client for RL environments.

    Stores only serializable connection parameters.  The SDK client is
    created lazily on first search call (including after unpickle).

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index.
        index_host: Optional host URL (bypasses index name lookup).
        namespace: Pinecone namespace within the index (default ``""``).
        embed_fn: Custom embedding function. When ``None``, Pinecone's
            hosted Inference API is used.
        embed_model: Pinecone hosted embedding model name. Ignored
            when ``embed_fn`` is provided.
        field_mapping: Maps Pinecone metadata keys to internal names.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        *,
        index_host: str | None = None,
        namespace: str = "",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        embed_model: str = "multilingual-e5-large",
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._index_host = index_host
        self._namespace = namespace
        self._embed_fn = embed_fn
        self._embed_model = embed_model
        self._field_mapping = field_mapping
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from .index_client import PineconeIndexClient

            self._client = PineconeIndexClient(
                api_key=self._api_key,
                index_name=self._index_name,
                index_host=self._index_host,
                namespace=self._namespace,
                embed_fn=self._embed_fn,
                embed_model=self._embed_model,
                field_mapping=self._field_mapping,
            )
        return self._client

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results."""
        if mode not in ("auto", "vector"):
            raise ValueError(
                f"PineconeSearch only supports 'vector' mode, got '{mode}'. "
                f"Pinecone does not support lexical/hybrid search."
            )
        client = self._get_client()
        vec = client.embed_fn([query])[0]
        result = client.query(vector=vec, top_k=top_k)
        return [client.match_to_raw(m) for m in (result.matches or [])]

    def embed(self, text: str) -> list[float] | None:
        """Return embedding vector for *text*."""
        return self._get_client().embed_fn([text])[0]

    @property
    def available_modes(self) -> list[str]:
        """Pinecone is vector-only."""
        return ["vector"]

    def get_params(self) -> dict[str, Any]:
        return {
            "backend": "pinecone",
            "api_key": self._api_key[:8] + "...",
            "index_name": self._index_name,
            "namespace": self._namespace,
            "embed_model": self._embed_model,
        }

    # Pickle: strip the client, reconstruct lazily.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
