"""PineconeSearch — pickle-safe search client for RL environments.

Implements :class:`SearchClient` using the shared
:class:`PineconeIndexClient`.  No Chunk or Pydantic dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from castform.platform.credentials import TokenProvider, as_token_provider, env_token


class PineconeSearch:
    """Pickle-safe Pinecone search client for RL environments.

    Stores only serializable connection parameters.  The SDK client is
    created lazily on first search call (including after unpickle).

    The Pinecone key is resolved per request via ``token_provider``. With the
    default (``env_token("PINECONE_API_KEY")``) the key is read from the env at
    runtime and nothing is frozen — the self-serve / hand-written-env path.
    Platform-orchestrated training instead passes an explicit ``token_provider``
    that **bakes** the build-time key into the pickled env, because the trainer
    runs this env in a Ray actor that can't read the external secret from its
    environment at runtime. Pinecone is an external provider — we neither mint
    nor rotate this key.

    Args:
        index_name: Name of the Pinecone index.
        index_host: Optional host URL (bypasses index name lookup).
        namespace: Pinecone namespace within the index (default ``""``).
        embed_fn: Async custom embedding function. When ``None``, Pinecone's
            hosted Inference API is used.
        embed_model: Pinecone hosted embedding model name. Ignored
            when ``embed_fn`` is provided.
        field_mapping: Maps Pinecone metadata keys to internal names.
        content_field: Pinecone metadata key holding the chunk text — sugar
            over ``field_mapping`` for BYO indexes that don't use ``content``.
        token_provider: Optional override — a callable resolving the key per
            call, or a literal key (string sugar). Defaults to reading
            ``PINECONE_API_KEY``.
    """

    def __init__(
        self,
        index_name: str,
        *,
        index_host: str | None = None,
        namespace: str = "",
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        embed_model: str = "multilingual-e5-large",
        field_mapping: dict[str, str] | None = None,
        content_field: str | None = None,
        token_provider: str | TokenProvider | None = None,
    ) -> None:
        self._index_name = index_name
        self._index_host = index_host
        self._namespace = namespace
        self._embed_fn = embed_fn
        self._embed_model = embed_model
        self._field_mapping = field_mapping
        self._content_field = content_field
        self._token_provider = as_token_provider(token_provider, env_token("PINECONE_API_KEY"))
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from .index_client import PineconeIndexClient

            self._client = PineconeIndexClient(
                api_key=self._token_provider(),
                index_name=self._index_name,
                index_host=self._index_host,
                namespace=self._namespace,
                embed_fn=None,
                embed_model=self._embed_model,
                field_mapping=self._field_mapping,
                content_field=self._content_field,
            )
        return self._client

    async def search(
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
        vec = await self.embed(query)
        assert vec is not None
        result = await asyncio.to_thread(client.query, vector=vec, top_k=top_k)
        return [client.match_to_raw(m) for m in (result.matches or [])]

    async def embed(self, text: str) -> list[float] | None:
        """Return embedding vector for *text*."""
        if self._embed_fn is not None:
            return (await self._embed_fn([text]))[0]
        client = self._get_client()
        return (await asyncio.to_thread(client.embed_fn, [text]))[0]

    @property
    def available_modes(self) -> list[str]:
        """Pinecone is vector-only."""
        return ["vector"]

    def get_params(self) -> dict[str, Any]:
        return {
            "backend": "pinecone",
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
