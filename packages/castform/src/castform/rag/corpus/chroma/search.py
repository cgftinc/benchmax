"""ChromaSearch — pickle-safe search client for RL environments.

Implements :class:`SearchClient` using the shared
:class:`ChromaClient`.  No Chunk or Pydantic dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from castform.platform.credentials import TokenProvider, as_token_provider, env_token
from castform.rag.corpus.search_schema.search_exceptions import (
    LocalEmbeddingDownloadDisallowedError,
)


class ChromaSearch:
    """Pickle-safe Chroma search client for RL environments.

    Stores only serializable connection parameters.  The Chroma client
    is created lazily on first search call (including after unpickle).

    Pass **either** Cloud credentials (``token_provider`` + ``tenant`` +
    ``database``) **or** a self-hosted ``host``. Cloud takes precedence
    if both are set; generated Cloud configurations use Cloud.

    The Chroma Cloud key is resolved per request via ``token_provider``.
    With the default (``env_token("CHROMA_API_KEY")``) the key is read
    from the env at runtime and nothing is frozen — the self-serve /
    hand-written-env path.  Platform-orchestrated training instead passes
    an explicit ``token_provider`` that **bakes** the build-time key into
    the pickled env, because the trainer runs this env in a Ray actor that
    can't read the external secret from its environment at runtime.

    Args:
        collection_name: Name of the Chroma collection.
        tenant: Chroma Cloud tenant ID.
        database: Chroma Cloud database name.
        host: Self-hosted server hostname.
        port: Self-hosted server port (default 8000).
        embed_fn: Async custom embedding function. When ``None``, Chroma's
            built-in embeddings are used.
        enable_bm25: Enable BM25 for lexical/hybrid modes.
        content_attr: Metadata fields to treat as content.
        token_provider: Optional override — a callable resolving the key
            per call, or a literal key (string sugar). Defaults to reading
            ``CHROMA_API_KEY``.
    """

    def __init__(
        self,
        collection_name: str,
        tenant: str | None = None,
        database: str | None = None,
        host: str | None = None,
        port: int = 8000,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        enable_bm25: bool = True,
        content_attr: list[str] | None = None,
        token_provider: str | TokenProvider | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._tenant = tenant
        self._database = database
        self._host = host
        self._port = port
        self._embed_fn = embed_fn
        self._enable_bm25 = enable_bm25
        self._content_attr = content_attr
        self._token_provider = as_token_provider(
            token_provider, env_token("CHROMA_API_KEY")
        )
        self._client: Any = None

    def _resolve_api_key(self) -> str | None:
        """Resolve the API key, returning None for self-hosted mode.

        When Cloud credentials (tenant+database) are set without a
        fallback host, let RuntimeError propagate so the user sees a
        clear "missing CHROMA_API_KEY" message instead of a confusing
        downstream failure from ChromaClient getting api_key=None.
        """
        if self._host and not (self._tenant and self._database):
            return None
        try:
            return self._token_provider()
        except RuntimeError:
            if not self._host:
                raise
            return None

    def _get_client(self) -> Any:
        if self._client is None:
            from .client import ChromaClient

            api_key = self._resolve_api_key()
            self._client = ChromaClient(
                collection_name=self._collection_name,
                api_key=api_key,
                tenant=self._tenant,
                database=self._database,
                host=self._host,
                port=self._port,
                # Search computes custom embeddings itself so Chroma never
                # invokes an async embedder through its synchronous callback.
                embed_fn=None,
                enable_bm25=self._enable_bm25,
                content_attr=self._content_attr,
            )
        return self._client

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results."""
        client = self._get_client()
        # Initialize the collection first so capabilities reflect the real index
        # (BM25 downgrade) and the embedder config is readable below.
        await asyncio.to_thread(client.get_collection)
        modes = client.modes
        has_lexical = "lexical" in modes

        # Never download a client-side embedding model at inference/rollout time.
        # When a dense embed isn't safe — no embed_fn and no Chroma-hosted
        # server-side embedding function — use the BM25 lexical index if the
        # collection has one, otherwise refuse rather than fetch all-MiniLM.
        dense_embed_is_safe = self._embed_fn is not None or client.dense_embed_is_safe()
        if not dense_embed_is_safe:
            if not has_lexical:
                raise LocalEmbeddingDownloadDisallowedError(
                    "chroma", self._collection_name
                )
            mode = "lexical"
        elif mode == "auto":
            if "hybrid" in modes:
                mode = "hybrid"
            elif has_lexical:
                mode = "lexical"
            else:
                mode = "vector"
        elif mode not in modes:
            raise ValueError(
                f"ChromaSearch does not support mode '{mode}'. "
                f"Available modes: {sorted(modes)}"
            )

        if client.search_api and mode in ("lexical", "hybrid"):
            vec = await self.embed(query) if mode == "hybrid" else None
            rows = await asyncio.to_thread(
                client.search_api_raw,
                text_query=query,
                vector_query=vec,
                mode=mode,
                top_k=top_k,
            )
        else:
            vec = await self.embed(query)
            rows = await asyncio.to_thread(
                client.query_raw,
                text_query=query,
                vector_query=vec,
                top_k=top_k,
            )

        return [
            {
                "content": client.extract_content(r["content"], r["metadata"]),
                "source": str(
                    r["metadata"].get("file") or r["metadata"].get("file_path") or ""
                ),
                "metadata": r["metadata"],
                "score": float(r.get("score", 0.0) or 0.0),
            }
            for r in rows
        ]

    async def embed(self, text: str) -> list[float] | None:
        """Return embedding vector, or None for auto-embed."""
        if self._embed_fn is None:
            return None
        client = self._get_client()
        vec = (await self._embed_fn([text]))[0]
        client._validate_embed_dim(vec)
        return vec

    @property
    def available_modes(self) -> list[str]:
        return sorted(self._get_client().modes)

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "backend": "chroma",
            "collection_name": self._collection_name,
            "enable_bm25": self._enable_bm25,
        }
        if self._tenant and self._database:
            api_key = self._resolve_api_key()
            params.update(
                mode="cloud",
                tenant=self._tenant,
                database=self._database,
                api_key=(api_key[:8] + "...") if api_key else None,
            )
        else:
            params.update(mode="self_hosted", host=self._host, port=self._port)
        return params

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
