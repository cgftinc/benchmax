"""CorporaSearch — pickle-safe search client for RL environments.

Delegates to the Corpora API HTTP client. Lexical (BM25) only.
"""

from __future__ import annotations

from typing import Any

from .client import CorpusClient


class CorporaSearch:
    """Pickle-safe Corpora API search client for RL environments.

    Supports lexical (BM25) search only.

    Args:
        api_key: CGFT API key.
        corpus_name: Name of the corpus.
        base_url: Corpora API base URL.
        corpus_id: Optional corpus ID (skips name lookup).
    """

    def __init__(
        self,
        api_key: str,
        corpus_name: str,
        base_url: str,
        *,
        corpus_id: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._corpus_name = corpus_name
        self._base_url = base_url
        self._corpus_id = corpus_id
        self._client: CorpusClient | None = None

    def _get_client(self) -> CorpusClient:
        if self._client is None:
            self._client = CorpusClient(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _get_corpus_id(self) -> str:
        if self._corpus_id is None:
            client = self._get_client()
            corpus = client.get_or_create_corpus(self._corpus_name)
            self._corpus_id = corpus.id
        return self._corpus_id

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search and return structured results. Lexical (BM25) only."""
        if mode not in ("auto", "lexical"):
            raise ValueError(
                f"CorporaSearch only supports 'lexical' mode, got '{mode}'. "
                f"The Corpora API uses BM25 search only."
            )
        client = self._get_client()
        corpus_id = self._get_corpus_id()
        result = client.search(corpus_id=corpus_id, query=query, limit=top_k)
        return [
            {
                "content": chunk.content,
                "source": (chunk.metadata or {}).get("file", ""),
                "metadata": dict(chunk.metadata or {}),
                "score": chunk.score or 0.0,
            }
            for chunk in result.results
        ]

    def embed(self, text: str) -> list[float] | None:
        """Corpora API does not support embeddings."""
        return None

    @property
    def available_modes(self) -> list[str]:
        return ["lexical"]

    def get_params(self) -> dict[str, Any]:
        return {
            "backend": "corpora",
            "api_key": self._api_key[:8] + "...",
            "corpus_name": self._corpus_name,
            "base_url": self._base_url,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client = None
