"""Async SearchClient protocol — pickle-safe retrieval for RL environments.

No Pydantic, no Chunk objects. Designed to survive cloudpickle roundtrips for
remote training and to keep model authentication and network I/O on the
caller's event loop.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class SearchResult(TypedDict):
    """One provider-independent result returned to :class:`RagEnv`."""

    content: str
    source: str
    metadata: dict[str, Any]
    score: float


@runtime_checkable
class SearchClient(Protocol):
    """Minimal async search interface for RL training environments.

    Implementations store only serializable connection parameters and
    reconstruct SDK clients lazily.  No Chunk or Pydantic dependency.

    This is the env-facing search interface.  For the full data-prep
    interface (chunking, indexing, metadata, file awareness), see
    :class:`ChunkSource`.
    """

    async def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search and return structured results.

        Args:
            query: Text query string.
            mode: Search mode (``"vector"``, ``"lexical"``, ``"hybrid"``,
                or ``"auto"`` to pick the best available).
            top_k: Maximum number of results.

        Returns:
            List of dicts with keys ``content``, ``source``, ``metadata``,
            and ``score``, ordered by relevance.
        """
        ...

    @property
    def available_modes(self) -> list[str]:
        """Supported modes without performing I/O or resolving credentials.

        ``RagEnv`` reads this property during construction, before a bundle
        is uploaded. Implementations must derive it from local configuration.
        """
        ...


__all__ = ["SearchClient", "SearchResult"]
