"""SearchClient protocol — pickle-safe search interface for RL environments.

No Pydantic, no Chunk objects. Designed to survive cloudpickle roundtrips for remote training.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SearchClient(Protocol):
    """Minimal search interface for RL training environments.

    Implementations store only serializable connection parameters and
    reconstruct SDK clients lazily.  No Chunk or Pydantic dependency.

    This is the env-facing search interface.  For the full data-prep
    interface (chunking, indexing, metadata, file awareness), see
    :class:`ChunkSource`.
    """

    def search(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
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

    def embed(self, text: str) -> list[float] | None:
        """Return an embedding vector for *text*, or ``None`` if the
        backend auto-embeds.
        """
        ...

    @property
    def available_modes(self) -> list[str]:
        """Supported search modes (e.g. ``["vector"]`` or
        ``["lexical", "vector", "hybrid"]``).
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serializable connection parameters.

        Used for inspection and debugging.  The returned dict must be
        JSON-serializable.
        """
        ...
