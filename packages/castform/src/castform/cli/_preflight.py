"""Read-only checks for Castform's optional environment dependencies."""

from __future__ import annotations

import importlib.util

# One sentinel import name per extra — used by ``castform doctor`` to report whether an
# extra is installed without importing the (heavy) package itself.
_EXTRA_SENTINEL: dict[str, str] = {
    "rag": "keybert",
    "turbopuffer": "turbopuffer",
    "pinecone": "pinecone",
    "chroma": "chromadb",
}


def extra_is_installed(extra: str) -> bool:
    """True if ``extra``'s sentinel module resolves (no import executed)."""
    sentinel = _EXTRA_SENTINEL.get(extra)
    if sentinel is None:
        return False
    try:
        return importlib.util.find_spec(sentinel) is not None
    except (ImportError, ValueError):
        # A parent package missing (or a namespace edge case) => treat as absent.
        return False
