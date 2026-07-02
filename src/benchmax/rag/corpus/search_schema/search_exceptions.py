"""Exceptions for backend-agnostic search/filter handling."""

from __future__ import annotations

from typing import Any


class InvalidFilterError(ValueError):
    """Raised when a filter AST is malformed for translation."""

    def __init__(self, backend: str, message: str, predicate: Any | None = None):
        detail = f"[{backend}] invalid filter: {message}"
        if predicate is not None:
            detail += f" | predicate={predicate!r}"
        super().__init__(detail)


class UnsupportedFilterError(ValueError):
    """Raised when a filter requests unsupported capabilities."""

    def __init__(self, backend: str, message: str, predicate: Any | None = None):
        detail = f"[{backend}] unsupported filter: {message}"
        if predicate is not None:
            detail += f" | predicate={predicate!r}"
        super().__init__(detail)


class InvalidSearchSpecError(ValueError):
    """Raised when a search spec has invalid shape for requested mode."""

    def __init__(self, backend: str, message: str, spec: Any | None = None):
        detail = f"[{backend}] invalid search spec: {message}"
        if spec is not None:
            detail += f" | spec={spec!r}"
        super().__init__(detail)


class UnsupportedSearchModeError(ValueError):
    """Raised when a backend cannot satisfy requested retrieval mode."""

    def __init__(self, backend: str, mode: str, supported_modes: set[str]):
        super().__init__(
            f"[{backend}] unsupported search mode '{mode}'. "
            f"Supported modes: {sorted(supported_modes)}"
        )


class LocalEmbeddingDownloadDisallowedError(RuntimeError):
    """Raised when serving a search would download a client-side embedding model.

    The collection has no server-side (hosted) embedding function and no BM25
    index, and the caller supplied no ``embed_fn`` — so embedding a text query
    would make chromadb download and run a local model (e.g. all-MiniLM). We
    refuse rather than trigger that download.
    """

    def __init__(self, backend: str, collection: str):
        super().__init__(
            f"[{backend}] collection {collection!r} has no server-side embedding "
            "function and no BM25 index, so search would download a local "
            "embedding model. Re-ingest the corpus with a hosted embedder "
            "(chroma-cloud-qwen) or a BM25 index, or supply an embed_fn."
        )
