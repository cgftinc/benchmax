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
