"""Small runtime-only query types for the Neon example backend."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

SearchMode = Literal["lexical", "vector", "hybrid"]
FilterPredicate = Any


class HybridOptions(TypedDict, total=False):
    lexical_weight: float
    vector_weight: float


__all__ = ["FilterPredicate", "HybridOptions", "SearchMode"]
