"""Backend-neutral search schema and predicate AST models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

FieldOperator = Literal["eq", "in", "gte", "lte", "contains_any", "contains_all"]
LogicalOperator = Literal["and", "or", "not"]
SearchMode = Literal["lexical", "vector", "hybrid"]


@dataclass(frozen=True)
class FieldPredicate:
    """Leaf predicate over one metadata field."""

    field: str
    op: FieldOperator
    value: Any


@dataclass(frozen=True)
class AndPredicate:
    """Logical AND over child predicates."""

    clauses: tuple["FilterPredicate", ...]


@dataclass(frozen=True)
class OrPredicate:
    """Logical OR over child predicates."""

    clauses: tuple["FilterPredicate", ...]


@dataclass(frozen=True)
class NotPredicate:
    """Logical NOT over one child predicate."""

    clause: "FilterPredicate"


FilterPredicate: TypeAlias = FieldPredicate | AndPredicate | OrPredicate | NotPredicate


class HybridOptions(TypedDict, total=False):
    """Options for hybrid lexical+vector retrieval.

    Keys:
        lexical_weight: Relative weight for lexical/BM25 contribution.
        vector_weight: Relative weight for vector-similarity contribution.
        blend: Optional backend-specific blending strategy name.
    """

    lexical_weight: float
    vector_weight: float
    blend: str


class SearchSpec(TypedDict):
    """Backend-neutral search request.

    Keys:
        mode: Retrieval mode. One of: "lexical", "vector", "hybrid".
        top_k: Maximum number of results requested.
        text_query: Text query string; required for lexical/hybrid modes.
        vector_query: Query embedding vector; required for vector/hybrid modes.
        hybrid: Optional hybrid blending knobs (see HybridOptions).
        filter: Optional backend-neutral predicate AST for metadata filtering.
        include_metadata: Whether result metadata should be included in the response.
        include_values: Whether raw backend values/vectors should be included in the response.
    """

    mode: SearchMode
    top_k: int
    text_query: NotRequired[str | None]
    vector_query: NotRequired[list[float] | None]
    hybrid: NotRequired[HybridOptions | None]
    filter: NotRequired[FilterPredicate | None]
    include_metadata: NotRequired[bool]
    include_values: NotRequired[bool]


class FilterOps(TypedDict):
    """Supported filter operators reported by a backend."""

    field: set[FieldOperator]
    logical: set[LogicalOperator]


class SearchCapabilities(TypedDict):
    """Capability schema for one backend.

    Keys:
        backend: Backend identifier (e.g., "corpora", "turbopuffer").
        modes: Supported retrieval modes for this source.
        filter_ops: Supported field/logical filter operators.
        ranking: Supported ranking/reranking knobs.
        constraints: Additional limits/requirements (e.g., max_top_k).
        graph_expansion: Whether chunk-graph expansion is supported.
    """

    backend: str
    modes: set[SearchMode]
    filter_ops: FilterOps
    ranking: set[str]
    constraints: dict[str, Any]
    graph_expansion: bool


def required_operators(predicate: FilterPredicate | None) -> tuple[set[str], set[str]]:
    """Return required field/logical operators for a predicate tree."""
    field_ops: set[str] = set()
    logical_ops: set[str] = set()

    if predicate is None:
        return field_ops, logical_ops

    if isinstance(predicate, FieldPredicate):
        field_ops.add(predicate.op)
        return field_ops, logical_ops

    if isinstance(predicate, AndPredicate):
        logical_ops.add("and")
        for clause in predicate.clauses:
            child_field_ops, child_logical_ops = required_operators(clause)
            field_ops |= child_field_ops
            logical_ops |= child_logical_ops
        return field_ops, logical_ops

    if isinstance(predicate, OrPredicate):
        logical_ops.add("or")
        for clause in predicate.clauses:
            child_field_ops, child_logical_ops = required_operators(clause)
            field_ops |= child_field_ops
            logical_ops |= child_logical_ops
        return field_ops, logical_ops

    if isinstance(predicate, NotPredicate):
        logical_ops.add("not")
        child_field_ops, child_logical_ops = required_operators(predicate.clause)
        field_ops |= child_field_ops
        logical_ops |= child_logical_ops
        return field_ops, logical_ops

    # Defensive fallback for static type drift.
    raise TypeError(f"Unsupported predicate type: {type(predicate).__name__}")


def validate_search_spec_shape(spec: SearchSpec) -> list[str]:
    """Return a list of validation errors for a search spec shape."""
    errors: list[str] = []
    mode = spec.get("mode")
    top_k = spec.get("top_k")
    text_query = spec.get("text_query")
    vector_query = spec.get("vector_query")

    if mode not in {"lexical", "vector", "hybrid"}:
        errors.append("mode must be one of: lexical, vector, hybrid")

    if not isinstance(top_k, int) or top_k <= 0:
        errors.append("top_k must be a positive integer")

    if mode == "lexical":
        if not isinstance(text_query, str) or not text_query.strip():
            errors.append("lexical mode requires non-empty text_query")
    elif mode == "vector":
        if not isinstance(vector_query, list) or not vector_query:
            errors.append("vector mode requires non-empty vector_query")
    elif mode == "hybrid":
        has_text = isinstance(text_query, str) and bool(text_query.strip())
        has_vector = isinstance(vector_query, list) and bool(vector_query)
        if not (has_text and has_vector):
            errors.append("hybrid mode requires both text_query and vector_query")

    return errors
