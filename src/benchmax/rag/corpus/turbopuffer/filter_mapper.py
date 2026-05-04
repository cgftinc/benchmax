"""Turbopuffer backend filter translation from shared predicate AST."""

from __future__ import annotations

from typing import Any

from benchmax.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from benchmax.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    FilterPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)

_TPUF_FIELD_OP_MAP = {
    "eq": "Eq",
    "in": "In",
    "gte": "Gte",
    "lte": "Lte",
}


def _ensure_supported(capabilities: SearchCapabilities, predicate: FilterPredicate | None) -> None:
    backend = str(capabilities.get("backend", "unknown"))
    filter_ops = capabilities.get("filter_ops", {"field": set(), "logical": set()})
    field_operators = set(filter_ops.get("field", []))
    logical_operators = set(filter_ops.get("logical", []))

    if predicate is None:
        return

    if isinstance(predicate, FieldPredicate):
        if predicate.op not in field_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message=f"field operator '{predicate.op}' is not supported",
                predicate=predicate,
            )
        if not isinstance(predicate.field, str) or not predicate.field.strip():
            raise InvalidFilterError(
                backend=backend,
                message="field predicate must have a non-empty field name",
                predicate=predicate,
            )
        return

    if isinstance(predicate, AndPredicate):
        if "and" not in logical_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message="logical operator 'and' is not supported",
                predicate=predicate,
            )
        if not predicate.clauses:
            raise InvalidFilterError(
                backend=backend,
                message="'and' must include at least one clause",
                predicate=predicate,
            )
        for clause in predicate.clauses:
            _ensure_supported(capabilities, clause)
        return

    if isinstance(predicate, OrPredicate):
        if "or" not in logical_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message="logical operator 'or' is not supported",
                predicate=predicate,
            )
        if not predicate.clauses:
            raise InvalidFilterError(
                backend=backend,
                message="'or' must include at least one clause",
                predicate=predicate,
            )
        for clause in predicate.clauses:
            _ensure_supported(capabilities, clause)
        return

    if isinstance(predicate, NotPredicate):
        if "not" not in logical_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message="logical operator 'not' is not supported",
                predicate=predicate,
            )
        _ensure_supported(capabilities, predicate.clause)
        return

    raise InvalidFilterError(
        backend=backend,
        message=f"unexpected predicate type '{type(predicate).__name__}'",
        predicate=predicate,
    )


def to_turbopuffer_filters(
    predicate: FilterPredicate | None,
    capabilities: SearchCapabilities,
) -> Any | None:
    """Translate predicate AST into Turbopuffer filter expression."""
    _ensure_supported(capabilities, predicate)
    if predicate is None:
        return None

    if isinstance(predicate, FieldPredicate):
        op = _TPUF_FIELD_OP_MAP.get(predicate.op)
        if op is None:
            raise UnsupportedFilterError(
                backend=str(capabilities.get("backend", "unknown")),
                message=f"field operator '{predicate.op}' has no Turbopuffer mapping",
                predicate=predicate,
            )
        return [predicate.field, op, predicate.value]
    if isinstance(predicate, AndPredicate):
        return [
            "And",
            [to_turbopuffer_filters(clause, capabilities) for clause in predicate.clauses],
        ]
    if isinstance(predicate, OrPredicate):
        return [
            "Or",
            [to_turbopuffer_filters(clause, capabilities) for clause in predicate.clauses],
        ]
    if isinstance(predicate, NotPredicate):
        return ["Not", to_turbopuffer_filters(predicate.clause, capabilities)]

    backend = str(capabilities.get("backend", "unknown"))
    raise InvalidFilterError(
        backend=backend,
        message=f"unexpected predicate type '{type(predicate).__name__}'",
        predicate=predicate,
    )
