"""Chroma backend filter translation from shared predicate AST."""

from __future__ import annotations

from typing import Any

from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from castform.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    FilterPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)

_CHROMA_FIELD_OP_MAP: dict[str, str] = {
    "eq": "$eq",
    "in": "$in",
    "gte": "$gte",
    "lte": "$lte",
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
        # Chroma has no native $not — only supported for simple field eq → $ne
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


def to_chroma_filters(
    predicate: FilterPredicate | None,
    capabilities: SearchCapabilities,
) -> dict[str, Any] | None:
    """Translate predicate AST into a Chroma ``where`` filter dict."""
    _ensure_supported(capabilities, predicate)
    if predicate is None:
        return None

    backend = str(capabilities.get("backend", "unknown"))

    if isinstance(predicate, FieldPredicate):
        op = _CHROMA_FIELD_OP_MAP.get(predicate.op)
        if op is None:
            raise UnsupportedFilterError(
                backend=backend,
                message=f"field operator '{predicate.op}' has no Chroma mapping",
                predicate=predicate,
            )
        return {predicate.field: {op: predicate.value}}

    if isinstance(predicate, AndPredicate):
        return {"$and": [to_chroma_filters(c, capabilities) for c in predicate.clauses]}

    if isinstance(predicate, OrPredicate):
        return {"$or": [to_chroma_filters(c, capabilities) for c in predicate.clauses]}

    if isinstance(predicate, NotPredicate):
        # Chroma lacks $not. We emulate for the single-field eq case → $ne.
        inner = predicate.clause
        if isinstance(inner, FieldPredicate) and inner.op == "eq":
            return {inner.field: {"$ne": inner.value}}
        raise UnsupportedFilterError(
            backend=backend,
            message=(
                "Chroma does not support $not; only Not(FieldPredicate(op='eq')) "
                "is emulated via $ne"
            ),
            predicate=predicate,
        )

    raise InvalidFilterError(
        backend=backend,
        message=f"unexpected predicate type '{type(predicate).__name__}'",
        predicate=predicate,
    )
