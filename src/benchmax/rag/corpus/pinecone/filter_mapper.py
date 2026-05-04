"""Pinecone backend filter translation from shared predicate AST.

Pinecone filter syntax:
  - Field:   {"field_name": {"$eq": value}}
  - And:     {"$and": [clause, ...]}
  - Or:      {"$or": [clause, ...]}
  - Not:     negate via complement operators (Pinecone has no $not at top level,
             so we push negation into the field operator where possible)
"""

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

_PINECONE_FIELD_OP_MAP: dict[str, str] = {
    "eq": "$eq",
    "in": "$in",
    "gte": "$gte",
    "lte": "$lte",
}


def _ensure_supported(capabilities: SearchCapabilities, predicate: FilterPredicate | None) -> None:
    """Validate that the predicate tree uses only supported operators."""
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


# Pinecone doesn't have a top-level $not. We negate field ops where possible.
_NEGATE_OP: dict[str, str] = {
    "$eq": "$ne",
    "$ne": "$eq",
    "$gt": "$lte",
    "$lte": "$gt",
    "$gte": "$lt",
    "$lt": "$gte",
    "$in": "$nin",
    "$nin": "$in",
}


def _negate(filter_dict: dict[str, Any]) -> dict[str, Any]:
    """Negate a Pinecone filter expression.

    For field predicates we flip the operator.  For $and/$or we apply
    De Morgan's law.
    """
    if "$and" in filter_dict:
        return {"$or": [_negate(c) for c in filter_dict["$and"]]}
    if "$or" in filter_dict:
        return {"$and": [_negate(c) for c in filter_dict["$or"]]}

    # Field predicate: {"field": {"$op": val}}
    for field, ops in filter_dict.items():
        if isinstance(ops, dict):
            negated_ops = {}
            for op, val in ops.items():
                neg_op = _NEGATE_OP.get(op)
                if neg_op is None:
                    raise UnsupportedFilterError(
                        backend="pinecone",
                        message=f"cannot negate operator '{op}'",
                        predicate=None,
                    )
                negated_ops[neg_op] = val
            return {field: negated_ops}

    raise InvalidFilterError(
        backend="pinecone",
        message="cannot negate filter expression",
        predicate=filter_dict,
    )


def to_pinecone_filters(
    predicate: FilterPredicate | None,
    capabilities: SearchCapabilities,
) -> dict[str, Any] | None:
    """Translate predicate AST into a Pinecone filter dict."""
    _ensure_supported(capabilities, predicate)
    if predicate is None:
        return None

    if isinstance(predicate, FieldPredicate):
        op = _PINECONE_FIELD_OP_MAP.get(predicate.op)
        if op is None:
            raise UnsupportedFilterError(
                backend=str(capabilities.get("backend", "unknown")),
                message=f"field operator '{predicate.op}' has no Pinecone mapping",
                predicate=predicate,
            )
        return {predicate.field: {op: predicate.value}}

    if isinstance(predicate, AndPredicate):
        return {"$and": [to_pinecone_filters(clause, capabilities) for clause in predicate.clauses]}

    if isinstance(predicate, OrPredicate):
        return {"$or": [to_pinecone_filters(clause, capabilities) for clause in predicate.clauses]}

    if isinstance(predicate, NotPredicate):
        inner = to_pinecone_filters(predicate.clause, capabilities)
        if inner is None:
            return None
        return _negate(inner)

    backend = str(capabilities.get("backend", "unknown"))
    raise InvalidFilterError(
        backend=backend,
        message=f"unexpected predicate type '{type(predicate).__name__}'",
        predicate=predicate,
    )
