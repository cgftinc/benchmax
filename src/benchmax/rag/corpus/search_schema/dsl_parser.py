"""Parsing helpers for JSON-style filter objects into predicate AST."""

from __future__ import annotations

from typing import Any

from .search_types import AndPredicate, FieldPredicate, FilterPredicate, NotPredicate, OrPredicate


def dsl_to_predicate(node: Any) -> FilterPredicate | None:
    """Convert a JSON-style filter object to a predicate AST.

    Supported shapes:
    - {"field": "<name>", "op": "<op>", "value": ...}
    - {"and": [<clause>, ...]}
    - {"or": [<clause>, ...]}
    - {"not": <clause>}
    """
    if node is None or not isinstance(node, dict):
        return None

    if "field" in node and "op" in node:
        field_name = str(node.get("field", "")).strip()
        op = str(node.get("op", "")).strip()
        if not field_name or not op:
            return None
        return FieldPredicate(field=field_name, op=op, value=node.get("value"))

    if "and" in node and isinstance(node["and"], list):
        clauses = [
            pred
            for pred in (dsl_to_predicate(clause) for clause in node["and"])
            if pred is not None
        ]
        return AndPredicate(clauses=tuple(clauses)) if clauses else None

    if "or" in node and isinstance(node["or"], list):
        clauses = [
            pred for pred in (dsl_to_predicate(clause) for clause in node["or"]) if pred is not None
        ]
        return OrPredicate(clauses=tuple(clauses)) if clauses else None

    if "not" in node:
        clause = dsl_to_predicate(node["not"])
        return NotPredicate(clause=clause) if clause is not None else None

    return None
