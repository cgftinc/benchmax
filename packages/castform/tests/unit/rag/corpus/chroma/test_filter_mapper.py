"""Tests for Chroma filter mapper — predicate AST → Chroma filter dicts."""

from __future__ import annotations

import pytest
from castform.rag.corpus.chroma.filter_mapper import to_chroma_filters
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from castform.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)

_CAPS: SearchCapabilities = {
    "backend": "chroma",
    "modes": {"vector"},
    "filter_ops": {
        "field": {"eq", "in", "gte", "lte"},
        "logical": {"and", "or", "not"},
    },
    "ranking": {"cosine"},
    "constraints": {},
    "graph_expansion": False,
}


# ---------------------------------------------------------------------------
# Field predicates → {"field": {"$op": value}}
# ---------------------------------------------------------------------------


class TestFieldPredicates:
    def test_eq(self):
        pred = FieldPredicate(field="color", op="eq", value="red")
        assert to_chroma_filters(pred, _CAPS) == {"color": {"$eq": "red"}}

    def test_in(self):
        pred = FieldPredicate(field="status", op="in", value=["a", "b"])
        assert to_chroma_filters(pred, _CAPS) == {"status": {"$in": ["a", "b"]}}

    def test_gte(self):
        pred = FieldPredicate(field="score", op="gte", value=5)
        assert to_chroma_filters(pred, _CAPS) == {"score": {"$gte": 5}}

    def test_lte(self):
        pred = FieldPredicate(field="score", op="lte", value=10)
        assert to_chroma_filters(pred, _CAPS) == {"score": {"$lte": 10}}


# ---------------------------------------------------------------------------
# Logical predicates → {"$and"/"$or": [...]}
# ---------------------------------------------------------------------------


class TestLogicalPredicates:
    def test_and(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                FieldPredicate(field="b", op="eq", value=2),
            )
        )
        assert to_chroma_filters(pred, _CAPS) == {"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}

    def test_or(self):
        pred = OrPredicate(
            clauses=(
                FieldPredicate(field="x", op="eq", value="yes"),
                FieldPredicate(field="y", op="eq", value="no"),
            )
        )
        assert to_chroma_filters(pred, _CAPS) == {
            "$or": [{"x": {"$eq": "yes"}}, {"y": {"$eq": "no"}}]
        }

    def test_not_eq_emulated_as_ne(self):
        """Chroma has no native $not — NOT(eq) → $ne."""
        pred = NotPredicate(clause=FieldPredicate(field="status", op="eq", value="deleted"))
        assert to_chroma_filters(pred, _CAPS) == {"status": {"$ne": "deleted"}}

    def test_not_non_eq_raises(self):
        """NOT on non-eq predicates is unsupported."""
        pred = NotPredicate(clause=FieldPredicate(field="score", op="gte", value=5))
        with pytest.raises(UnsupportedFilterError, match="\\$not"):
            to_chroma_filters(pred, _CAPS)


# ---------------------------------------------------------------------------
# Nested predicates
# ---------------------------------------------------------------------------


class TestNestedPredicates:
    def test_and_with_or(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                OrPredicate(
                    clauses=(
                        FieldPredicate(field="b", op="eq", value=2),
                        FieldPredicate(field="c", op="eq", value=3),
                    )
                ),
            )
        )
        assert to_chroma_filters(pred, _CAPS) == {
            "$and": [
                {"a": {"$eq": 1}},
                {"$or": [{"b": {"$eq": 2}}, {"c": {"$eq": 3}}]},
            ]
        }

    def test_deeply_nested(self):
        pred = OrPredicate(
            clauses=(
                AndPredicate(
                    clauses=(
                        FieldPredicate(field="x", op="eq", value=1),
                        FieldPredicate(field="y", op="gte", value=10),
                    )
                ),
                FieldPredicate(field="z", op="in", value=["a", "b"]),
            )
        )
        result = to_chroma_filters(pred, _CAPS)
        assert "$or" in result
        assert len(result["$or"]) == 2


# ---------------------------------------------------------------------------
# None / empty / unsupported
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_returns_none(self):
        assert to_chroma_filters(None, _CAPS) is None

    def test_unsupported_field_op(self):
        caps_limited: SearchCapabilities = {
            "backend": "chroma",
            "modes": {"vector"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"cosine"},
            "constraints": {},
            "graph_expansion": False,
        }
        pred = FieldPredicate(field="x", op="gte", value=5)
        with pytest.raises(UnsupportedFilterError):
            to_chroma_filters(pred, caps_limited)

    def test_empty_and_raises(self):
        pred = AndPredicate(clauses=())
        with pytest.raises(InvalidFilterError, match="at least one clause"):
            to_chroma_filters(pred, _CAPS)

    def test_empty_or_raises(self):
        pred = OrPredicate(clauses=())
        with pytest.raises(InvalidFilterError, match="at least one clause"):
            to_chroma_filters(pred, _CAPS)

    def test_empty_field_name_raises(self):
        pred = FieldPredicate(field="", op="eq", value="x")
        with pytest.raises(InvalidFilterError, match="non-empty"):
            to_chroma_filters(pred, _CAPS)
