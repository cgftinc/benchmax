"""Tests for Pinecone filter mapper."""

from __future__ import annotations

import pytest

from benchmax.rag.corpus.pinecone.filter_mapper import _negate, to_pinecone_filters
from benchmax.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from benchmax.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)

_CAPS: SearchCapabilities = {
    "backend": "pinecone",
    "modes": {"vector"},
    "filter_ops": {
        "field": {"eq", "in", "gte", "lte"},
        "logical": {"and", "or", "not"},
    },
    "ranking": {"cosine"},
    "constraints": {},
    "graph_expansion": False,
}


class TestFieldPredicates:
    def test_eq(self):
        pred = FieldPredicate(field="status", op="eq", value="active")
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {"status": {"$eq": "active"}}

    def test_in(self):
        pred = FieldPredicate(field="category", op="in", value=["a", "b"])
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {"category": {"$in": ["a", "b"]}}

    def test_gte(self):
        pred = FieldPredicate(field="score", op="gte", value=0.5)
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {"score": {"$gte": 0.5}}

    def test_lte(self):
        pred = FieldPredicate(field="score", op="lte", value=100)
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {"score": {"$lte": 100}}


class TestLogicalPredicates:
    def test_and(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                FieldPredicate(field="b", op="eq", value=2),
            )
        )
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {
            "$and": [
                {"a": {"$eq": 1}},
                {"b": {"$eq": 2}},
            ]
        }

    def test_or(self):
        pred = OrPredicate(
            clauses=(
                FieldPredicate(field="x", op="eq", value="foo"),
                FieldPredicate(field="y", op="eq", value="bar"),
            )
        )
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {
            "$or": [
                {"x": {"$eq": "foo"}},
                {"y": {"$eq": "bar"}},
            ]
        }

    def test_not_field(self):
        """Not of eq → $ne."""
        pred = NotPredicate(clause=FieldPredicate(field="status", op="eq", value="deleted"))
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {"status": {"$ne": "deleted"}}

    def test_not_and_becomes_or(self):
        """De Morgan: NOT(a AND b) → (NOT a) OR (NOT b)."""
        pred = NotPredicate(
            clause=AndPredicate(
                clauses=(
                    FieldPredicate(field="a", op="eq", value=1),
                    FieldPredicate(field="b", op="eq", value=2),
                )
            )
        )
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {
            "$or": [
                {"a": {"$ne": 1}},
                {"b": {"$ne": 2}},
            ]
        }


class TestNone:
    def test_none_returns_none(self):
        assert to_pinecone_filters(None, _CAPS) is None


class TestNested:
    def test_nested_and_or(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="type", op="eq", value="doc"),
                OrPredicate(
                    clauses=(
                        FieldPredicate(field="lang", op="eq", value="en"),
                        FieldPredicate(field="lang", op="eq", value="fr"),
                    )
                ),
            )
        )
        result = to_pinecone_filters(pred, _CAPS)
        assert result == {
            "$and": [
                {"type": {"$eq": "doc"}},
                {
                    "$or": [
                        {"lang": {"$eq": "en"}},
                        {"lang": {"$eq": "fr"}},
                    ]
                },
            ]
        }


class TestUnsupported:
    def test_unsupported_field_op(self):
        # "contains_any" not in our capabilities
        caps: SearchCapabilities = {
            "backend": "pinecone",
            "modes": {"vector"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"cosine"},
            "constraints": {},
            "graph_expansion": False,
        }
        pred = FieldPredicate(field="tags", op="contains_any", value=["a"])
        with pytest.raises(UnsupportedFilterError):
            to_pinecone_filters(pred, caps)

    def test_empty_field_name(self):
        pred = FieldPredicate(field="", op="eq", value="x")
        with pytest.raises(InvalidFilterError):
            to_pinecone_filters(pred, _CAPS)


class TestNegateHelper:
    def test_negate_eq(self):
        assert _negate({"f": {"$eq": 1}}) == {"f": {"$ne": 1}}

    def test_negate_in(self):
        assert _negate({"f": {"$in": [1, 2]}}) == {"f": {"$nin": [1, 2]}}

    def test_negate_and(self):
        result = _negate({"$and": [{"a": {"$eq": 1}}]})
        assert result == {"$or": [{"a": {"$ne": 1}}]}
