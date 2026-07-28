"""Tests for Postgres filter mapper: AST → Postgres DSL format."""

from __future__ import annotations

import pytest
from castform.rag.corpus.postgres.filter_mapper import to_corpora_filters
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

_CORPORA_CAPS: SearchCapabilities = {
    "backend": "corpora",
    "modes": {"lexical"},
    "filter_ops": {
        "field": {"eq", "in", "gte", "lte", "contains_any", "contains_all"},
        "logical": {"and", "or", "not"},
    },
    "ranking": {"bm25"},
    "constraints": {},
    "graph_expansion": True,
}


class TestNone:
    def test_none_returns_none(self):
        assert to_corpora_filters(None, _CORPORA_CAPS) is None


class TestFieldPredicates:
    def test_eq(self):
        pred = FieldPredicate(field="status", op="eq", value="active")
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {"field": "status", "op": "eq", "value": "active"}

    def test_contains_any(self):
        pred = FieldPredicate(field="tags", op="contains_any", value=["a", "b"])
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {
            "field": "tags",
            "op": "contains_any",
            "value": ["a", "b"],
        }

    def test_contains_all(self):
        pred = FieldPredicate(field="tags", op="contains_all", value=["x", "y"])
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {
            "field": "tags",
            "op": "contains_all",
            "value": ["x", "y"],
        }


class TestLogicalPredicates:
    def test_and(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                FieldPredicate(field="b", op="eq", value=2),
            )
        )
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {
            "and": [
                {"field": "a", "op": "eq", "value": 1},
                {"field": "b", "op": "eq", "value": 2},
            ]
        }

    def test_or(self):
        pred = OrPredicate(
            clauses=(
                FieldPredicate(field="x", op="eq", value="foo"),
                FieldPredicate(field="y", op="eq", value="bar"),
            )
        )
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {
            "or": [
                {"field": "x", "op": "eq", "value": "foo"},
                {"field": "y", "op": "eq", "value": "bar"},
            ]
        }

    def test_not(self):
        pred = NotPredicate(clause=FieldPredicate(field="z", op="eq", value="deleted"))
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {"not": {"field": "z", "op": "eq", "value": "deleted"}}

    def test_nested(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="type", op="eq", value="doc"),
                NotPredicate(clause=FieldPredicate(field="status", op="eq", value="draft")),
            )
        )
        result = to_corpora_filters(pred, _CORPORA_CAPS)
        assert result == {
            "and": [
                {"field": "type", "op": "eq", "value": "doc"},
                {"not": {"field": "status", "op": "eq", "value": "draft"}},
            ]
        }


class TestUnsupported:
    def test_restricted_caps_rejects_unsupported_op(self):
        restricted: SearchCapabilities = {
            "backend": "corpora",
            "modes": {"lexical"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }
        pred = FieldPredicate(field="x", op="gte", value=5)
        with pytest.raises(UnsupportedFilterError):
            to_corpora_filters(pred, restricted)

    def test_empty_field_name(self):
        pred = FieldPredicate(field="  ", op="eq", value="x")
        with pytest.raises(InvalidFilterError):
            to_corpora_filters(pred, _CORPORA_CAPS)

    def test_logical_not_supported(self):
        restricted: SearchCapabilities = {
            "backend": "corpora",
            "modes": {"lexical"},
            "filter_ops": {"field": {"eq"}, "logical": set()},
            "ranking": {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }
        pred = AndPredicate(clauses=(FieldPredicate(field="x", op="eq", value=1),))
        with pytest.raises(UnsupportedFilterError):
            to_corpora_filters(pred, restricted)
