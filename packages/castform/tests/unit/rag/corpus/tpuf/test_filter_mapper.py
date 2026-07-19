"""Tests for Turbopuffer filter mapper: AST → Turbopuffer format."""

from __future__ import annotations

import pytest

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
from castform.rag.corpus.turbopuffer.filter_mapper import to_turbopuffer_filters

_TPUF_CAPS: SearchCapabilities = {
    "backend": "turbopuffer",
    "modes": {"lexical"},
    "filter_ops": {
        "field": {"eq", "in", "gte", "lte"},
        "logical": {"and", "or", "not"},
    },
    "ranking": {"bm25"},
    "constraints": {},
    "graph_expansion": False,
}


class TestNone:
    def test_none_returns_none(self):
        assert to_turbopuffer_filters(None, _TPUF_CAPS) is None


class TestFieldPredicates:
    def test_eq(self):
        pred = FieldPredicate(field="status", op="eq", value="active")
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["status", "Eq", "active"]

    def test_in(self):
        pred = FieldPredicate(field="tag", op="in", value=["a", "b"])
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["tag", "In", ["a", "b"]]

    def test_gte(self):
        pred = FieldPredicate(field="score", op="gte", value=0.5)
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["score", "Gte", 0.5]

    def test_lte(self):
        pred = FieldPredicate(field="count", op="lte", value=100)
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["count", "Lte", 100]


class TestLogicalPredicates:
    def test_and(self):
        pred = AndPredicate(
            clauses=(
                FieldPredicate(field="a", op="eq", value=1),
                FieldPredicate(field="b", op="eq", value=2),
            )
        )
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["And", [["a", "Eq", 1], ["b", "Eq", 2]]]

    def test_or(self):
        pred = OrPredicate(
            clauses=(
                FieldPredicate(field="x", op="eq", value="foo"),
                FieldPredicate(field="y", op="eq", value="bar"),
            )
        )
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["Or", [["x", "Eq", "foo"], ["y", "Eq", "bar"]]]

    def test_not(self):
        pred = NotPredicate(clause=FieldPredicate(field="z", op="eq", value="deleted"))
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == ["Not", ["z", "Eq", "deleted"]]

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
        result = to_turbopuffer_filters(pred, _TPUF_CAPS)
        assert result == [
            "And",
            [
                ["type", "Eq", "doc"],
                ["Or", [["lang", "Eq", "en"], ["lang", "Eq", "fr"]]],
            ],
        ]


class TestUnsupported:
    def test_unsupported_field_op(self):
        pred = FieldPredicate(field="tags", op="contains_any", value=["a"])
        with pytest.raises(UnsupportedFilterError):
            to_turbopuffer_filters(pred, _TPUF_CAPS)

    def test_empty_field_name(self):
        pred = FieldPredicate(field="", op="eq", value="x")
        with pytest.raises(InvalidFilterError):
            to_turbopuffer_filters(pred, _TPUF_CAPS)

    def test_empty_and_clauses(self):
        pred = AndPredicate(clauses=())
        with pytest.raises(InvalidFilterError):
            to_turbopuffer_filters(pred, _TPUF_CAPS)
