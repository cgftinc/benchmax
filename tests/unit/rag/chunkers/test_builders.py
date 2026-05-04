"""Tests for filter predicate builder DSL."""

from __future__ import annotations

from benchmax.rag.corpus.search_schema.builders import all_of, any_of, f, not_
from benchmax.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
)


class TestFieldRef:
    def test_eq(self):
        pred = f("status").eq("active")
        assert pred == FieldPredicate(field="status", op="eq", value="active")

    def test_in(self):
        pred = f("color").in_(["red", "blue"])
        assert pred == FieldPredicate(field="color", op="in", value=["red", "blue"])

    def test_gte(self):
        pred = f("score").gte(0.5)
        assert pred == FieldPredicate(field="score", op="gte", value=0.5)

    def test_lte(self):
        pred = f("count").lte(100)
        assert pred == FieldPredicate(field="count", op="lte", value=100)

    def test_contains_any(self):
        pred = f("tags").contains_any(["python", "rust"])
        assert pred == FieldPredicate(field="tags", op="contains_any", value=["python", "rust"])

    def test_contains_all(self):
        pred = f("tags").contains_all(["a", "b"])
        assert pred == FieldPredicate(field="tags", op="contains_all", value=["a", "b"])


class TestLogicalCombinators:
    def test_all_of(self):
        a = f("x").eq(1)
        b = f("y").eq(2)
        pred = all_of(a, b)
        assert isinstance(pred, AndPredicate)
        assert pred.clauses == (a, b)

    def test_any_of(self):
        a = f("x").eq(1)
        b = f("y").eq(2)
        pred = any_of(a, b)
        assert isinstance(pred, OrPredicate)
        assert pred.clauses == (a, b)

    def test_not(self):
        a = f("x").eq(1)
        pred = not_(a)
        assert isinstance(pred, NotPredicate)
        assert pred.clause == a


class TestNested:
    def test_nested_compound(self):
        pred = all_of(
            f("x").eq(1),
            not_(f("y").gte(2)),
        )
        assert isinstance(pred, AndPredicate)
        assert len(pred.clauses) == 2
        assert isinstance(pred.clauses[0], FieldPredicate)
        assert isinstance(pred.clauses[1], NotPredicate)
        inner = pred.clauses[1].clause
        assert isinstance(inner, FieldPredicate)
        assert inner.op == "gte"
        assert inner.value == 2
