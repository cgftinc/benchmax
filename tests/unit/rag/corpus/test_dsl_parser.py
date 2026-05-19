"""Tests for dsl_parser: JSON → predicate AST conversion."""

from __future__ import annotations

from benchmax.rag.corpus.postgres.filter_mapper import to_corpora_filters
from benchmax.rag.corpus.search_schema.builders import all_of, f, not_
from benchmax.rag.corpus.search_schema.dsl_parser import dsl_to_predicate
from benchmax.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)


class TestFieldPredicate:
    def test_basic(self):
        result = dsl_to_predicate({"field": "x", "op": "eq", "value": 1})
        assert isinstance(result, FieldPredicate)
        assert result.field == "x"
        assert result.op == "eq"
        assert result.value == 1

    def test_value_is_list(self):
        result = dsl_to_predicate({"field": "tags", "op": "in", "value": ["a", "b"]})
        assert isinstance(result, FieldPredicate)
        assert result.value == ["a", "b"]


class TestLogicalPredicates:
    def test_and(self):
        result = dsl_to_predicate({"and": [{"field": "a", "op": "eq", "value": 1}]})
        assert isinstance(result, AndPredicate)
        assert len(result.clauses) == 1

    def test_or(self):
        result = dsl_to_predicate(
            {
                "or": [
                    {"field": "x", "op": "eq", "value": "foo"},
                    {"field": "y", "op": "eq", "value": "bar"},
                ]
            }
        )
        assert isinstance(result, OrPredicate)
        assert len(result.clauses) == 2

    def test_not(self):
        result = dsl_to_predicate({"not": {"field": "x", "op": "eq", "value": "deleted"}})
        assert isinstance(result, NotPredicate)
        assert isinstance(result.clause, FieldPredicate)

    def test_nested_compound(self):
        result = dsl_to_predicate(
            {
                "and": [
                    {"field": "type", "op": "eq", "value": "doc"},
                    {
                        "or": [
                            {"field": "lang", "op": "eq", "value": "en"},
                            {"field": "lang", "op": "eq", "value": "fr"},
                        ]
                    },
                ]
            }
        )
        assert isinstance(result, AndPredicate)
        assert isinstance(result.clauses[1], OrPredicate)


class TestEdgeCases:
    def test_none_input(self):
        assert dsl_to_predicate(None) is None

    def test_non_dict_input(self):
        assert dsl_to_predicate("not a dict") is None
        assert dsl_to_predicate(42) is None

    def test_empty_field_name(self):
        assert dsl_to_predicate({"field": "", "op": "eq", "value": 1}) is None

    def test_empty_op(self):
        assert dsl_to_predicate({"field": "x", "op": "", "value": 1}) is None

    def test_empty_and_clauses(self):
        assert dsl_to_predicate({"and": []}) is None

    def test_not_with_none_inner(self):
        assert dsl_to_predicate({"not": None}) is None

    def test_and_filters_none_children(self):
        result = dsl_to_predicate({"and": [None, {"field": "x", "op": "eq", "value": 1}]})
        assert isinstance(result, AndPredicate)
        assert len(result.clauses) == 1
        assert result.clauses[0].field == "x"


class TestRoundTrip:
    def test_builder_to_corpora_to_parser(self):
        """Build predicate → serialize to Postgres DSL → parse back."""
        original = all_of(
            f("file").eq("guide.md"),
            not_(f("index").gte(5)),
        )
        caps: SearchCapabilities = {
            "backend": "corpora",
            "modes": {"lexical"},
            "filter_ops": {
                "field": {"eq", "gte"},
                "logical": {"and", "not"},
            },
            "ranking": {"bm25"},
            "constraints": {},
            "graph_expansion": False,
        }
        serialized = to_corpora_filters(original, caps)
        parsed = dsl_to_predicate(serialized)

        assert isinstance(parsed, AndPredicate)
        assert len(parsed.clauses) == 2
        # First clause: field eq
        assert parsed.clauses[0] == FieldPredicate(field="file", op="eq", value="guide.md")
        # Second clause: not(field gte)
        assert isinstance(parsed.clauses[1], NotPredicate)
        assert parsed.clauses[1].clause == FieldPredicate(field="index", op="gte", value=5)
