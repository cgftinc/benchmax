"""Unit tests for Neon predicate AST to parameterized SQL translation."""

from __future__ import annotations

import re

import pytest

from castform.rag.corpus.neon.filter_mapper import _ensure_supported, to_neon_filters
from castform.rag.corpus.search_schema.builders import all_of, any_of, f, not_
from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from castform.rag.corpus.search_schema.search_types import (
    FieldPredicate,
    SearchCapabilities,
)

_CAPABILITIES: SearchCapabilities = {
    "backend": "neon-test",
    "modes": {"lexical"},
    "filter_ops": {
        "field": {
            "eq",
            "ne",
            "in",
            "gt",
            "gte",
            "lt",
            "lte",
            "contains_any",
            "contains_all",
        },
        "logical": {"and", "or", "not"},
    },
    "ranking": set(),
    "constraints": {},
    "graph_expansion": False,
}


@pytest.mark.parametrize(
    ("predicate", "expected_sql", "expected_params"),
    [
        (
            f("year").eq(2026),
            "metadata @> jsonb_build_object(%(k0)s, to_jsonb(%(v0)s::numeric))",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("year").ne(2026),
            "(metadata @> jsonb_build_object(%(k0)s, to_jsonb(%(v0)s::numeric))) "
            "IS NOT TRUE",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("year").in_([2025, 2026]),
            "(metadata @> jsonb_build_object(%(k0)s, to_jsonb(%(v0)s::numeric)) OR "
            "metadata @> jsonb_build_object(%(k0)s, to_jsonb(%(v1)s::numeric)))",
            {"k0": "year", "v0": 2025, "v1": 2026},
        ),
        (
            f("year").gt(2026),
            "CASE WHEN jsonb_typeof(metadata -> %(k0)s) = 'number' "
            "THEN (metadata ->> %(k0)s)::numeric > %(v0)s::numeric ELSE NULL END",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("year").gte(2026),
            "CASE WHEN jsonb_typeof(metadata -> %(k0)s) = 'number' "
            "THEN (metadata ->> %(k0)s)::numeric >= %(v0)s::numeric ELSE NULL END",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("year").lt(2026),
            "CASE WHEN jsonb_typeof(metadata -> %(k0)s) = 'number' "
            "THEN (metadata ->> %(k0)s)::numeric < %(v0)s::numeric ELSE NULL END",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("year").lte(2026),
            "CASE WHEN jsonb_typeof(metadata -> %(k0)s) = 'number' "
            "THEN (metadata ->> %(k0)s)::numeric <= %(v0)s::numeric ELSE NULL END",
            {"k0": "year", "v0": 2026},
        ),
        (
            f("tags").contains_any(["rag", "sql"]),
            "(metadata @> jsonb_build_object(%(k0)s, "
            "jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
            "metadata @> jsonb_build_object(%(k0)s, "
            "jsonb_build_array(to_jsonb(%(v1)s::text))))",
            {"k0": "tags", "v0": "rag", "v1": "sql"},
        ),
        (
            f("tags").contains_all(["rag", "sql"]),
            "metadata @> jsonb_build_object(%(k0)s, "
            "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))",
            {"k0": "tags", "v0": "rag", "v1": "sql"},
        ),
    ],
)
def test_all_nine_operators(
    predicate: FieldPredicate,
    expected_sql: str,
    expected_params: dict[str, object],
) -> None:
    assert to_neon_filters(predicate) == (expected_sql, expected_params)


def test_builders_expose_new_operators() -> None:
    assert f("x").ne(1) == FieldPredicate(field="x", op="ne", value=1)
    assert f("x").gt(1) == FieldPredicate(field="x", op="gt", value=1)
    assert f("x").lt(1) == FieldPredicate(field="x", op="lt", value=1)


@pytest.mark.parametrize(
    ("value", "cast"),
    [("5", "text"), (5, "numeric"), (True, "boolean")],
)
def test_scalar_typing_selects_json_compatible_cast(value: object, cast: str) -> None:
    where_sql, params = to_neon_filters(f("typed").eq(value))

    assert f"%(v0)s::{cast}" in where_sql
    assert params == {"k0": "typed", "v0": value}


@pytest.mark.parametrize(
    ("values", "cast"),
    [([1, 2], "numeric"), ([True, False], "boolean")],
)
def test_contains_uses_typed_array_containment(values: list[object], cast: str) -> None:
    where_sql, _ = to_neon_filters(f("tags").contains_any(values))

    assert where_sql.count("metadata @>") == 2
    assert where_sql.count(f"::{cast}") == 2
    assert "?|" not in where_sql
    assert "?&" not in where_sql


def test_nested_and_or_not_uses_three_valued_leaf_for_negation() -> None:
    predicate = all_of(
        f("kind").eq("document"),
        any_of(f("year").gt(2020), not_(f("status").eq("deleted"))),
    )

    where_sql, params = to_neon_filters(predicate)

    assert " AND " in where_sql
    assert " OR " in where_sql
    assert "NOT (CASE WHEN NOT jsonb_exists(metadata, %(k2)s) THEN NULL" in where_sql
    assert "jsonb_typeof(metadata -> %(k2)s) <> 'string' THEN NULL" in where_sql
    assert params == {
        "k0": "kind",
        "v0": "document",
        "k1": "year",
        "v1": 2020,
        "k2": "status",
        "v2": "deleted",
    }


@pytest.mark.parametrize(
    ("predicate", "stored_type"),
    [
        (f("value").eq(1), "number"),
        (f("value").in_([1, 2]), "number"),
        (f("value").contains_any(["a"]), "array"),
        (f("value").contains_all(["a"]), "array"),
    ],
)
def test_negated_containment_is_case_guarded(
    predicate: FieldPredicate,
    stored_type: str,
) -> None:
    where_sql, _ = to_neon_filters(not_(predicate))

    assert where_sql.startswith(
        "NOT (CASE WHEN NOT jsonb_exists(metadata, %(k0)s) THEN NULL"
    )
    assert f"jsonb_typeof(metadata -> %(k0)s) <> '{stored_type}' THEN NULL" in where_sql


def test_ensure_supported_rejects_unsupported_field_operator() -> None:
    capabilities: SearchCapabilities = {
        **_CAPABILITIES,
        "filter_ops": {"field": {"eq"}, "logical": {"and", "or", "not"}},
    }

    with pytest.raises(UnsupportedFilterError, match="operator 'gt'"):
        _ensure_supported(capabilities, f("year").gt(2020))


def test_ensure_supported_rejects_unsupported_logical_operator() -> None:
    capabilities: SearchCapabilities = {
        **_CAPABILITIES,
        "filter_ops": {"field": {"eq"}, "logical": set()},
    }

    with pytest.raises(UnsupportedFilterError, match="operator 'not'"):
        _ensure_supported(capabilities, not_(f("status").eq("deleted")))


def test_contains_all_empty_requires_a_present_array() -> None:
    where_sql, params = to_neon_filters(f("tags").contains_all([]))

    assert where_sql == ("metadata @> jsonb_build_object(%(k0)s, jsonb_build_array())")
    assert params == {"k0": "tags"}


@pytest.mark.parametrize("operator", ["in", "contains_any"])
def test_empty_disjunction_is_false(operator: str) -> None:
    predicate = FieldPredicate(field="tags", op=operator, value=[])
    assert to_neon_filters(predicate) == ("FALSE", {"k0": "tags"})


@pytest.mark.parametrize(
    ("operator", "expected_sql"),
    [
        (
            "in",
            "NOT (CASE WHEN NOT jsonb_exists(metadata, %(k0)s) THEN NULL "
            "WHEN jsonb_typeof(metadata -> %(k0)s) = 'null' THEN NULL "
            "ELSE FALSE END)",
        ),
        (
            "contains_any",
            "NOT (CASE WHEN NOT jsonb_exists(metadata, %(k0)s) THEN NULL "
            "WHEN jsonb_typeof(metadata -> %(k0)s) <> 'array' THEN NULL "
            "ELSE FALSE END)",
        ),
    ],
)
def test_negated_empty_list_wrong_type_asymmetry(
    operator: str,
    expected_sql: str,
) -> None:
    # An empty `in` has no operand type; `contains_any` still requires an array.
    predicate = not_(FieldPredicate(field="tags", op=operator, value=[]))

    assert to_neon_filters(predicate) == (expected_sql, {"k0": "tags"})


@pytest.mark.parametrize(
    "predicate",
    [
        f("year").gt(True),
        f("year").in_([1, "2"]),
        f("tags").contains_all([True, 1]),
        FieldPredicate(field="tags", op="contains_any", value="rag"),
    ],
)
def test_invalid_value_shapes_are_rejected(predicate: FieldPredicate) -> None:
    with pytest.raises(InvalidFilterError):
        to_neon_filters(predicate)


def test_keys_and_values_are_bound_during_injection_attempt() -> None:
    malicious_key = "x') OR TRUE --"
    malicious_value = "v'); DROP TABLE neon_corpus_versions; --"

    where_sql, params = to_neon_filters(f(malicious_key).eq(malicious_value))

    assert malicious_key not in where_sql
    assert malicious_value not in where_sql
    assert params == {"k0": malicious_key, "v0": malicious_value}
    assert set(re.findall(r"%\(([^)]+)\)s", where_sql)) == set(params)
    assert '"x' not in where_sql


def test_none_is_an_unfiltered_where_expression() -> None:
    assert to_neon_filters(None) == ("TRUE", {})
