"""Contract #3: type-directed, indexable, correctly-negating filter truth table.

The frozen table (positive SQL, three-valued negated leaves, edge outcomes,
indexability, typed contains shapes) is asserted here. Slice 4 implements
``predicate_to_sql``: single-leaf emission that dispatches the cast on the value's
JSON type and one placeholder per list element, matching the frozen forms exactly.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.filter_mapper import (
    CONTAINS_ATOM_BY_TYPE,
    FILTER_TRUTH_TABLE_BY_OP,
    LIST_OPS,
    NEGATION_TEMPLATE,
    NEON_FIELD_OPERATORS,
    RANGE_OPS,
    predicate_to_sql,
)
from castform.rag.corpus.search_schema.search_exceptions import InvalidFilterError
from castform.rag.corpus.search_schema.search_types import FieldPredicate

EXPECTED_POSITIVE_SQL: dict[str, str] = {
    "eq": "metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))",
    "ne": (
        "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))) IS NOT TRUE"
    ),
    "in": (
        "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v0)s::numeric)) OR "
        "metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v1)s::numeric)))"
    ),
    "gt": (
        "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
        "THEN (metadata ->> %(k)s)::numeric > %(v)s::numeric ELSE NULL END"
    ),
    "gte": (
        "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
        "THEN (metadata ->> %(k)s)::numeric >= %(v)s::numeric ELSE NULL END"
    ),
    "lt": (
        "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
        "THEN (metadata ->> %(k)s)::numeric < %(v)s::numeric ELSE NULL END"
    ),
    "lte": (
        "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
        "THEN (metadata ->> %(k)s)::numeric <= %(v)s::numeric ELSE NULL END"
    ),
    "contains_any": (
        "(metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
        "metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v1)s::text))))"
    ),
    "contains_all": (
        "metadata @> jsonb_build_object(%(k)s::text, "
        "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
    ),
}

# op -> (missing_key, json_null, wrong_type, empty_operand) POSITIVE-leaf outcomes.
EXPECTED_EDGE_OUTCOMES: dict[str, tuple[str, str, str, str]] = {
    "eq": ("exclude", "exclude", "exclude", "na"),
    "ne": ("include", "include", "include", "na"),
    "in": ("exclude", "exclude", "exclude", "exclude"),
    "gt": ("exclude", "exclude", "exclude", "na"),
    "gte": ("exclude", "exclude", "exclude", "na"),
    "lt": ("exclude", "exclude", "exclude", "na"),
    "lte": ("exclude", "exclude", "exclude", "na"),
    "contains_any": ("exclude", "exclude", "exclude", "exclude"),
    "contains_all": ("exclude", "exclude", "exclude", "include"),
}

EXPECTED_INDEXABLE = {
    "eq": True,
    "ne": False,
    "in": True,
    "gt": False,
    "gte": False,
    "lt": False,
    "lte": False,
    "contains_any": True,
    "contains_all": True,
}

VALID_FIXTURES: dict[str, object] = {
    "eq": 2026,
    "ne": 2026,
    "in": [2025, 2026],
    "gt": 2026,
    "gte": 2026,
    "lt": 2026,
    "lte": 2026,
    "contains_any": ["a", "b"],
    "contains_all": ["a", "b"],
}


def test_nine_operators_frozen() -> None:
    assert set(NEON_FIELD_OPERATORS) == set(EXPECTED_POSITIVE_SQL)
    assert len(NEON_FIELD_OPERATORS) == 9
    assert {"ne", "gt", "lt"} <= set(NEON_FIELD_OPERATORS)
    assert RANGE_OPS == {"gt", "gte", "lt", "lte"}
    assert LIST_OPS == {"in", "contains_any", "contains_all"}


@pytest.mark.parametrize("op", list(EXPECTED_POSITIVE_SQL))
def test_positive_sql_frozen(op: str) -> None:
    assert FILTER_TRUTH_TABLE_BY_OP[op].positive_sql == EXPECTED_POSITIVE_SQL[op]


@pytest.mark.parametrize("op", list(EXPECTED_EDGE_OUTCOMES))
def test_edge_outcomes_frozen(op: str) -> None:
    o = FILTER_TRUTH_TABLE_BY_OP[op].outcomes
    assert (
        o["missing_key"],
        o["json_null"],
        o["wrong_type"],
        o["empty_operand"],
    ) == EXPECTED_EDGE_OUTCOMES[op]


@pytest.mark.parametrize("op", list(EXPECTED_INDEXABLE))
def test_indexability_frozen(op: str) -> None:
    assert FILTER_TRUTH_TABLE_BY_OP[op].indexable == EXPECTED_INDEXABLE[op]


@pytest.mark.parametrize("op", ["eq", "in", "contains_any", "contains_all"])
def test_negated_leaf_is_three_valued(op: str) -> None:
    # A NotPredicate wraps this; it must yield NULL (not FALSE) for
    # missing/null/wrong-type so NOT(NULL)=NULL keeps them excluded.
    neg = FILTER_TRUTH_TABLE_BY_OP[op].negated_leaf_sql
    assert neg.startswith("CASE WHEN NOT jsonb_exists(metadata, %(k)s) THEN NULL")
    assert "THEN NULL" in neg and neg.rstrip().endswith("END")
    # and it is NOT just the bare (two-valued) positive containment.
    assert neg != FILTER_TRUTH_TABLE_BY_OP[op].positive_sql


@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
def test_range_cast_is_inside_case_not_bare_and(op: str) -> None:
    spec = FILTER_TRUTH_TABLE_BY_OP[op]
    assert spec.positive_sql.startswith(
        "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number'"
    )
    assert "ELSE NULL END" in spec.positive_sql
    # the unsafe `guard AND (...)::numeric` form must NOT appear.
    assert "' AND (metadata ->>" not in spec.positive_sql


def test_typed_contains_shapes_frozen() -> None:
    # numeric and boolean contains shapes are frozen explicitly, not only text.
    assert CONTAINS_ATOM_BY_TYPE["number"].endswith("to_jsonb(%(v)s::numeric)))")
    assert CONTAINS_ATOM_BY_TYPE["boolean"].endswith("to_jsonb(%(v)s::boolean)))")
    assert CONTAINS_ATOM_BY_TYPE["text"].endswith("to_jsonb(%(v)s::text)))")


def test_empty_operand_indexability() -> None:
    # contains_all [] is @> '[]' which jsonb_path_ops cannot accelerate.
    assert FILTER_TRUTH_TABLE_BY_OP["contains_all"].empty_operand_indexable is False
    # in [] / contains_any [] collapse to constant FALSE (no scan).
    assert FILTER_TRUTH_TABLE_BY_OP["in"].empty_operand_indexable is True
    assert FILTER_TRUTH_TABLE_BY_OP["contains_any"].empty_operand_indexable is True


def test_negation_template_frozen() -> None:
    assert NEGATION_TEMPLATE == "NOT ({inner})"


@pytest.mark.parametrize("op", list(EXPECTED_POSITIVE_SQL))
def test_predicate_to_sql_emits_positive(op: str) -> None:
    pred = FieldPredicate(field="year", op=op, value=VALID_FIXTURES[op])  # type: ignore[arg-type]
    sql, params = predicate_to_sql(pred)
    assert sql == EXPECTED_POSITIVE_SQL[op]
    assert params["k"] == "year"


@pytest.mark.parametrize(
    "value, cast",
    [("abc", "text"), (5, "numeric"), (3.5, "numeric"), (True, "boolean")],
)
def test_predicate_to_sql_eq_cast_follows_value_type(value, cast) -> None:
    # The cast is directed by the VALUE's JSON type, not a fixed template.
    sql, params = predicate_to_sql(FieldPredicate(field="f", op="eq", value=value))
    assert sql == (
        f"metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::{cast}))"
    )
    assert params == {"k": "f", "v": value}


@pytest.mark.parametrize("op", ["in", "contains_any", "contains_all"])
def test_predicate_to_sql_list_arity(op: str) -> None:
    # One placeholder per element — never a missing or extra %(vN)s.
    sql, params = predicate_to_sql(
        FieldPredicate(field="f", op=op, value=["a", "b", "c"])
    )
    assert params == {"k": "f", "v0": "a", "v1": "b", "v2": "c"}
    for i in range(3):
        assert f"%(v{i})s" in sql
    assert "%(v3)s" not in sql


def test_predicate_to_sql_one_item_list() -> None:
    sql, params = predicate_to_sql(FieldPredicate(field="f", op="in", value=["a"]))
    assert params == {"k": "f", "v0": "a"}
    assert sql == "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v0)s::text)))"


@pytest.mark.parametrize("op", ["in", "contains_any"])
def test_predicate_to_sql_empty_list_collapses_to_false(op: str) -> None:
    sql, params = predicate_to_sql(FieldPredicate(field="f", op=op, value=[]))
    assert sql == "FALSE"
    assert params == {"k": "f"}  # no dangling value placeholder


def test_predicate_to_sql_rejects_bool_as_number_in_list() -> None:
    with pytest.raises(InvalidFilterError):
        predicate_to_sql(FieldPredicate(field="f", op="in", value=[1, True]))


def test_predicate_to_sql_rejects_range_non_number() -> None:
    with pytest.raises(InvalidFilterError):
        predicate_to_sql(FieldPredicate(field="f", op="gt", value="2026"))
