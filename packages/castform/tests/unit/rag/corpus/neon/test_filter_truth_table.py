"""Contract #3: type-directed, indexable 9-op filter truth table.

The frozen table (canonical SQL, edge outcomes, indexability) is asserted here
and passes; the SQL emission via ``predicate_to_sql`` is an xfail skeleton that
must raise NotImplementedError until Slice 4 fills it.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.filter_mapper import (
    FILTER_TRUTH_TABLE_BY_OP,
    LIST_OPS,
    NEGATION_TEMPLATE,
    NEON_FIELD_OPERATORS,
    RANGE_OPS,
    predicate_to_sql,
)
from castform.rag.corpus.search_schema.search_types import FieldPredicate

# Frozen canonical value-present SQL per op (containment for eq/ne/in/contains_*,
# guarded numeric cast for ranges). Slice 4 must emit exactly these.
EXPECTED_CANONICAL_SQL: dict[str, str] = {
    "eq": "metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))",
    "ne": (
        "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) IS NOT TRUE"
    ),
    "in": (
        "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v0)s::numeric)) OR "
        "metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v1)s::numeric)))"
    ),
    "gt": (
        "jsonb_typeof(metadata -> %(k)s) = 'number' "
        "AND (metadata ->> %(k)s)::numeric > %(v)s::numeric"
    ),
    "gte": (
        "jsonb_typeof(metadata -> %(k)s) = 'number' "
        "AND (metadata ->> %(k)s)::numeric >= %(v)s::numeric"
    ),
    "lt": (
        "jsonb_typeof(metadata -> %(k)s) = 'number' "
        "AND (metadata ->> %(k)s)::numeric < %(v)s::numeric"
    ),
    "lte": (
        "jsonb_typeof(metadata -> %(k)s) = 'number' "
        "AND (metadata ->> %(k)s)::numeric <= %(v)s::numeric"
    ),
    "contains_any": (
        "(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
        "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v1)s::text))))"
    ),
    "contains_all": (
        "metadata @> jsonb_build_object(%(k)s, "
        "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
    ),
}

# op -> (missing_key, json_null, wrong_type, empty_operand) outcomes.
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

# Only containment forms are GIN-indexable; ne (negation) and ranges are not.
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

# Valid per-op fixtures — list ops get lists, ranges get numbers, eq/ne a scalar.
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
    assert set(NEON_FIELD_OPERATORS) == set(EXPECTED_CANONICAL_SQL)
    assert len(NEON_FIELD_OPERATORS) == 9
    assert {"ne", "gt", "lt"} <= set(NEON_FIELD_OPERATORS)
    assert RANGE_OPS == {"gt", "gte", "lt", "lte"}
    assert LIST_OPS == {"in", "contains_any", "contains_all"}


@pytest.mark.parametrize("op", list(EXPECTED_CANONICAL_SQL))
def test_canonical_sql_frozen(op: str) -> None:
    assert FILTER_TRUTH_TABLE_BY_OP[op].canonical_sql == EXPECTED_CANONICAL_SQL[op]


@pytest.mark.parametrize("op", list(EXPECTED_EDGE_OUTCOMES))
def test_edge_outcomes_frozen(op: str) -> None:
    spec = FILTER_TRUTH_TABLE_BY_OP[op]
    o = spec.outcomes
    assert (
        o["missing_key"],
        o["json_null"],
        o["wrong_type"],
        o["empty_operand"],
    ) == EXPECTED_EDGE_OUTCOMES[op]


@pytest.mark.parametrize("op", list(EXPECTED_INDEXABLE))
def test_indexability_frozen(op: str) -> None:
    assert FILTER_TRUTH_TABLE_BY_OP[op].indexable == EXPECTED_INDEXABLE[op]


def test_negation_template_frozen() -> None:
    assert NEGATION_TEMPLATE == "NOT ({inner})"


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Slice 4")
@pytest.mark.parametrize("op", list(EXPECTED_CANONICAL_SQL))
def test_predicate_to_sql_emits_canonical(op: str) -> None:
    pred = FieldPredicate(field="year", op=op, value=VALID_FIXTURES[op])  # type: ignore[arg-type]
    sql, params = predicate_to_sql(pred)
    assert sql == EXPECTED_CANONICAL_SQL[op]
    assert params["k"] == "year"
