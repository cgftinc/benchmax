"""Contract #3: 9-op filter truth table.

The frozen table shape is asserted here (passes); the SQL emission via
``predicate_to_sql`` is an xfail skeleton filled by Slice 4. Expected value-present
SQL per op lives in ``EXPECTED_VALUE_PRESENT_SQL`` so Slice 4 has exact targets.
"""

from __future__ import annotations

import pytest

from castform.rag.corpus.neon.filter_mapper import (
    FILTER_TRUTH_TABLE_BY_OP,
    NEGATION_TEMPLATE,
    NEON_FIELD_OPERATORS,
    predicate_to_sql,
)
from castform.rag.corpus.search_schema.search_types import FieldPredicate

EXPECTED_VALUE_PRESENT_SQL: dict[str, str] = {
    "eq": "(metadata ->> %(k)s) = %(v)s",
    "ne": "(metadata ->> %(k)s) IS DISTINCT FROM %(v)s",
    "in": "(metadata ->> %(k)s) = ANY(%(v)s)",
    "gt": "(metadata ->> %(k)s)::numeric > %(v)s",
    "gte": "(metadata ->> %(k)s)::numeric >= %(v)s",
    "lt": "(metadata ->> %(k)s)::numeric < %(v)s",
    "lte": "(metadata ->> %(k)s)::numeric <= %(v)s",
    "contains_any": "(metadata -> %(k)s) ?| %(v)s",
    "contains_all": "(metadata -> %(k)s) ?& %(v)s",
}

# op -> (null/missing-key semantic, empty-array-operand semantic).
EXPECTED_EDGE_SEMANTICS: dict[str, tuple[str, str | None]] = {
    "eq": ("exclude", None),
    "ne": ("include", None),
    "in": ("exclude", "exclude"),
    "gt": ("exclude", None),
    "gte": ("exclude", None),
    "lt": ("exclude", None),
    "lte": ("exclude", None),
    "contains_any": ("exclude", "exclude"),
    "contains_all": ("exclude", "include"),
}


def test_nine_operators_frozen() -> None:
    assert set(NEON_FIELD_OPERATORS) == set(EXPECTED_VALUE_PRESENT_SQL)
    assert len(NEON_FIELD_OPERATORS) == 9
    # ne/gt/lt are the three added on top of the shared six.
    assert {"ne", "gt", "lt"} <= set(NEON_FIELD_OPERATORS)


@pytest.mark.parametrize("op", list(EXPECTED_VALUE_PRESENT_SQL))
def test_truth_table_sql_shape_frozen(op: str) -> None:
    assert FILTER_TRUTH_TABLE_BY_OP[op].sql_template == EXPECTED_VALUE_PRESENT_SQL[op]


@pytest.mark.parametrize("op", list(EXPECTED_EDGE_SEMANTICS))
def test_truth_table_edge_semantics_frozen(op: str) -> None:
    spec = FILTER_TRUTH_TABLE_BY_OP[op]
    null_sem, empty_sem = EXPECTED_EDGE_SEMANTICS[op]
    assert spec.null_or_missing == null_sem
    assert spec.empty_array == empty_sem


def test_negation_template_frozen() -> None:
    assert NEGATION_TEMPLATE == "NOT ({inner})"


@pytest.mark.xfail(reason="filter SQL emission built in Slice 4", strict=False)
@pytest.mark.parametrize("op", list(EXPECTED_VALUE_PRESENT_SQL))
def test_predicate_to_sql_emits_expected(op: str) -> None:
    pred = FieldPredicate(field="year", op=op, value=2026)  # type: ignore[arg-type]
    sql, params = predicate_to_sql(pred)
    assert sql == EXPECTED_VALUE_PRESENT_SQL[op]
    assert params["k"] == "year"
