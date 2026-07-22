"""Filter DSL -> parameterized SQL truth table for the Neon corpus.

Contract-freeze artifact (Slice A). This module freezes the *shape* of the SQL
each of the nine field operators emits and how each behaves against the five
edge conditions (value present, JSON null, missing key, empty-array operand,
negation). The translator that walks a ``FilterPredicate`` tree and emits the
final ``(sql, params)`` pair is built in Slice 4 — ``predicate_to_sql`` is a stub.

Bound-path discipline
---------------------
Metadata is accessed through JSON path operators with the *key bound as a
parameter* (``metadata ->> %(k)s``), never by interpolating the key into an
identifier. ``psycopg.sql.Identifier`` is reserved for trusted schema-owned
names (table/column), never for arbitrary caller-supplied JSON keys. Values are
always bound; the only per-op variation is the cast (``::numeric`` vs text) and
the array/containment operator family.

Operator set
------------
Nine field operators. The shared ``FieldOperator`` enum in
``search_schema/search_types.py`` today carries six (``eq, in, gte, lte,
contains_any, contains_all``); Neon adds ``ne, gt, lt``. Promoting the three new
ops into the shared enum is deferred to Slice 4 — this module declares a local
superset so the contract can be frozen without a cross-cutting schema edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from castform.rag.corpus.search_schema.search_types import FilterPredicate

NeonFieldOperator = Literal[
    "eq", "ne", "in", "gt", "gte", "lt", "lte", "contains_any", "contains_all"
]
"""The nine Neon field operators (shared six + ne/gt/lt)."""

NEON_FIELD_OPERATORS: tuple[NeonFieldOperator, ...] = (
    "eq",
    "ne",
    "in",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains_any",
    "contains_all",
)

# Cast rule: comparison and equality ops cast to ``numeric`` iff the DSL value is
# an int/float (or a list thereof for ``in``); otherwise text via ``->>``.
# ``contains_any``/``contains_all`` treat the field as a JSONB array of text and
# use the JSONB key-existence operators ``?|`` / ``?&`` (the ``@>`` / ``&&``
# array form is the deferred typed-array fallback, noted per row).

CastKind = Literal["numeric", "text", "jsonb_array"]

# NULL / missing-key emission semantics for a single leaf. ``exclude`` = the
# leaf is NULL under 3-valued logic and the row drops; ``include`` = the leaf is
# TRUE for a NULL/missing left operand (only ``ne`` via IS DISTINCT FROM).
NullSemantic = Literal["exclude", "include"]


@dataclass(frozen=True)
class FilterOpSpec:
    """Frozen SQL contract for one field operator.

    Args:
        op: The Neon field operator.
        sql_template: Emitted SQL for the value-present case. ``%(k)s`` binds the
            metadata key, ``%(v)s`` binds the value (or ``%(v)s::type[]`` array).
        cast: How the left operand / value is cast.
        null_or_missing: Behavior when the key is JSON-null or absent.
        empty_array: Behavior when the operand array is empty (in/contains_*),
            or ``None`` for scalar ops.
        typed_array_fallback: Deferred alternative representation, if any.
    """

    op: NeonFieldOperator
    sql_template: str
    cast: CastKind
    null_or_missing: NullSemantic
    empty_array: Literal["exclude", "include"] | None
    typed_array_fallback: str | None = None


# --- Contract #3: the frozen 9-op truth table ---------------------------------
# Value-present SQL per op. Numeric variants shown; the text variant drops the
# ``::numeric`` cast and compares ``metadata ->> %(k)s`` directly.
FILTER_TRUTH_TABLE: tuple[FilterOpSpec, ...] = (
    FilterOpSpec(
        op="eq",
        sql_template="(metadata ->> %(k)s) = %(v)s",
        cast="text",
        null_or_missing="exclude",
        empty_array=None,
    ),
    FilterOpSpec(
        # IS DISTINCT FROM makes ne null-safe: a missing/JSON-null key is
        # "distinct from" any concrete value, so ne *includes* such rows.
        op="ne",
        sql_template="(metadata ->> %(k)s) IS DISTINCT FROM %(v)s",
        cast="text",
        null_or_missing="include",
        empty_array=None,
    ),
    FilterOpSpec(
        op="in",
        sql_template="(metadata ->> %(k)s) = ANY(%(v)s)",
        cast="text",
        null_or_missing="exclude",
        empty_array="exclude",
    ),
    FilterOpSpec(
        op="gt",
        sql_template="(metadata ->> %(k)s)::numeric > %(v)s",
        cast="numeric",
        null_or_missing="exclude",
        empty_array=None,
    ),
    FilterOpSpec(
        op="gte",
        sql_template="(metadata ->> %(k)s)::numeric >= %(v)s",
        cast="numeric",
        null_or_missing="exclude",
        empty_array=None,
    ),
    FilterOpSpec(
        op="lt",
        sql_template="(metadata ->> %(k)s)::numeric < %(v)s",
        cast="numeric",
        null_or_missing="exclude",
        empty_array=None,
    ),
    FilterOpSpec(
        op="lte",
        sql_template="(metadata ->> %(k)s)::numeric <= %(v)s",
        cast="numeric",
        null_or_missing="exclude",
        empty_array=None,
    ),
    FilterOpSpec(
        # JSONB key-existence-any over an array-of-text field. Empty operand
        # array => matches nothing (exclude).
        op="contains_any",
        sql_template="(metadata -> %(k)s) ?| %(v)s",
        cast="jsonb_array",
        null_or_missing="exclude",
        empty_array="exclude",
        typed_array_fallback="(metadata -> %(k)s) && %(v)s  -- typed array &&",
    ),
    FilterOpSpec(
        # JSONB key-existence-all. Empty operand array is vacuously TRUE
        # (include) — the one op where an empty operand does not exclude.
        op="contains_all",
        sql_template="(metadata -> %(k)s) ?& %(v)s",
        cast="jsonb_array",
        null_or_missing="exclude",
        empty_array="include",
        typed_array_fallback="(metadata -> %(k)s) @> %(v)s  -- typed array @>",
    ),
)

# Negation contract: ``NotPredicate`` wraps its inner SQL as ``NOT (<inner>)`` and
# inherits SQL three-valued logic — ``NOT (NULL)`` is NULL, so negating a leaf
# over a missing key still *excludes* the row. Making negation null-inclusive
# would require ``(<inner>) IS NOT TRUE``; the frozen contract is plain
# ``NOT (...)``. This sharp edge is documented in CONTRACT.md.
NEGATION_TEMPLATE = "NOT ({inner})"

FILTER_TRUTH_TABLE_BY_OP: dict[NeonFieldOperator, FilterOpSpec] = {
    spec.op: spec for spec in FILTER_TRUTH_TABLE
}


def predicate_to_sql(
    predicate: FilterPredicate | None,
) -> tuple[str, dict[str, object]]:
    """Translate a predicate tree to a parameterized SQL fragment + bound params.

    Returns ``(sql, params)`` where every metadata key and value is a bound
    parameter. Capability-gates unsupported ops via the search-schema
    exceptions. Design-lock stub: the tree walk lands in Slice 4.
    """
    raise NotImplementedError("filter SQL emission is built in Slice 4")
