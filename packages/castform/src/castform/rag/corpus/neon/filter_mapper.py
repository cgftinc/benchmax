"""Filter DSL -> parameterized, INDEXABLE SQL truth table for the Neon corpus.

Contract-freeze artifact (Slice A). This module freezes, per operator: the
type-directed positive SQL, the three-valued *negated-leaf* SQL, the five edge
outcomes, index-eligibility, and the value-validation rules. The translator that
walks a ``FilterPredicate`` tree and emits ``(sql, params)`` is Slice 4 —
``predicate_to_sql`` is a stub.

Three safety properties are frozen (they drove the review):

1. **Type-directed, never-throwing.** ``eq/ne/in`` and ``contains_*`` emit JSONB
   **containment** (``metadata @> jsonb_build_object(...)``), which is type-aware
   (``'{"a":5}' @> '{"a":"5"}'`` is false) and needs no cast. The range ops are
   the only cast path, and the cast is placed **inside a CASE** (``CASE WHEN
   jsonb_typeof(...) = 'number' THEN (...)::numeric ... ELSE NULL END``) — NOT
   behind ``AND``. Postgres does not guarantee ``AND`` short-circuits, but a CASE
   only evaluates the matching branch, so ``::numeric`` never sees a non-number.
2. **Correct negation.** Bare containment is two-valued (FALSE for a
   missing/null/wrong-type key), so ``NOT(FALSE)`` would wrongly INCLUDE those
   rows. Every op therefore also exposes a **three-valued negated leaf**
   (``negated_leaf_sql``) that yields SQL ``NULL`` for missing/null/wrong-type via
   a guarding CASE; ``NotPredicate`` wraps *that* in ``NOT(...)``, so
   ``NOT(NULL) = NULL`` keeps those rows excluded (matching the truth table).
3. **Indexable positives.** The positive containment forms are served by the
   ``meta_gin`` ``jsonb_path_ops`` GIN. Negated leaves (CASE) and range CASEs are
   not GIN-eligible — and neither is empty-array ``contains_all`` (``@> '[]'`` has
   no scalar token in ``jsonb_path_ops``), so its ``empty_operand_indexable`` is
   False (B3 caveat).

Bound-path discipline: the metadata key and every value are bound parameters
(``%(k)s`` / ``%(v)s``); key existence uses ``jsonb_exists(metadata, %(k)s)`` (the
function form, not the ``?`` operator, which collides with psycopg placeholder
parsing). ``psycopg.sql.Identifier`` is never used on caller keys.

Operator set: nine field ops. The shared enum carries six (``eq, in, gte, lte,
contains_any, contains_all``); Neon adds ``ne, gt, lt`` in a local superset;
promoting them into ``search_schema/search_types.py`` is a Slice 4 edit kept out
of this slice.
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

# JSON scalar type of a bound value -> the explicit SQL cast used inside
# ``to_jsonb(... ::cast)``. Python ``bool`` is checked BEFORE ``int`` (bool is an
# int subclass) so a boolean is never silently treated as a number.
ScalarJsonType = Literal["text", "number", "boolean"]
CAST_BY_SCALAR_TYPE: dict[ScalarJsonType, str] = {
    "text": "text",
    "number": "numeric",
    "boolean": "boolean",
}
# The ``jsonb_typeof`` token the CASE guard compares against, per value type.
JSON_TYPEOF_BY_SCALAR_TYPE: dict[ScalarJsonType, str] = {
    "text": "string",
    "number": "number",
    "boolean": "boolean",
}

# Explicit contains_* element shapes per type (freezes numeric/boolean, not just
# text): the single-element containment atom Slice 4 ORs (contains_any) or
# array-joins (contains_all).
CONTAINS_ATOM_BY_TYPE: dict[ScalarJsonType, str] = {
    "text": "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v)s::text)))",
    "number": "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v)s::numeric)))",
    "boolean": "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v)s::boolean)))",
}

# Per-condition outcome for a single leaf. ``depends`` = matches iff the stored
# value satisfies the op; the other four are fixed regardless of value.
Outcome = Literal["depends", "exclude", "include", "na"]
EdgeConditions = Literal["missing_key", "json_null", "wrong_type", "empty_operand"]


@dataclass(frozen=True)
class FilterOpSpec:
    """Frozen SQL + edge-outcome contract for one field operator.

    Args:
        op: The Neon field operator.
        family: ``containment`` (indexable ``@>``), ``negated_containment``
            (``@> ... IS NOT TRUE``), or ``range`` (guarded CASE cast).
        positive_sql: The WHERE form for a NON-negated leaf. Representative
            value-present case; ``%(k)s`` binds the key, ``%(v)s``/``%(v0)s``/
            ``%(v1)s`` bind values. Indexable iff ``indexable``.
        negated_leaf_sql: The THREE-VALUED expression a ``NotPredicate`` wraps in
            ``NOT(...)``. Yields NULL for missing/null/wrong-type so negation
            still excludes them.
        value_types: Scalar JSON types this op accepts (element types for lists).
        indexable: Whether ``meta_gin`` can serve ``positive_sql`` (non-empty
            operand).
        empty_operand_indexable: For list ops, whether the empty-operand case is
            index-accelerated (False for ``contains_all []``).
        outcomes: Fixed include/exclude behavior of the POSITIVE leaf per edge
            condition (negation inverts via ``negated_leaf_sql``).
    """

    op: NeonFieldOperator
    family: Literal["containment", "negated_containment", "range"]
    positive_sql: str
    negated_leaf_sql: str
    value_types: tuple[ScalarJsonType, ...]
    indexable: bool
    empty_operand_indexable: bool
    outcomes: dict[EdgeConditions, Outcome]


_SCALAR_EXCLUDING = {
    "missing_key": "exclude",
    "json_null": "exclude",
    "wrong_type": "exclude",
    "empty_operand": "na",
}
_NE_INCLUDING = {
    "missing_key": "include",
    "json_null": "include",
    "wrong_type": "include",
    "empty_operand": "na",
}

# Guarding CASE that makes a containment leaf three-valued for negation. The
# ``{inner}`` is the positive containment; the type guard also catches json-null
# (its jsonb_typeof is 'null').
_NEG_CASE = (
    "CASE WHEN NOT jsonb_exists(metadata, %(k)s) THEN NULL "
    "WHEN jsonb_typeof(metadata -> %(k)s) <> '{jtype}' THEN NULL "
    "ELSE {inner} END"
)


# --- Contract #3: the frozen 9-op truth table ---------------------------------
FILTER_TRUTH_TABLE: tuple[FilterOpSpec, ...] = (
    FilterOpSpec(
        op="eq",
        family="containment",
        positive_sql="metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))",
        negated_leaf_sql=_NEG_CASE.format(
            jtype="number",
            inner="metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))",
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        empty_operand_indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        # ne is null-INCLUSIVE by design: (@>) IS NOT TRUE includes
        # missing/null/wrong-type and excludes only an equal stored value. This
        # differs from NotPredicate(eq), which is null-EXCLUSIVE.
        op="ne",
        family="negated_containment",
        positive_sql=(
            "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) "
            "IS NOT TRUE"
        ),
        negated_leaf_sql=(
            "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) "
            "IS NOT TRUE"
        ),
        value_types=("text", "number", "boolean"),
        indexable=False,
        empty_operand_indexable=False,
        outcomes=dict(_NE_INCLUDING),
    ),
    FilterOpSpec(
        op="in",
        family="containment",
        positive_sql=(
            "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v0)s::numeric)) OR "
            "metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v1)s::numeric)))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="number",
            inner=(
                "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v0)s::numeric)) OR "
                "metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v1)s::numeric)))"
            ),
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        empty_operand_indexable=True,  # empty in => constant FALSE, no scan
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "exclude"},
    ),
    FilterOpSpec(
        op="gt",
        family="range",
        positive_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric > %(v)s::numeric ELSE NULL END"
        ),
        negated_leaf_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric > %(v)s::numeric ELSE NULL END"
        ),
        value_types=("number",),
        indexable=False,
        empty_operand_indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="gte",
        family="range",
        positive_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric >= %(v)s::numeric ELSE NULL END"
        ),
        negated_leaf_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric >= %(v)s::numeric ELSE NULL END"
        ),
        value_types=("number",),
        indexable=False,
        empty_operand_indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="lt",
        family="range",
        positive_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric < %(v)s::numeric ELSE NULL END"
        ),
        negated_leaf_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric < %(v)s::numeric ELSE NULL END"
        ),
        value_types=("number",),
        indexable=False,
        empty_operand_indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="lte",
        family="range",
        positive_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric <= %(v)s::numeric ELSE NULL END"
        ),
        negated_leaf_sql=(
            "CASE WHEN jsonb_typeof(metadata -> %(k)s) = 'number' "
            "THEN (metadata ->> %(k)s)::numeric <= %(v)s::numeric ELSE NULL END"
        ),
        value_types=("number",),
        indexable=False,
        empty_operand_indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        # Array membership via per-element containment OR. Empty operand => no
        # atoms => FALSE => exclude (constant, no scan).
        op="contains_any",
        family="containment",
        positive_sql=(
            "(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
            "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v1)s::text))))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="array",
            inner=(
                "(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
                "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v1)s::text))))"
            ),
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        empty_operand_indexable=True,  # empty => constant FALSE
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "exclude"},
    ),
    FilterOpSpec(
        # Superset via single array containment. Empty operand => ``@> '[]'`` =>
        # TRUE iff the field is a present array (missing key still excludes), and
        # ``@> '[]'`` has no jsonb_path_ops scalar token => NOT index-accelerated.
        op="contains_all",
        family="containment",
        positive_sql=(
            "metadata @> jsonb_build_object(%(k)s, "
            "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="array",
            inner=(
                "metadata @> jsonb_build_object(%(k)s, "
                "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
            ),
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        empty_operand_indexable=False,  # @> '[]' full-scans (B3 caveat)
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "include"},
    ),
)

# ``NotPredicate`` wraps the op's THREE-VALUED ``negated_leaf_sql`` (never the
# indexable positive), so ``NOT(NULL) = NULL`` keeps missing/null/wrong-type rows
# excluded. Null-inclusive negation would need ``IS NOT TRUE`` (that is ``ne``).
NEGATION_TEMPLATE = "NOT ({inner})"

FILTER_TRUTH_TABLE_BY_OP: dict[NeonFieldOperator, FilterOpSpec] = {
    spec.op: spec for spec in FILTER_TRUTH_TABLE
}

# Value-validation rules frozen for Slice 4's predicate_to_sql (each raises
# InvalidFilterError from search_schema.search_exceptions):
#   - range ops require a numeric value (int/float, NOT bool);
#   - in/contains_* require a homogeneous list — mixed JSON types are rejected,
#     and a Python bool is never accepted where a number is expected;
#   - eq/ne accept a single text/number/boolean scalar.
RANGE_OPS: frozenset[NeonFieldOperator] = frozenset({"gt", "gte", "lt", "lte"})
LIST_OPS: frozenset[NeonFieldOperator] = frozenset(
    {"in", "contains_any", "contains_all"}
)


def predicate_to_sql(
    predicate: FilterPredicate | None,
) -> tuple[str, dict[str, object]]:
    """Translate a predicate tree to parameterized SQL + bound params.

    Enforces the value-validation rules above (rejecting mixed-type lists and
    bool-as-number), emits the type-directed containment/range SQL (positive
    leaves for plain predicates, ``NOT(negated_leaf_sql)`` under ``NotPredicate``),
    and capability-gates unsupported ops via the search-schema exceptions.
    Design-lock stub: the tree walk lands in Slice 4.
    """
    raise NotImplementedError("filter SQL emission is built in Slice 4")
