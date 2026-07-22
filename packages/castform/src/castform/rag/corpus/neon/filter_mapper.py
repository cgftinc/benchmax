"""Filter DSL -> parameterized, INDEXABLE SQL truth table for the Neon corpus.

Contract-freeze artifact (Slice A). This module freezes, per operator: the
type-directed SQL shape, its five distinct edge outcomes (missing key, JSON null,
wrong stored type, empty operand, negated), whether the shape is index-eligible,
and the value-validation rules. The translator that walks a ``FilterPredicate``
tree and emits ``(sql, params)`` is Slice 4 — ``predicate_to_sql`` is a stub.

Two safety properties are frozen here (they drove the review REVISE):

1. **Type-directed, never-throwing.** ``eq/ne/in`` and the ``contains_*`` ops are
   emitted as JSONB **containment** (``metadata @> jsonb_build_object(...)``),
   which is *type-aware* (``'{"a":5}' @> '{"a":"5"}'`` is false) and therefore
   needs no cast — a heterogeneous stored value can never abort the query. The
   range ops (``gt/gte/lt/lte``) are the only cast path, and they are **guarded**
   by ``jsonb_typeof(metadata -> key) = 'number'`` so ``::numeric`` is reached
   only for numbers.
2. **Indexable.** Containment is served by a ``jsonb_path_ops`` GIN (see
   ``schema.CREATE_META_GIN_INDEX_SKELETON``). The earlier ``?|``/``?&`` forms are
   NOT used: a whole-doc GIN cannot serve ``(metadata -> key) ?| values`` and
   ``jsonb_path_ops`` does not index key-existence at all (B3). Range predicates
   are not GIN-indexable; a per-key expression btree is an operational add-on.

Bound-path discipline: the metadata key and every value are bound parameters
(``%(k)s`` / ``%(v)s``); ``psycopg.sql.Identifier`` is reserved for trusted
schema-owned names, never for caller JSON keys.

Operator set: nine field ops. The shared enum carries six (``eq, in, gte, lte,
contains_any, contains_all``); Neon adds ``ne, gt, lt`` in a local superset;
promoting them into ``search_schema/search_types.py`` is a Slice 4 cross-cutting
edit kept out of this slice.
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
            (``@> ... IS NOT TRUE``), or ``range`` (guarded numeric cast).
        canonical_sql: The exact emitted SQL for a representative value-present
            case. ``%(k)s`` binds the key; ``%(v)s`` / ``%(v0)s`` / ``%(v1)s`` bind
            values. ``in``/``contains_any`` show the two-element OR expansion;
            ``contains_all`` shows the two-element single-containment form.
        value_types: Scalar JSON types this op accepts (element types for the
            list ops).
        indexable: Whether ``meta_gin`` (``jsonb_path_ops``) can serve it.
        outcomes: Fixed include/exclude behavior per edge condition.
    """

    op: NeonFieldOperator
    family: Literal["containment", "negated_containment", "range"]
    canonical_sql: str
    value_types: tuple[ScalarJsonType, ...]
    indexable: bool
    outcomes: dict[EdgeConditions, Outcome]


# Every op excludes on missing key / JSON null / wrong stored type EXCEPT ``ne``
# (null-safe: ``IS NOT TRUE`` flips NULL/false to true, so a missing/null/wrong
# value is *included*).
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


# --- Contract #3: the frozen 9-op truth table ---------------------------------
FILTER_TRUTH_TABLE: tuple[FilterOpSpec, ...] = (
    FilterOpSpec(
        op="eq",
        family="containment",
        canonical_sql="metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))",
        value_types=("text", "number", "boolean"),
        indexable=True,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        # Null-safe: missing/null/wrong-type rows are INCLUDED; only an equal
        # stored value is excluded.
        op="ne",
        family="negated_containment",
        canonical_sql=(
            "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v)s::numeric))) "
            "IS NOT TRUE"
        ),
        value_types=("text", "number", "boolean"),
        indexable=False,
        outcomes=dict(_NE_INCLUDING),
    ),
    FilterOpSpec(
        op="in",
        family="containment",
        canonical_sql=(
            "(metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v0)s::numeric)) OR "
            "metadata @> jsonb_build_object(%(k)s, to_jsonb(%(v1)s::numeric)))"
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "exclude"},
    ),
    FilterOpSpec(
        op="gt",
        family="range",
        canonical_sql=(
            "jsonb_typeof(metadata -> %(k)s) = 'number' "
            "AND (metadata ->> %(k)s)::numeric > %(v)s::numeric"
        ),
        value_types=("number",),
        indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="gte",
        family="range",
        canonical_sql=(
            "jsonb_typeof(metadata -> %(k)s) = 'number' "
            "AND (metadata ->> %(k)s)::numeric >= %(v)s::numeric"
        ),
        value_types=("number",),
        indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="lt",
        family="range",
        canonical_sql=(
            "jsonb_typeof(metadata -> %(k)s) = 'number' "
            "AND (metadata ->> %(k)s)::numeric < %(v)s::numeric"
        ),
        value_types=("number",),
        indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        op="lte",
        family="range",
        canonical_sql=(
            "jsonb_typeof(metadata -> %(k)s) = 'number' "
            "AND (metadata ->> %(k)s)::numeric <= %(v)s::numeric"
        ),
        value_types=("number",),
        indexable=False,
        outcomes=dict(_SCALAR_EXCLUDING),
    ),
    FilterOpSpec(
        # Array membership via per-element containment OR. Empty operand => no
        # atoms => FALSE => exclude.
        op="contains_any",
        family="containment",
        canonical_sql=(
            "(metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
            "metadata @> jsonb_build_object(%(k)s, jsonb_build_array(to_jsonb(%(v1)s::text))))"
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "exclude"},
    ),
    FilterOpSpec(
        # Superset via single array containment. Empty operand => ``@> '[]'`` =>
        # TRUE iff the field is present as an array (missing key still excludes).
        op="contains_all",
        family="containment",
        canonical_sql=(
            "metadata @> jsonb_build_object(%(k)s, "
            "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        outcomes={**_SCALAR_EXCLUDING, "empty_operand": "include"},
    ),
)

# Negation contract: ``NotPredicate`` wraps inner SQL as ``NOT (<inner>)`` and
# inherits SQL three-valued logic — ``NOT (NULL)`` is NULL, so negating a leaf
# over a missing key still *excludes* that row. Null-inclusive negation would
# need ``(<inner>) IS NOT TRUE``; the frozen contract is plain ``NOT (...)``.
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
    bool-as-number), emits the type-directed containment/range SQL, and
    capability-gates unsupported ops via the search-schema exceptions.
    Design-lock stub: the tree walk lands in Slice 4.
    """
    raise NotImplementedError("filter SQL emission is built in Slice 4")
