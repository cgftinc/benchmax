"""Translate Castform data-preparation filters to parameterized Neon SQL.

``to_neon_filters`` and ``to_neon_where`` walk a ``FilterPredicate`` tree and
emit parameterized WHERE SQL. ``predicate_to_sql`` renders a single positive
field predicate.

The implementation maintains three safety properties:

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
   false.

Bound-path discipline: the metadata key and every value are bound parameters
(``%(k)s`` / ``%(v)s``); key existence uses ``jsonb_exists(metadata, %(k)s)`` (the
function form, not the ``?`` operator, which collides with psycopg placeholder
parsing). ``psycopg.sql.Identifier`` is never used on caller keys. The key param is
cast ``%(k)s::text`` inside every ``jsonb_build_object`` — that function is VARIADIC
``"any"`` and cannot infer a bound param's type, so an uncast key raises
``IndeterminateDatatype`` on live Postgres (the range ops' ``metadata -> %(k)s``
resolves the param to text on its own, so it needs no explicit cast).

The SQL table also implements ``ne``, ``gt``, and ``lt`` for internal use. The
public capability check accepts only the six operators declared by Benchmax's
shared predicate model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from castform.rag.corpus.search_schema.search_exceptions import (
    InvalidFilterError,
    UnsupportedFilterError,
)
from castform.rag.corpus.search_schema.search_types import (
    AndPredicate,
    FieldPredicate,
    FilterPredicate,
    NotPredicate,
    OrPredicate,
    SearchCapabilities,
)
from psycopg import sql

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

# Explicit contains_* element shapes per type, including numeric and boolean
# values. ``contains_any`` ORs atoms; ``contains_all`` joins them with AND.
CONTAINS_ATOM_BY_TYPE: dict[ScalarJsonType, str] = {
    "text": "metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v)s::text)))",
    "number": (
        "metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v)s::numeric)))"
    ),
    "boolean": (
        "metadata @> jsonb_build_object(%(k)s::text, jsonb_build_array(to_jsonb(%(v)s::boolean)))"
    ),
}

# Per-condition outcome for a single leaf. ``depends`` = matches iff the stored
# value satisfies the op; the other four are fixed regardless of value.
Outcome = Literal["depends", "exclude", "include", "na"]
EdgeConditions = Literal["missing_key", "json_null", "wrong_type", "empty_operand"]


@dataclass(frozen=True)
class FilterOpSpec:
    """SQL and edge-case behavior for one field operator.

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


# --- operator truth table -----------------------------------------------------
FILTER_TRUTH_TABLE: tuple[FilterOpSpec, ...] = (
    FilterOpSpec(
        op="eq",
        family="containment",
        positive_sql="metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))",
        negated_leaf_sql=_NEG_CASE.format(
            jtype="number",
            inner="metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))",
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
            "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))) IS NOT TRUE"
        ),
        negated_leaf_sql=(
            "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v)s::numeric))) IS NOT TRUE"
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
            "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v0)s::numeric)) OR "
            "metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v1)s::numeric)))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="number",
            inner=(
                "(metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v0)s::numeric)) OR "
                "metadata @> jsonb_build_object(%(k)s::text, to_jsonb(%(v1)s::numeric)))"
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
            "(metadata @> jsonb_build_object(%(k)s::text, "
            "jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
            "metadata @> jsonb_build_object(%(k)s::text, "
            "jsonb_build_array(to_jsonb(%(v1)s::text))))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="array",
            inner=(
                "(metadata @> jsonb_build_object(%(k)s::text, "
                "jsonb_build_array(to_jsonb(%(v0)s::text))) OR "
                "metadata @> jsonb_build_object(%(k)s::text, "
                "jsonb_build_array(to_jsonb(%(v1)s::text))))"
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
            "metadata @> jsonb_build_object(%(k)s::text, "
            "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
        ),
        negated_leaf_sql=_NEG_CASE.format(
            jtype="array",
            inner=(
                "metadata @> jsonb_build_object(%(k)s::text, "
                "jsonb_build_array(to_jsonb(%(v0)s::text), to_jsonb(%(v1)s::text)))"
            ),
        ),
        value_types=("text", "number", "boolean"),
        indexable=True,
        empty_operand_indexable=False,  # @> '[]' requires a full scan
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

# Value-validation rules enforced by ``to_neon_filters`` (each raises
# ``InvalidFilterError`` from ``search_schema.search_exceptions``):
#   - range ops require a numeric value (int/float, NOT bool);
#   - in/contains_* require a homogeneous list — mixed JSON types are rejected,
#     and a Python bool is never accepted where a number is expected;
#   - eq/ne accept a single text/number/boolean scalar.
RANGE_OPS: frozenset[NeonFieldOperator] = frozenset({"gt", "gte", "lt", "lte"})
LIST_OPS: frozenset[NeonFieldOperator] = frozenset({"in", "contains_any", "contains_all"})

_NEON_FILTER_CAPABILITIES: SearchCapabilities = {
    "backend": "neon",
    "modes": {"lexical", "vector", "hybrid"},
    "filter_ops": {
        "field": {"eq", "in", "gte", "lte", "contains_any", "contains_all"},
        "logical": {"and", "or", "not"},
    },
    "ranking": set(),
    "constraints": {},
    "graph_expansion": False,
}

_RANGE_SQL_OPERATOR: dict[NeonFieldOperator, str] = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


def _ensure_supported(
    capabilities: SearchCapabilities,
    predicate: FilterPredicate | None,
) -> None:
    """Validate a predicate tree against backend filter capabilities."""
    backend = str(capabilities.get("backend", "unknown"))
    filter_ops = capabilities.get("filter_ops", {"field": set(), "logical": set()})
    field_operators = set(filter_ops.get("field", []))
    logical_operators = set(filter_ops.get("logical", []))

    if predicate is None:
        return

    if isinstance(predicate, FieldPredicate):
        if predicate.op not in field_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message=f"field operator '{predicate.op}' is not supported",
                predicate=predicate,
            )
        if not isinstance(predicate.field, str) or not predicate.field.strip():
            raise InvalidFilterError(
                backend=backend,
                message="field predicate must have a non-empty field name",
                predicate=predicate,
            )
        return

    if isinstance(predicate, (AndPredicate, OrPredicate)):
        operator = "and" if isinstance(predicate, AndPredicate) else "or"
        if operator not in logical_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message=f"logical operator '{operator}' is not supported",
                predicate=predicate,
            )
        if not predicate.clauses:
            raise InvalidFilterError(
                backend=backend,
                message=f"'{operator}' must include at least one clause",
                predicate=predicate,
            )
        for clause in predicate.clauses:
            _ensure_supported(capabilities, clause)
        return

    if isinstance(predicate, NotPredicate):
        if "not" not in logical_operators:
            raise UnsupportedFilterError(
                backend=backend,
                message="logical operator 'not' is not supported",
                predicate=predicate,
            )
        _ensure_supported(capabilities, predicate.clause)
        return

    raise InvalidFilterError(
        backend=backend,
        message=f"unexpected predicate type '{type(predicate).__name__}'",
        predicate=predicate,
    )


@dataclass(frozen=True)
class _RenderedPredicate:
    """Positive and three-valued forms of one rendered predicate tree."""

    positive: sql.Composable
    negatable: sql.Composable


class _PredicateRenderer:
    """Render a validated predicate while allocating unique bound parameters."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}
        self._key_index = 0
        self._value_index = 0

    def render(self, predicate: FilterPredicate) -> _RenderedPredicate:
        if isinstance(predicate, FieldPredicate):
            return self._render_field(predicate)

        if isinstance(predicate, (AndPredicate, OrPredicate)):
            operator = " AND " if isinstance(predicate, AndPredicate) else " OR "
            children = [self.render(clause) for clause in predicate.clauses]
            return _RenderedPredicate(
                positive=self._join([child.positive for child in children], operator),
                negatable=self._join([child.negatable for child in children], operator),
            )

        if isinstance(predicate, NotPredicate):
            child = self.render(predicate.clause)
            rendered = sql.SQL("NOT ({})").format(child.negatable)
            return _RenderedPredicate(positive=rendered, negatable=rendered)

        raise InvalidFilterError(
            backend="neon",
            message=f"unexpected predicate type '{type(predicate).__name__}'",
            predicate=predicate,
        )

    def _render_field(self, predicate: FieldPredicate) -> _RenderedPredicate:
        key = self._bind_key(predicate.field)
        op = predicate.op

        if op in {"eq", "ne"}:
            scalar_type = _scalar_json_type(predicate.value, predicate)
            value = self._bind_value(predicate.value)
            containment = _scalar_containment(key, value, scalar_type)
            if op == "ne":
                rendered = sql.SQL("({}) IS NOT TRUE").format(containment)
                return _RenderedPredicate(positive=rendered, negatable=rendered)
            return _RenderedPredicate(
                positive=containment,
                negatable=_guarded_containment(key, scalar_type, containment),
            )

        if op in RANGE_OPS:
            if not _is_number(predicate.value):
                raise InvalidFilterError(
                    backend="neon",
                    message=f"field operator '{op}' requires a numeric value",
                    predicate=predicate,
                )
            value = self._bind_value(predicate.value)
            rendered = sql.SQL(
                "CASE WHEN jsonb_typeof(metadata -> {}) = 'number' "
                "THEN (metadata ->> {})::numeric {} {}::numeric ELSE NULL END"
            ).format(key, key, sql.SQL(_RANGE_SQL_OPERATOR[op]), value)
            return _RenderedPredicate(positive=rendered, negatable=rendered)

        if op in LIST_OPS:
            values, scalar_type = _homogeneous_list(predicate.value, predicate)
            if op == "in":
                return self._render_in(key, values, scalar_type)
            if op == "contains_any":
                return self._render_contains_any(key, values, scalar_type)
            return self._render_contains_all(key, values, scalar_type)

        raise UnsupportedFilterError(
            backend="neon",
            message=f"field operator '{op}' has no Neon SQL mapping",
            predicate=predicate,
        )

    def _render_in(
        self,
        key: sql.Placeholder,
        values: list[object],
        scalar_type: ScalarJsonType | None,
    ) -> _RenderedPredicate:
        if not values:
            return _RenderedPredicate(
                positive=sql.SQL("FALSE"),
                negatable=_guarded_empty_scalar_list(key),
            )

        atoms = [_scalar_containment(key, self._bind_value(value), scalar_type) for value in values]
        rendered = self._join(atoms, " OR ")
        return _RenderedPredicate(
            positive=rendered,
            negatable=_guarded_containment(key, scalar_type, rendered),
        )

    def _render_contains_any(
        self,
        key: sql.Placeholder,
        values: list[object],
        scalar_type: ScalarJsonType | None,
    ) -> _RenderedPredicate:
        if not values:
            return _RenderedPredicate(
                positive=sql.SQL("FALSE"),
                negatable=_guarded_array_expression(key, sql.SQL("FALSE")),
            )

        atoms = [_array_atom(key, self._bind_value(value), scalar_type) for value in values]
        rendered = self._join(atoms, " OR ")
        return _RenderedPredicate(
            positive=rendered,
            negatable=_guarded_array_expression(key, rendered),
        )

    def _render_contains_all(
        self,
        key: sql.Placeholder,
        values: list[object],
        scalar_type: ScalarJsonType | None,
    ) -> _RenderedPredicate:
        elements = [_to_jsonb(self._bind_value(value), scalar_type) for value in values]
        array = sql.SQL("jsonb_build_array({})").format(sql.SQL(", ").join(elements))
        rendered = sql.SQL("metadata @> jsonb_build_object({}::text, {})").format(key, array)
        return _RenderedPredicate(
            positive=rendered,
            negatable=_guarded_array_expression(key, rendered),
        )

    def _bind_key(self, key: str) -> sql.Placeholder:
        name = f"k{self._key_index}"
        self._key_index += 1
        self.params[name] = key
        return sql.Placeholder(name)

    def _bind_value(self, value: object) -> sql.Placeholder:
        name = f"v{self._value_index}"
        self._value_index += 1
        self.params[name] = value
        return sql.Placeholder(name)

    @staticmethod
    def _join(parts: list[sql.Composable], separator: str) -> sql.Composable:
        return sql.SQL("({})").format(sql.SQL(separator).join(parts))


class _SingleLeafRenderer(_PredicateRenderer):
    """Render one field leaf with stable truth-table parameter naming.

    Reuses the type- and arity-aware ``_render_field`` machinery unchanged, only
    overriding the binder: the sole key binds as ``%(k)s`` and values as ``%(v)s``
    (single-value ops) or ``%(v0)s``/``%(v1)s``… (list ops), which is the naming the
    positive SQL forms use. And/Or/Not trees keep the indexed
    naming via :class:`_PredicateRenderer`.

    Args:
        single_value: True for ops with one value (eq/ne/range), so it binds ``%(v)s``.
    """

    def __init__(self, *, single_value: bool) -> None:
        super().__init__()
        self._single_value = single_value

    def _bind_key(self, key: str) -> sql.Placeholder:
        self.params["k"] = key
        return sql.Placeholder("k")

    def _bind_value(self, value: object) -> sql.Placeholder:
        if self._single_value:
            self.params["v"] = value
            return sql.Placeholder("v")
        name = f"v{self._value_index}"
        self._value_index += 1
        self.params[name] = value
        return sql.Placeholder(name)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_json_type(value: object, predicate: FieldPredicate) -> ScalarJsonType:
    if isinstance(value, bool):
        return "boolean"
    if _is_number(value):
        return "number"
    if isinstance(value, str):
        return "text"
    raise InvalidFilterError(
        backend="neon",
        message="eq/ne require a text, numeric, or boolean scalar",
        predicate=predicate,
    )


def _homogeneous_list(
    value: object,
    predicate: FieldPredicate,
) -> tuple[list[object], ScalarJsonType | None]:
    if not isinstance(value, list):
        raise InvalidFilterError(
            backend="neon",
            message=f"field operator '{predicate.op}' requires a list",
            predicate=predicate,
        )
    if not value:
        return [], None

    scalar_types = [_scalar_json_type(item, predicate) for item in value]
    if any(scalar_type != scalar_types[0] for scalar_type in scalar_types[1:]):
        raise InvalidFilterError(
            backend="neon",
            message=f"field operator '{predicate.op}' requires a homogeneous list",
            predicate=predicate,
        )
    return list(value), scalar_types[0]


def _to_jsonb(
    value: sql.Placeholder,
    scalar_type: ScalarJsonType | None,
) -> sql.Composable:
    if scalar_type is None:
        raise AssertionError("non-empty values require a scalar type")
    return sql.SQL("to_jsonb({}::{})").format(
        value,
        sql.SQL(CAST_BY_SCALAR_TYPE[scalar_type]),
    )


def _scalar_containment(
    key: sql.Placeholder,
    value: sql.Placeholder,
    scalar_type: ScalarJsonType | None,
) -> sql.Composable:
    # The key param is cast ``::text`` — jsonb_build_object is VARIADIC "any", which
    # cannot infer a bound param's type, so an uncast key errors on live Postgres.
    return sql.SQL("metadata @> jsonb_build_object({}::text, {})").format(
        key,
        _to_jsonb(value, scalar_type),
    )


def _array_atom(
    key: sql.Placeholder,
    value: sql.Placeholder,
    scalar_type: ScalarJsonType | None,
) -> sql.Composable:
    return sql.SQL("metadata @> jsonb_build_object({}::text, jsonb_build_array({}))").format(
        key, _to_jsonb(value, scalar_type)
    )


def _guarded_containment(
    key: sql.Placeholder,
    scalar_type: ScalarJsonType | None,
    inner: sql.Composable,
) -> sql.Composable:
    if scalar_type is None:
        raise AssertionError("non-empty values require a scalar type")
    return sql.SQL(
        "CASE WHEN NOT jsonb_exists(metadata, {}) THEN NULL "
        "WHEN jsonb_typeof(metadata -> {}) <> {} THEN NULL ELSE {} END"
    ).format(
        key,
        key,
        sql.Literal(JSON_TYPEOF_BY_SCALAR_TYPE[scalar_type]),
        inner,
    )


def _guarded_array_expression(
    key: sql.Placeholder,
    inner: sql.Composable,
) -> sql.Composable:
    return sql.SQL(
        "CASE WHEN NOT jsonb_exists(metadata, {}) THEN NULL "
        "WHEN jsonb_typeof(metadata -> {}) <> 'array' THEN NULL ELSE {} END"
    ).format(key, key, inner)


def _guarded_empty_scalar_list(key: sql.Placeholder) -> sql.Composable:
    return sql.SQL(
        "CASE WHEN NOT jsonb_exists(metadata, {}) THEN NULL "
        "WHEN jsonb_typeof(metadata -> {}) = 'null' THEN NULL ELSE FALSE END"
    ).format(key, key)


def to_neon_where(
    predicate: FilterPredicate | None,
) -> tuple[sql.Composable, dict[str, object]]:
    """Translate a predicate tree into a ``sql.Composable`` WHERE + bound params.

    The single Composable-producing entrypoint the query layer splices into BOTH
    the vector and BM25 legs via ``NeonClient.*_candidates_sql(where=...)``.
    ``.positive`` is correct for every top-level shape: the renderer already
    substitutes the three-valued guarded form *inside* any ``NotPredicate``, so a
    negated tree returns its fully-negated Composable. ``None`` maps to ``TRUE``.
    Param keys (``k0``/``v0``…) never collide with the candidate binds
    (``vector``/``text``/``top_k``/``bm25_*``).
    """
    _ensure_supported(_NEON_FILTER_CAPABILITIES, predicate)
    if predicate is None:
        return sql.SQL("TRUE"), {}

    renderer = _PredicateRenderer()
    rendered = renderer.render(predicate)
    return rendered.positive, renderer.params


def to_neon_filters(
    predicate: FilterPredicate | None,
) -> tuple[str, dict[str, object]]:
    """Translate a predicate AST into parameterized Neon WHERE SQL as a ``str``.

    The string form of :func:`to_neon_where`. The returned SQL contains named
    psycopg placeholders for every metadata key and value; caller input is
    returned only in ``params`` and is never composed into SQL text.
    """
    where, params = to_neon_where(predicate)
    return where.as_string(), params


def predicate_to_sql(
    predicate: FilterPredicate | None,
) -> tuple[str, dict[str, object]]:
    """Return one field leaf's positive SQL + bound params (type- and arity-directed).

    Rendered through the SAME :class:`_PredicateRenderer` machinery as
    :func:`to_neon_where` (via :class:`_SingleLeafRenderer`), so the cast follows
    the VALUE's JSON type (``eq("x")`` -> ``::text``, ``eq(5)`` -> ``::numeric``,
    ``eq(True)`` -> ``::boolean``) and the list ops emit exactly one placeholder per
    element (empty ``in``/``contains_any`` collapse to ``FALSE``). Only the binder
    naming differs from the tree path — ``%(k)s`` for the field, ``%(v)s`` for a
    single value, ``%(v0)s``/``%(v1)s``… for lists. Values and keys are bound
    parameters, never interpolated.
    Validation (range->non-bool number, list->homogeneous, eq/ne->single scalar) and
    the capability gate raise ``InvalidFilterError``/``UnsupportedFilterError``. For
    And/Or/Not trees use :func:`to_neon_where` / :func:`to_neon_filters`.
    """
    if not isinstance(predicate, FieldPredicate):
        raise InvalidFilterError(
            backend="neon",
            message="predicate_to_sql renders a single field leaf; compose "
            "And/Or/Not via to_neon_where",
            predicate=predicate,
        )
    _ensure_supported(_NEON_FILTER_CAPABILITIES, predicate)
    renderer = _SingleLeafRenderer(single_value=predicate.op not in LIST_OPS)
    rendered = renderer.render(predicate)
    return rendered.positive.as_string(), renderer.params
