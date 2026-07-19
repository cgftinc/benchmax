"""Ergonomic helpers for building filter predicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .search_types import AndPredicate, FieldPredicate, FilterPredicate, NotPredicate, OrPredicate


@dataclass(frozen=True)
class FieldRef:
    """Field predicate builder for one metadata field."""

    name: str

    def eq(self, value: Any) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="eq", value=value)

    def in_(self, values: list[Any]) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="in", value=values)

    def gte(self, value: Any) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="gte", value=value)

    def lte(self, value: Any) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="lte", value=value)

    def contains_any(self, values: list[Any]) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="contains_any", value=values)

    def contains_all(self, values: list[Any]) -> FieldPredicate:
        return FieldPredicate(field=self.name, op="contains_all", value=values)


def field(name: str) -> FieldRef:
    """Create a field builder."""
    return FieldRef(name=name)


def f(name: str) -> FieldRef:
    """Short alias for `field(name)`."""
    return field(name)


def all_of(*clauses: FilterPredicate) -> AndPredicate:
    """Combine predicates with logical AND."""
    return AndPredicate(clauses=tuple(clauses))


def any_of(*clauses: FilterPredicate) -> OrPredicate:
    """Combine predicates with logical OR."""
    return OrPredicate(clauses=tuple(clauses))


def not_(clause: FilterPredicate) -> NotPredicate:
    """Negate one predicate."""
    return NotPredicate(clause=clause)
