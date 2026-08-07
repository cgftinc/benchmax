"""Strict JSON parsing, tree checking, and canonical encoding for `benchmax-sft-v1`.

This module owns the byte-level half of the format contract:

- parsing rejects duplicate object keys and the non-finite literals ``NaN`` /
  ``Infinity`` / ``-Infinity``;
- tree checking rejects non-JSON Python types (including subclasses), non-string
  object keys, lone surrogates in any string, non-finite floats (covering
  overflow like ``1e999``), and container nesting deeper than
  :data:`MAX_JSON_DEPTH` levels;
- canonical encoding is UTF-8 without a BOM, ``ensure_ascii=False``,
  ``allow_nan=False``, sorted keys, and compact separators, so equivalent
  objects always produce identical bytes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

__all__ = [
    "MAX_JSON_DEPTH",
    "StrictJsonError",
    "canonical_json_bytes",
    "check_json_tree",
    "parse_strict_json",
]

MAX_JSON_DEPTH = 64

_TOO_DEEP = f"JSON nesting exceeds {MAX_JSON_DEPTH} levels"


class StrictJsonError(ValueError):
    """Raised by :func:`parse_strict_json` for text violating the strict rules."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(constant: str) -> object:
    raise StrictJsonError(f"non-finite number literal {constant!r} is not allowed")


def parse_strict_json(text: str) -> object:
    """Parse one JSON document, rejecting duplicate keys and non-finite literals.

    Raises :class:`StrictJsonError` with a human-readable reason on any
    violation, including malformed JSON. Overlong nesting that overflows the
    parser is reported as exceeding :data:`MAX_JSON_DEPTH`; the precise depth
    check for parseable documents lives in :func:`check_json_tree`.
    """

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except StrictJsonError:
        raise
    except RecursionError:
        raise StrictJsonError(_TOO_DEEP) from None
    except ValueError as error:
        raise StrictJsonError(f"invalid JSON: {error}") from None


def _has_lone_surrogate(text: str) -> bool:
    return any("\ud800" <= char <= "\udfff" for char in text)


def check_json_tree(
    value: object,
    location: str,
    emit: Callable[[str, str], None],
) -> bool:
    """Walk ``value`` in document order, reporting strict-JSON violations.

    ``emit`` receives ``(location, message)`` for every violation. Returns True
    when the tree is fully canonical-encodable. Containers past the depth limit
    are reported once and not descended into.
    """

    clean = True
    # Depth-first pre-order via an explicit stack; children pushed reversed so
    # they pop in document order. Root containers sit at depth 1.
    stack: list[tuple[object, str, int]] = [(value, location, 1)]
    while stack:
        node, node_location, depth = stack.pop()
        node_type = type(node)
        if node_type is dict:
            if depth > MAX_JSON_DEPTH:
                emit(node_location, _TOO_DEEP)
                clean = False
                continue
            children: list[tuple[object, str, int]] = []
            for key, child in node.items():
                if type(key) is not str:
                    emit(node_location, f"object key {key!r} must be a string")
                    clean = False
                    continue
                if _has_lone_surrogate(key):
                    emit(node_location, "object key contains a lone surrogate")
                    clean = False
                    continue
                children.append((child, f"{node_location}.{key}", depth + 1))
            stack.extend(reversed(children))
        elif node_type is list:
            if depth > MAX_JSON_DEPTH:
                emit(node_location, _TOO_DEEP)
                clean = False
                continue
            stack.extend(
                (child, f"{node_location}[{index}]", depth + 1)
                for index, child in reversed(list(enumerate(node)))
            )
        elif node_type is str:
            if _has_lone_surrogate(node):
                emit(node_location, "string contains a lone surrogate")
                clean = False
        elif node_type is float:
            if not math.isfinite(node):
                emit(node_location, "number must be finite")
                clean = False
        elif node_type is int or node_type is bool or node is None:
            pass
        else:
            emit(
                node_location,
                f"unsupported type {node_type.__name__}; only JSON object, array, "
                "string, number, boolean, and null values are allowed",
            )
            clean = False
    return clean


def canonical_json_bytes(value: object) -> bytes:
    """Encode a tree accepted by :func:`check_json_tree` as canonical bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
