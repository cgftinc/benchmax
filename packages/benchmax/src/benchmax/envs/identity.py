"""Canonical identity for JSON-backed environment examples."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

_JS_MAX_SAFE_INT = 2**53 - 1
_JS_MIN_SAFE_INT = -(2**53 - 1)


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("strings used for identity must be valid UTF-8") from exc
        return value
    if isinstance(value, int):
        if not _JS_MIN_SAFE_INT <= value <= _JS_MAX_SAFE_INT:
            raise ValueError("integers used for identity must fit in JavaScript's safe range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers cannot be used for identity")
        if value == 0:
            return 0
        if value.is_integer():
            return _normalize(int(value))
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys used for identity must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    raise ValueError(f"{type(value).__name__} is not JSON-canonicalizable")


def canonical_example_id(value: Any) -> str:
    """Hash JSON semantics independently of key order and filesystem location."""

    serialized = json.dumps(
        {"v": 1, "value": _normalize(value)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["canonical_example_id"]
