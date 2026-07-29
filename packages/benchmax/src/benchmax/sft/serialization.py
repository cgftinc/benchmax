"""Canonical byte rendering for a single SFT row.

This lives in its own module so :mod:`benchmax.sft.schema` (which asks "would
this row serialize?") and :mod:`benchmax.sft.dataset` (which refuses to
serialize rows the schema rejects) can both depend on it without depending on
each other. Importing it pulls in nothing beyond the standard library.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_row_bytes(row_data: dict[str, Any]) -> bytes:
    """The exact bytes ``canonical_jsonl`` emits for one row, without the trailing newline.

    Shared with :func:`benchmax.sft.schema.validate_row` and
    :func:`benchmax.sft.validate.validate_sft_dataset` so "is this row
    serializable" is answered by the same code path that actually
    canonicalizes it — ``ensure_ascii=True`` (json.dumps' default) never
    attempts the UTF-8 encode step, so a schema check that only called
    ``json.dumps`` could pass a row containing a lone surrogate that then
    raises ``UnicodeEncodeError`` here. ``allow_nan=False`` rejects
    ``NaN``/``Infinity`` tokens that aren't valid JSON.

    Raises ``TypeError``/``ValueError`` (including ``UnicodeEncodeError``) on a
    row that cannot be rendered, and ``RecursionError`` on one nested deeper
    than the encoder's stack allows. Callers that must not crash on bad input
    catch all three.
    """
    return json.dumps(row_data, ensure_ascii=False, allow_nan=False).encode("utf-8")
