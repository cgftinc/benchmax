"""Canonical example identity.

``canonical_example_id(seed_messages, task)`` returns a SHA-256 hex digest
that is stable across processes and languages: a TypeScript port lives in
``platform-service/src/lib/canonical-example-id.ts`` and is exercised by a
parity test.

Determinism is achieved by:
- normalizing numeric values so JSON output matches between Python and JS
  (JS has no int/float distinction; integer-valued floats are coerced to int,
  -0.0 to 0; NaN/Inf are rejected).
- rejecting values whose JSON serialization diverges between Python and JS:
  non-string dict keys, integers outside JS ``Number.MAX_SAFE_INTEGER``,
  byte strings, lone surrogates, and unknown types.
- emitting canonical JSON with sorted keys, no whitespace, and no ASCII
  escaping (modern JSON.stringify also preserves non-ASCII).

The hash is computed over ``{"v": 1, "seed_messages": ..., "task": ...}`` —
the version tag lets us bump the algorithm later without ambiguity.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from benchmax.envs.types import Example, Messages

_JS_MAX_SAFE_INT = 2**53 - 1  # Number.MAX_SAFE_INTEGER
_JS_MIN_SAFE_INT = -(2**53 - 1)


def _normalize(v: Any) -> Any:
    # bool is a subclass of int; handle it before the int branch so True stays
    # True (and not coerced to 1).
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, int):
        if v > _JS_MAX_SAFE_INT or v < _JS_MIN_SAFE_INT:
            raise ValueError(
                f"integer {v} exceeds JS Number.MAX_SAFE_INTEGER (2^53-1); "
                "would diverge between Python and TypeScript canonical_example_id"
            )
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("NaN/Inf are not allowed in canonical_example_id input")
        if v == 0.0:
            return 0  # collapses -0.0 to 0 (matches JS JSON.stringify)
        if v.is_integer():
            iv = int(v)
            if iv > _JS_MAX_SAFE_INT or iv < _JS_MIN_SAFE_INT:
                raise ValueError(
                    f"integer-valued float {v!r} exceeds JS Number.MAX_SAFE_INTEGER "
                    "(2^53-1); would diverge between Python and TypeScript"
                )
            return iv
        return v
    if isinstance(v, str):
        # Lone surrogates would error or silently differ between encoders.
        try:
            v.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(
                "string contains lone surrogates that cannot be UTF-8 encoded"
            ) from e
        return v
    if isinstance(v, (list, tuple)):
        return [_normalize(x) for x in v]
    if isinstance(v, dict):
        out: dict[str, Any] = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"dict keys must be str for canonical hashing; got {type(k).__name__}"
                )
            out[k] = _normalize(x)
        return out
    raise ValueError(
        f"type {type(v).__name__} is not JSON-canonicalizable; "
        "supported: None, bool, int (within safe range), float, str, list, dict"
    )


def canonical_example_id(
    seed_messages: Messages,
    task: dict[str, Any] | None,
) -> str:
    payload = {"v": 1, "seed_messages": seed_messages, "task": task}
    serialized = json.dumps(
        _normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_example(
    seed_messages: Messages,
    task: dict[str, Any] | None = None,
    init_rollout_args: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> Example:
    """Build an :class:`Example` with the canonical id pre-computed.

    If ``system_prompt`` is non-empty, it is prepended to ``seed_messages``
    as ``{"role": "system", "content": system_prompt}`` so the env's system
    prompt is part of the example's identity. Two envs with the same user
    prompt but different system prompts (e.g. "be concise" vs "be verbose")
    will therefore hash to distinct example_ids — same dataset row, but the
    model is being graded on materially different inputs, so they shouldn't
    collapse into one group.

    Tool definitions (rendered via ``render_tools_prompt``) are NOT included
    in the hash — they're a dynamic property of the env instance that
    requires an async ``list_tools()`` call, which doesn't fit
    ``dataset_preprocess``'s classmethod contract. The trainer renders
    tools into the first system message at LLM-call time without mutating
    ``seed_messages``. Envs that need tool-set sensitivity in their group
    identity should bake a tool-signature string into ``task``.

    Convenience for env authors overriding ``dataset_preprocess``::

        @classmethod
        def dataset_preprocess(cls, row, **_):
            return make_example(
                seed_messages=[{"role": "user", "content": row["question"]}],
                task={"answer": row["answer"]},
                system_prompt=cls.system_prompt,
            )
    """
    if system_prompt:
        seed_messages = [
            {"role": "system", "content": system_prompt},
            *seed_messages,
        ]
    return Example(
        id=canonical_example_id(seed_messages, task),
        seed_messages=seed_messages,
        task=task,
        init_rollout_args=init_rollout_args,
    )
