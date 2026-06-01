"""Canonical example identity.

``canonical_example_id(prompt_messages, task)`` returns a SHA-256 hex digest
stable across processes. Identity is computed only here, in Python — both the
trainer and rollout-service hash via this module.

Normalization keeps the digest loader-independent:
- integer-valued floats → int, -0.0 → 0; NaN/Inf rejected.
- dict keys whose value is ``None`` are dropped, so a key absent in one loader
  and present-but-null in another (Arrow schema-unification) hashes the same;
  nulls *inside lists* are kept (length/order are identity).
- ambiguous values rejected: non-str dict keys, ints beyond
  ``Number.MAX_SAFE_INTEGER``, byte strings, lone surrogates, unknown types.
- canonical JSON: sorted keys, no whitespace, no ASCII escaping.

Payload tag ``v:3``. History: v:1→v:2 = the 2026-05 ``seed_messages`` →
``prompt_messages`` rename; v:2→v:3 = drop null-valued dict keys (loader skew).
Older hashes are obsolete.
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
            nx = _normalize(x)
            # Drop null-valued keys (v:3): Arrow fills schema-unified columns
            # with null where a row omits a key, json.loads omits it entirely;
            # stripping makes both loaders agree. List nulls are kept (above).
            if nx is None:
                continue
            out[k] = nx
        return out
    raise ValueError(
        f"type {type(v).__name__} is not JSON-canonicalizable; "
        "supported: None, bool, int (within safe range), float, str, list, dict"
    )


def canonical_example_id(
    prompt_messages: Messages,
    task: dict[str, Any] | None,
) -> str:
    payload = {"v": 3, "prompt_messages": prompt_messages, "task": task}
    serialized = json.dumps(
        _normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_example(
    prompt_messages: Messages,
    task: dict[str, Any] | None = None,
    init_rollout_args: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> Example:
    """Build an :class:`Example` with the canonical id pre-computed.

    If ``system_prompt`` is non-empty, it is prepended to ``prompt_messages``
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
    ``prompt_messages``. Envs that need tool-set sensitivity in their group
    identity should bake a tool-signature string into ``task``.

    Convenience for env authors overriding ``dataset_preprocess``::

        @classmethod
        def dataset_preprocess(cls, row, **_):
            return make_example(
                prompt_messages=[{"role": "user", "content": row["question"]}],
                task={"answer": row["answer"]},
                system_prompt=cls.system_prompt,
            )
    """
    if system_prompt:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            *prompt_messages,
        ]
    return Example(
        id=canonical_example_id(prompt_messages, task),
        prompt_messages=prompt_messages,
        task=task,
        init_rollout_args=init_rollout_args,
    )
