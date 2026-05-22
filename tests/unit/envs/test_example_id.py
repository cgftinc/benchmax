"""Golden-hash tests for ``canonical_example_id``.

These pin the byte-level output of the hash function. The same expected
hashes are checked by the TypeScript port's parity test in
``platform-service``. If you change any expected value here, you MUST
update both sides and bump the ``v`` payload tag in
``benchmax.envs.example_id`` — existing rollouts in the database key on the
old hash and would silently misgroup.
"""
from __future__ import annotations

import math

import pytest

from benchmax.envs.example_id import canonical_example_id


# Each case: (name, prompt_messages, task, expected_hex)
GOLDEN_CASES = [
    (
        "bare_single_user",
        [{"role": "user", "content": "Hello"}],
        None,
        "04d553add09f1f28f35139014176fe64cf3cbbecb0de22e99370c49b3e35088b",
    ),
    (
        "multi_turn",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "Solve 2+2"},
        ],
        {"ground_truth": "4"},
        "dcf4e5fcce040c41d527b27c3c6e9c4f1df0745c7f48c3f11b0e21a4500ea5fa",
    ),
    (
        "task_none",
        [{"role": "user", "content": "x"}],
        None,
        "e4d534ac3d401153ec029cd68bae30cdcbe03314cc69296ffc48ce4458dddff8",
    ),
    (
        "task_empty_dict",
        [{"role": "user", "content": "x"}],
        {},
        "5d6763943667224756336ff9ffc2eff2e9267ac8a815745db54a75373073134a",
    ),
    (
        "unicode",
        [{"role": "user", "content": "你好 émoji 🚀"}],
        {"locale": "zh-CN"},
        "d1f6a202e088221a60711f105c02e8cfce9da69c14ef479efcbf7416d63280b9",
    ),
    (
        "nested_task",
        [{"role": "user", "content": "q"}],
        {
            "ground_truth": "ans",
            "tools_enabled": True,
            "max_steps": 5,
            "scoring": {"weights": {"correctness": 1, "format": 0}},
        },
        "cee4165ba98d0c7b7ac8b12ce2587bdad88e38ef63d2235a071423fb3968915d",
    ),
]


@pytest.mark.parametrize("name,prompt_messages,task,expected", GOLDEN_CASES)
def test_golden_hash(name: str, prompt_messages, task, expected: str) -> None:
    assert canonical_example_id(prompt_messages, task) == expected


def test_none_task_differs_from_empty_task() -> None:
    a = canonical_example_id([{"role": "user", "content": "x"}], None)
    b = canonical_example_id([{"role": "user", "content": "x"}], {})
    assert a != b


def test_key_order_independent() -> None:
    msg = [{"role": "user", "content": "q"}]
    a = canonical_example_id(msg, {"a": 1, "b": 2, "c": 3})
    b = canonical_example_id(msg, {"c": 3, "b": 2, "a": 1})
    assert a == b


def test_int_vs_float_normalized() -> None:
    """Integer-valued floats hash the same as ints (JS Number has no int/float
    distinction, so the algorithm normalizes both sides)."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": 1}) == canonical_example_id(msg, {"x": 1.0})


def test_bool_not_int() -> None:
    """Booleans must stay as JSON booleans, not be coerced to 0/1."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": True}) != canonical_example_id(msg, {"x": 1})
    assert canonical_example_id(msg, {"x": False}) != canonical_example_id(msg, {"x": 0})


def test_negative_zero_normalized() -> None:
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": -0.0}) == canonical_example_id(msg, {"x": 0})


def test_nan_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="NaN/Inf"):
        canonical_example_id(msg, {"x": math.nan})


def test_inf_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="NaN/Inf"):
        canonical_example_id(msg, {"x": math.inf})


def test_returns_64_hex_chars() -> None:
    h = canonical_example_id([{"role": "user", "content": "x"}], None)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_non_string_dict_key_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="dict keys must be str"):
        canonical_example_id(msg, {1: "x"})
    with pytest.raises(ValueError, match="dict keys must be str"):
        canonical_example_id(msg, {True: "x"})  # bool keys diverge between Py/JS


def test_unsupported_type_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="not JSON-canonicalizable"):
        canonical_example_id(msg, {"x": b"bytes"})
    with pytest.raises(ValueError, match="not JSON-canonicalizable"):
        canonical_example_id(msg, {"x": {1, 2, 3}})  # set


def test_integer_above_js_safe_int_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="MAX_SAFE_INTEGER"):
        canonical_example_id(msg, {"x": 2**53})
    with pytest.raises(ValueError, match="MAX_SAFE_INTEGER"):
        canonical_example_id(msg, {"x": -(2**53)})


def test_integer_at_js_safe_int_boundary_ok() -> None:
    msg = [{"role": "user", "content": "q"}]
    canonical_example_id(msg, {"x": 2**53 - 1})
    canonical_example_id(msg, {"x": -(2**53 - 1)})


def test_integer_valued_float_above_js_safe_int_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="MAX_SAFE_INTEGER"):
        canonical_example_id(msg, {"x": float(2**60)})


def test_lone_surrogate_rejected() -> None:
    msg = [{"role": "user", "content": "q"}]
    with pytest.raises(ValueError, match="surrogates"):
        canonical_example_id(msg, {"x": "\ud800"})
