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


# Each case: (name, seed_messages, task, expected_hex)
GOLDEN_CASES = [
    (
        "bare_single_user",
        [{"role": "user", "content": "Hello"}],
        None,
        "0bbb7577dbc0523b59e954d128ef3b112645477ae2d9081b0007189090421b8f",
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
        "9bd75cd9d5e39c53f02763adc43062bd80542f5c0b6b851d317f5a59dacefe05",
    ),
    (
        "task_none",
        [{"role": "user", "content": "x"}],
        None,
        "8b39309c8f43c56551281509f9e8f76973f64f94ef98afd7826a9f6eb161a8ca",
    ),
    (
        "task_empty_dict",
        [{"role": "user", "content": "x"}],
        {},
        "839198565d0c3acd3151efc10c9e65a2dac911b06f8eb199f2e107b567c23e14",
    ),
    (
        "unicode",
        [{"role": "user", "content": "你好 émoji 🚀"}],
        {"locale": "zh-CN"},
        "a50f4c4ad3fd5c6838098fe4b09e6b9a24a875a4c9c4ee717d304a53ebe31873",
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
        "c1d79889b683b3d5129d07a7b68bce827fbcae50437f164e76769667145b5b1f",
    ),
]


@pytest.mark.parametrize("name,seed_messages,task,expected", GOLDEN_CASES)
def test_golden_hash(name: str, seed_messages, task, expected: str) -> None:
    assert canonical_example_id(seed_messages, task) == expected


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
