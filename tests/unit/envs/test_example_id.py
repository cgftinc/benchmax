"""Golden-hash tests for ``canonical_example_id``.

These pin the byte-level hash output. Changing any expected value REQUIRES
bumping the ``v`` payload tag in ``benchmax.envs.example_id`` — existing
rollouts key on the old hash and would silently misgroup.
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
        "93a9b71ec42f0fcce151bbdc71ece1fe86bdccedd6090a28b2c81f307115bc36",
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
        "b418bf2def5cef378a7f30ab864c623dbdf211934c4d224612676ca73c3b3e2a",
    ),
    (
        "task_none",
        [{"role": "user", "content": "x"}],
        None,
        "8871e98f9164dbbfd053c9bc2fc8a2f98abcd5633914199fd31db11bb64113ba",
    ),
    (
        "task_empty_dict",
        [{"role": "user", "content": "x"}],
        {},
        "bd1bc06ed235f44cf956940931610ce5059b437b607a4160585cede249eac63a",
    ),
    (
        "unicode",
        [{"role": "user", "content": "你好 émoji 🚀"}],
        {"locale": "zh-CN"},
        "d824c3a7c891e80bc7420c9eb9c7d90e62b95b8acb01e1dcfc3f11cd527a56c1",
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
        "48809c34889a18c84e323c495f33a726be7cfc71d6e863cb5d06513429bb2889",
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


def test_null_dict_key_dropped() -> None:
    """v:3: a None-valued key hashes the same as the key being absent."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": None}) == canonical_example_id(msg, {})
    assert canonical_example_id(msg, {"a": 1, "b": None}) == canonical_example_id(
        msg, {"a": 1}
    )
    assert canonical_example_id(
        msg, {"m": {"h2": "H", "h3": None}}
    ) == canonical_example_id(msg, {"m": {"h2": "H"}})


def test_list_nulls_preserved() -> None:
    """Nulls inside lists are identity-bearing — only dict keys are dropped."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": [1, None, 3]}) != canonical_example_id(
        msg, {"x": [1, 3]}
    )
    # nulls are still stripped within dict elements of a list
    assert canonical_example_id(
        msg, {"x": [{"a": 1, "b": None}]}
    ) == canonical_example_id(msg, {"x": [{"a": 1}]})


def test_int_vs_float_normalized() -> None:
    """Integer-valued floats hash the same as ints (JS Number has no int/float
    distinction, so the algorithm normalizes both sides)."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": 1}) == canonical_example_id(msg, {"x": 1.0})


def test_bool_not_int() -> None:
    """Booleans must stay as JSON booleans, not be coerced to 0/1."""
    msg = [{"role": "user", "content": "q"}]
    assert canonical_example_id(msg, {"x": True}) != canonical_example_id(msg, {"x": 1})
    assert canonical_example_id(msg, {"x": False}) != canonical_example_id(
        msg, {"x": 0}
    )


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

