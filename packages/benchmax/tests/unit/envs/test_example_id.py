from __future__ import annotations

import math

import pytest
from benchmax.envs import canonical_example_id


def test_identity_is_stable_across_mapping_order():
    assert canonical_example_id({"prompt_messages": [], "a": 1, "b": 2}) == (
        canonical_example_id({"b": 2, "a": 1, "prompt_messages": []})
    )


def test_identity_depends_only_on_value_not_source_path():
    row = {"prompt_messages": [{"role": "user", "content": "q"}], "answer": "a"}

    assert canonical_example_id(row) == canonical_example_id(dict(row))


def test_identity_distinguishes_null_from_missing():
    assert canonical_example_id({"x": None}) != canonical_example_id({})


def test_identity_normalizes_json_number_semantics():
    assert canonical_example_id({"x": 1}) == canonical_example_id({"x": 1.0})
    assert canonical_example_id({"x": -0.0}) == canonical_example_id({"x": 0})
    assert canonical_example_id({"x": True}) != canonical_example_id({"x": 1})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_identity_rejects_non_finite_numbers(value: float):
    with pytest.raises(ValueError, match="non-finite"):
        canonical_example_id({"x": value})


def test_identity_rejects_non_json_values():
    with pytest.raises(ValueError, match="not JSON-canonicalizable"):
        canonical_example_id({"x": b"bytes"})


def test_identity_is_sha256_hex():
    value = canonical_example_id({"prompt_messages": []})
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")
