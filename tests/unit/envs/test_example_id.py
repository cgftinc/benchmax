"""Golden-hash tests for ``canonical_example_id``.

These pin the byte-level hash output. Changing any expected value REQUIRES
bumping the ``v`` payload tag in ``benchmax.envs.example_id`` — existing
rollouts key on the old hash and would silently misgroup.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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


# Loader parity — the regression v:3 prevents. Trainer loads via HuggingFace
# datasets (Arrow, schema-unifies columns → fills absent keys with null);
# rollout-service uses json.loads (keeps only present keys). Pre-v:3 these
# hashed the same example differently. These pin that they now agree.

# Heterogeneous metadata: row 1 omits ``h3``, only the last carries ``h4`` —
# Arrow injects ``h3: None`` AND ``h4: None`` into every other row.
_HETEROGENEOUS_ROWS = [
    {
        "question": "q0",
        "answer": "a0",
        "reference_chunks": [
            {
                "id": "c0",
                "metadata": {"h2": "H2a", "h3": "H3a", "file": "a.md", "index": 0},
            }
        ],
    },
    {
        "question": "q1",
        "answer": "a1",
        "reference_chunks": [
            {"id": "c1", "metadata": {"h2": "H2b", "file": "b.md", "index": 1}}
        ],
    },
    {
        "question": "q2",
        "answer": "a2",
        "reference_chunks": [
            {
                "id": "c2",
                "metadata": {"index": 2, "file": "c.md", "h3": "H3c", "h2": "H2c"},
            }
        ],
    },
    {
        "question": "q3",
        "answer": "a3",
        "reference_chunks": [
            {
                "id": "c3",
                "metadata": {"h2": "H2d", "h3": None, "file": "d.md", "index": 3},
            }
        ],
    },
    {
        "question": "q4",
        "answer": "a4",
        "reference_chunks": [
            {
                "id": "c5",
                "metadata": {
                    "h2": "H2g",
                    "h3": "H3g",
                    "h4": "H4g",
                    "file": "g.md",
                    "index": 6,
                },
            }
        ],
    },
]


def _project(row: dict) -> tuple[list, dict]:
    """Mirror ``SearchEnv.dataset_preprocess`` — reference_chunks pass through."""
    pm = [{"role": "user", "content": row.get("question", "")}]
    task = {
        "question": row.get("question", ""),
        "ground_truth": row.get("answer"),
        "reference_chunks": row.get("reference_chunks", []),
    }
    return pm, task


def test_loader_parity_datasets_vs_json(tmp_path: Path) -> None:
    """datasets/Arrow and json.loads yield identical ids per row (fails pre-v:3)."""
    load_dataset = pytest.importorskip("datasets").load_dataset

    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _HETEROGENEOUS_ROWS) + "\n")

    json_rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    arrow_rows = [
        dict(r) for r in load_dataset("json", data_files=str(path), split="train")
    ]

    assert len(json_rows) == len(arrow_rows) == len(_HETEROGENEOUS_ROWS)
    for i, (jr, ar) in enumerate(zip(json_rows, arrow_rows)):
        jpm, jt = _project(jr)
        apm, at = _project(ar)
        assert canonical_example_id(jpm, jt) == canonical_example_id(apm, at), (
            f"row {i}: loaders disagree on canonical_example_id"
        )


def test_loader_mixed_numeric_string_column_fails_loud(tmp_path: Path) -> None:
    """The residual Arrow coercion (number→string column) fails loud at load,
    so it can't become a silent id skew like the null case did."""
    datasets = pytest.importorskip("datasets")

    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps({"x": 1}) + "\n" + json.dumps({"x": "a"}) + "\n")

    with pytest.raises(Exception):  # noqa: B017 — pyarrow.lib.ArrowInvalid, surfaced via datasets
        datasets.load_dataset("json", data_files=str(path), split="train")
