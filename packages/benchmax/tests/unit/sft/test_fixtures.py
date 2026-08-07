"""Golden-fixture contract tests for `benchmax-sft-v1`.

These fixtures are the cross-repository contract: the Castform trainer consumes
the same files through the pinned submodule. Canonical outputs are compared
byte-for-byte and invalid diagnostics as exact ordered sequences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmax.sft import SftDataset, SftDatasetError

FIXTURES = Path(__file__).parents[2] / "fixtures" / "sft_v1"
VALID_CASES = sorted(
    path.name.removesuffix(".input.jsonl") for path in (FIXTURES / "valid").glob("*.input.jsonl")
)
INVALID_CASES = sorted(
    path.name.removesuffix(".input.jsonl") for path in (FIXTURES / "invalid").glob("*.input.jsonl")
)


def test_fixture_layout_is_complete() -> None:
    assert VALID_CASES, "no valid fixtures found"
    assert INVALID_CASES, "no invalid fixtures found"
    for case in VALID_CASES:
        assert (FIXTURES / "valid" / f"{case}.canonical.jsonl").is_file()
    expected = json.loads((FIXTURES / "invalid" / "expected_issues.json").read_text("utf-8"))
    assert sorted(expected) == INVALID_CASES


@pytest.mark.parametrize("case", VALID_CASES)
def test_valid_fixture_canonicalizes_byte_for_byte(case: str) -> None:
    dataset = SftDataset.from_jsonl(FIXTURES / "valid" / f"{case}.input.jsonl")
    expected = (FIXTURES / "valid" / f"{case}.canonical.jsonl").read_bytes()
    assert dataset.to_jsonl_bytes() == expected


@pytest.mark.parametrize("case", VALID_CASES)
def test_canonical_fixture_is_a_fixed_point(case: str) -> None:
    """Re-validating canonical bytes must reproduce them exactly."""

    path = FIXTURES / "valid" / f"{case}.canonical.jsonl"
    dataset = SftDataset.from_jsonl(path)
    assert dataset.to_jsonl_bytes() == path.read_bytes()


@pytest.mark.parametrize("case", VALID_CASES)
def test_valid_fixture_from_rows_matches_from_jsonl(case: str) -> None:
    from_file = SftDataset.from_jsonl(FIXTURES / "valid" / f"{case}.input.jsonl")
    from_rows = SftDataset.from_rows(list(from_file.rows))
    assert from_rows.to_jsonl_bytes() == from_file.to_jsonl_bytes()


@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_fixture_reports_exact_ordered_issues(case: str) -> None:
    expected = json.loads((FIXTURES / "invalid" / "expected_issues.json").read_text("utf-8"))
    with pytest.raises(SftDatasetError) as excinfo:
        SftDataset.from_jsonl(FIXTURES / "invalid" / f"{case}.input.jsonl")
    assert [issue.as_dict() for issue in excinfo.value.issues] == expected[case]
