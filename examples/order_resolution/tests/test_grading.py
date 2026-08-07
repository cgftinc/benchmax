"""Pure exact-state grading tests."""

from __future__ import annotations

import copy

from order_resolution.fixtures import (
    DEFAULT_SEED,
    generate_data,
    initial_snapshot,
    oracle_after_snapshot,
)
from order_resolution.grading import grade_snapshots


def _row() -> dict:
    return next(
        row
        for row in generate_data(DEFAULT_SEED).train
        if row["action_family"] == "cancel_item" and row["outcome_class"] == "execute"
    )


def _grade(row: dict, *, after: dict, required: list[dict] | None = None, forbidden=None):
    return grade_snapshots(
        before=initial_snapshot(row),
        after=after,
        required=row["required_state"] if required is None else required,
        forbidden=row["forbidden_state"] if forbidden is None else forbidden,
        expected_disposition=row["expected_disposition"],
        expected_reply=row["expected_reply"],
    )


def test_exact_oracle_state_receives_sparse_success() -> None:
    row = _row()
    grade = _grade(row, after=oracle_after_snapshot(row))

    assert grade.task_success == 1.0
    assert grade.required_state_fraction == 1.0
    assert grade.forbidden_mutation == 0.0
    assert grade.failures == ()


def test_missing_required_state_zeroes_reward() -> None:
    row = _row()
    after = oracle_after_snapshot(row)
    grade = _grade(
        row,
        after=after,
        required=[{"path": "support_case.outcome_code", "op": "eq", "value": "WRONG"}],
    )

    assert grade.task_success == 0.0
    assert grade.required_state_fraction == 0.0
    assert grade.failures[0].startswith("required support_case.outcome_code")


def test_forbidden_mutation_zeroes_reward() -> None:
    row = _row()
    after = oracle_after_snapshot(row)
    grade = _grade(
        row,
        after=after,
        forbidden=[{"path": "support_case.disposition", "op": "unchanged"}],
    )

    assert grade.task_success == 0.0
    assert grade.forbidden_mutation == 1.0
    assert "forbidden mutation at support_case.disposition" in grade.failures


def test_wrong_disposition_zeroes_reward() -> None:
    row = _row()
    after = oracle_after_snapshot(row)
    after["support_case"]["disposition"] = "cannot_complete"
    grade = _grade(row, after=after)

    assert grade.task_success == 0.0
    assert grade.correct_disposition == 0.0


def test_reply_must_be_exact_and_unique() -> None:
    row = _row()
    duplicate = oracle_after_snapshot(row)
    duplicate["reply_count"] = 2
    missing = copy.deepcopy(duplicate)
    missing["reply"] = None
    missing["reply_count"] = 0

    duplicate_grade = _grade(row, after=duplicate)
    missing_grade = _grade(row, after=missing)

    assert duplicate_grade.task_success == 0.0
    assert duplicate_grade.structured_reply_correct == 0.0
    assert missing_grade.task_success == 0.0
    assert missing_grade.structured_reply_correct == 0.0
