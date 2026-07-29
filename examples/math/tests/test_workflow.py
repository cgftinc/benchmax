from __future__ import annotations

from types import SimpleNamespace

import main
import pytest


def test_validate_action_uploads_and_reuses_the_same_assets(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundled_environment = object()
    assets = object()
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(main, "generate_data", lambda **kwargs: dataset_files)
    monkeypatch.setattr(
        main,
        "dump_bundle",
        lambda *args, **kwargs: bundled_environment,
    )
    monkeypatch.setattr(
        main,
        "upload_assets",
        lambda **kwargs: calls.append(("upload", kwargs)) or assets,
    )
    monkeypatch.setattr(
        main,
        "validate",
        lambda env, received: calls.append(("validate", received)) or report,
    )
    monkeypatch.setattr(main, "ensure_session", lambda: None)

    assert main.main(["validate"]) == 0
    assert calls == [
        (
            "upload",
            {
                "bundle": bundled_environment,
                "dataset_files": dataset_files,
                "run_name": "math",
            },
        ),
        ("validate", assets),
    ]


def test_launch_reuses_the_assets_that_were_validated(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundled_environment = object()
    assets = object()
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(main, "generate_data", lambda **kwargs: dataset_files)
    monkeypatch.setattr(
        main,
        "dump_bundle",
        lambda *args, **kwargs: bundled_environment,
    )
    monkeypatch.setattr(
        main,
        "upload_assets",
        lambda **kwargs: calls.append(("upload", kwargs)) or assets,
    )
    monkeypatch.setattr(
        main,
        "validate",
        lambda env, received: calls.append(("validate", received)) or report,
    )
    monkeypatch.setattr(
        main,
        "launch",
        lambda received, **kwargs: calls.append(("launch", received)) or "run-id",
    )
    monkeypatch.setattr(main, "ensure_session", lambda: None)

    assert main.main(["launch", "--yes"]) == 0
    assert calls == [
        (
            "upload",
            {
                "bundle": bundled_environment,
                "dataset_files": dataset_files,
                "run_name": "math",
            },
        ),
        ("validate", assets),
        ("launch", assets),
    ]


def test_failed_validation_does_not_launch(monkeypatch) -> None:
    assets = object()
    monkeypatch.setattr(main, "generate_data", lambda **kwargs: {})
    monkeypatch.setattr(main, "dump_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "upload_assets", lambda **kwargs: assets)
    monkeypatch.setattr(
        main,
        "validate",
        lambda env, received: SimpleNamespace(ok=False),
    )
    monkeypatch.setattr(
        main,
        "launch",
        lambda *args, **kwargs: pytest.fail("launch should not be called"),
    )
    monkeypatch.setattr(main, "ensure_session", lambda: None)

    assert main.main(["launch", "--yes"]) == 1


def test_validation_scorecard_includes_runtime_errors(capsys) -> None:
    report = SimpleNamespace(
        ok=False,
        local={},
        remote={},
        local_errors={},
        remote_errors={"remote-1": "model failed"},
    )

    main._print_validation(report)

    output = capsys.readouterr().out
    assert "❌ remote remote-1: model failed" in output
    assert "❌ validation failed" in output


def test_validation_scorecard_marks_accepted_outcomes_as_success(capsys) -> None:
    outcome = SimpleNamespace(
        termination_reason="max_turns_exceeded",
        rewards={"correctness": 0.0},
        error=None,
    )
    report = SimpleNamespace(
        ok=True,
        local={"local-1": outcome},
        remote={"remote-1": outcome},
        local_errors={},
        remote_errors={},
    )

    main._print_validation(report)

    output = capsys.readouterr().out
    assert "✅ local local-1: max_turns_exceeded" in output
    assert "✅ remote remote-1: max_turns_exceeded" in output
    assert "✅ validation passed" in output


def test_validation_scorecard_shows_settlement_errors(capsys) -> None:
    outcome = SimpleNamespace(
        termination_reason="max_turns_exceeded",
        rewards={},
        error="KeyError: 'missing ground_truth column'",
    )
    report = SimpleNamespace(
        ok=True,
        local={"local-1": outcome},
        remote=None,
        local_errors={},
        remote_errors={},
    )

    main._print_validation(report)

    output = capsys.readouterr().out
    assert "error=KeyError: 'missing ground_truth column'" in output
