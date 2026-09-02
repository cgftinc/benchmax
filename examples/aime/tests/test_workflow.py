from __future__ import annotations

from types import SimpleNamespace

import main as example
import pytest
from benchmax.envs.harbor import ModalCredentials

CREDENTIAL_ARGS = [
    "--modal-token-id",
    "modal-id",
    "--modal-token-secret",
    "modal-secret",
]


def test_launch_reuses_the_assets_that_were_validated(monkeypatch) -> None:
    bundled_environment = object()
    uploaded_assets = SimpleNamespace(
        env_cls_path="envs/test/env-cls.pkl",
        env_metadata_path="envs/test/env-metadata.json",
        dataset_path=None,
    )
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(example, "generate_data", lambda **kwargs: None)
    monkeypatch.setattr(
        example,
        "dump_bundle",
        lambda cls, *, constructor_args, pip_dependencies, local_modules: (
            calls.append(("bundle", constructor_args)) or bundled_environment
        ),
    )
    monkeypatch.setattr(
        example,
        "upload_assets",
        lambda **kwargs: calls.append(("upload", kwargs)) or uploaded_assets,
    )
    monkeypatch.setattr(
        example,
        "validate",
        lambda env, received: calls.append(("validate", received)) or report,
    )
    monkeypatch.setattr(
        example,
        "launch",
        lambda received, **kwargs: calls.append(("launch", (received, kwargs))) or "run-id",
    )
    monkeypatch.setattr(example, "ensure_session", lambda: None)

    assert example.main(["launch", "--yes", *CREDENTIAL_ARGS]) == 0
    assert calls == [
        (
            "bundle",
            {
                "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
                "sandbox_provider": "modal",
            },
        ),
        ("upload", {"bundle": bundled_environment, "run_name": "aime-modal"}),
        ("validate", uploaded_assets),
        (
            "launch",
            (
                uploaded_assets,
                {"assume_yes": True, "run_name": "aime-modal"},
            ),
        ),
    ]


def test_data_does_not_require_sandbox_credentials() -> None:
    assert example.main(["data"]) == 0


def test_validate_requires_selected_provider_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="Modal requires"):
        example.main(["validate"])

    assert "generating data" in capsys.readouterr().out


def test_print_validation_surfaces_contract_diagnostics(capsys) -> None:
    example._print_validation(
        SimpleNamespace(
            static_warnings={"agent.kwargs.max_tokens": "output cap may be clamped"},
            static_errors={"agent.kwargs.temperature": "trainer-owned"},
            local_warnings={"local-1": ["history not exercised"]},
            remote_warnings={},
            local={},
            remote=None,
            local_errors={},
            remote_errors={},
            ok=False,
        )
    )

    output = capsys.readouterr().out
    assert "⚠️ static agent.kwargs.max_tokens: output cap may be clamped" in output
    assert "⚠️ local local-1: history not exercised" in output
    assert "❌ static agent.kwargs.temperature: trainer-owned" in output
