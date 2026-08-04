from __future__ import annotations

import argparse
from types import SimpleNamespace

import main as example
import pytest
from benchmax.envs.harbor import (
    BundledAgentSource,
    BundledHarborAgent,
    ModalCredentials,
)

CREDENTIAL_ARGS = [
    "--modal-token-id",
    "modal-id",
    "--modal-token-secret",
    "modal-secret",
    "--judge-provider",
    "anthropic",
    "--judge-model",
    "anthropic/claude-sonnet-4-6",
    "--judge-api-key",
    "anthropic-key",
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
    stub_source = BundledAgentSource.from_files({"harvey_agent.py": b"stub"})

    monkeypatch.setattr(example, "_lab_source_bundle", lambda: stub_source)
    monkeypatch.setattr(example, "generate_data", lambda **kwargs: None)
    monkeypatch.setattr(
        example,
        "dump_bundle",
        lambda cls, *, constructor_args, pip_dependencies: (
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
        lambda received, **kwargs: calls.append(("launch", received)) or "run-id",
    )
    monkeypatch.setattr(example, "ensure_session", lambda: None)
    monkeypatch.setattr(example, "_check_judge_credentials", lambda args: None)

    assert example.main(["launch", "--yes", *CREDENTIAL_ARGS]) == 0
    stage, bundle_args = calls[0]
    assert stage == "bundle"
    bundle_args = dict(bundle_args)
    harness = bundle_args.pop("harness")
    assert isinstance(harness, BundledHarborAgent)
    assert harness.source is stub_source
    assert bundle_args == {
        "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
        "verifier_env": {"ANTHROPIC_API_KEY": "anthropic-key"},
        "judge_model": "anthropic/claude-sonnet-4-6",
        "judge_concurrency": 1,
    }
    assert calls[1:] == [
        ("upload", {"bundle": bundled_environment, "run_name": "harvey"}),
        ("validate", uploaded_assets),
        ("launch", uploaded_assets),
    ]


def test_main_requires_credential_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        example.main(["data"])

    stderr = capsys.readouterr().err
    for name in ("--modal-token-id", "--modal-token-secret", "--judge-api-key"):
        assert name in stderr


def test_judge_preflight_rejects_a_bad_key(monkeypatch) -> None:
    response = SimpleNamespace(
        status_code=401,
        text='{"error":{"code":"invalid_api_key"}}',
        is_error=True,
    )
    monkeypatch.setattr(example.httpx, "get", lambda *args, **kwargs: response)

    with pytest.raises(SystemExit) as excinfo:
        example.main(["validate", *CREDENTIAL_ARGS])

    assert "judge preflight: API key rejected (HTTP 401)" in str(excinfo.value)


def test_judge_preflight_probes_the_openai_judge_model(monkeypatch, capsys) -> None:
    probes: list[tuple[str, dict[str, object]]] = []

    def fake_post(url, *, headers, json, timeout):
        probes.append((url, json))
        return SimpleNamespace(status_code=200, text="", is_error=False)

    monkeypatch.setattr(example.httpx, "post", fake_post)
    monkeypatch.setattr(example, "generate_data", lambda **kwargs: None)

    example.main(
        [
            "data",
            "--modal-token-id",
            "modal-id",
            "--modal-token-secret",
            "modal-secret",
            "--judge-provider",
            "openai",
            "--judge-model",
            "openai/gpt-5.4-nano",
            "--judge-api-key",
            "castform-key",
            "--judge-base-url",
            "https://llm.castform.dev/v1",
        ]
    )
    assert probes == []  # the data stage never talks to the judge

    example._check_judge_credentials(
        argparse.Namespace(
            judge_provider="openai",
            judge_model="openai/gpt-5.4-nano",
            judge_api_key="castform-key",
            judge_base_url="https://llm.castform.dev/v1",
        )
    )
    (url, payload) = probes[-1]
    assert url == "https://llm.castform.dev/v1/chat/completions"
    assert payload["model"] == "gpt-5.4-nano"
    assert "judge preflight: credentials accepted" in capsys.readouterr().out


def test_main_rejects_base_url_for_anthropic(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        example.main(
            [
                "validate",
                *CREDENTIAL_ARGS,
                "--judge-base-url",
                "https://llm.castform.com/v1",
            ]
        )

    assert "--judge-provider openai" in capsys.readouterr().err
