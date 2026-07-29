from __future__ import annotations

from types import SimpleNamespace

import main as example
from benchmax.bundle import load_bundle
from benchmax.envs import Environment


def test_launch_reuses_the_assets_that_were_validated(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundled_environment = object()
    uploaded_assets = object()
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(example, "generate_data", lambda **kwargs: dataset_files)
    monkeypatch.setattr(
        example,
        "dump_bundle",
        lambda *args, **kwargs: bundled_environment,
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
    monkeypatch.setattr("castform.config.llm_url", lambda: "https://judge.example/v1")

    assert example.main(["launch", "--yes"]) == 0
    assert calls == [
        (
            "upload",
            {
                "bundle": bundled_environment,
                "dataset_files": dataset_files,
                "run_name": "telestich",
            },
        ),
        ("validate", uploaded_assets),
        ("launch", uploaded_assets),
    ]


def test_real_bundle_roundtrip_uses_automatic_local_capture() -> None:
    constructor_args = {"judge_base_url": "https://judge.example/v1"}
    bundle = example.dump_bundle(
        example.TelestichEnv,
        constructor_args=constructor_args,
        pip_dependencies=example.RUNTIME_DEPENDENCIES,
    )

    env_class, restored_args = load_bundle(bundle, instantiate=False)

    assert issubclass(env_class, Environment)
    assert env_class.__name__ == "TelestichEnv"
    assert restored_args == constructor_args
    assert bundle.metadata.pip_dependencies == (
        "english-words",
        "openai",
        "pronouncing",
        "wordfreq",
    )


def test_dataset_split_is_deterministic(monkeypatch) -> None:
    rows = [{"prompt": str(index)} for index in range(20)]
    monkeypatch.setattr(example, "_rows", lambda: rows)

    first = example._split_rows()
    second = example._split_rows()

    assert first == second
    assert len(first[0]) == 18
    assert len(first[1]) == 2
