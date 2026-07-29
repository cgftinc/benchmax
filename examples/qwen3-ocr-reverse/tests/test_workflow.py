from __future__ import annotations

from types import SimpleNamespace

import main


def test_launch_reuses_the_assets_that_were_validated(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundled_environment = object()
    uploaded_assets = SimpleNamespace(
        env_cls_path="envs/test/env-cls.pkl",
        env_metadata_path="envs/test/env-metadata.json",
        dataset_path=None,
    )
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
        lambda **kwargs: calls.append(("upload", kwargs)) or uploaded_assets,
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
                "run_name": "qwen3-ocr-reverse",
            },
        ),
        ("validate", uploaded_assets),
        ("launch", uploaded_assets),
    ]
