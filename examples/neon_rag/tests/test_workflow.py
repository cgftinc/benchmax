from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import pytest
from benchmax.bundle import dump_bundle, load_bundle

_MAIN_PATH = Path(__file__).parents[1] / "main.py"
_SPEC = importlib.util.spec_from_file_location("neon_rag_example_main", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOCAL_MODULE_NAMES = ("data", "neon_rag_env")
_PREVIOUS = {name: sys.modules.pop(name, None) for name in _LOCAL_MODULE_NAMES}
sys.path.insert(0, str(_MAIN_PATH.parent))
try:
    main = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = main
    _SPEC.loader.exec_module(main)
    _RUNTIME_MODULES = {
        name: sys.modules[name] for name in _LOCAL_MODULE_NAMES if name in sys.modules
    }
finally:
    sys.path.remove(str(_MAIN_PATH.parent))
    for _name, _module in _PREVIOUS.items():
        if _module is not None:
            sys.modules[_name] = _module
        else:
            sys.modules.pop(_name, None)


def test_environment_supports_hybrid_search() -> None:
    env = main.NeonRagEnv(
        judge_base_url="https://models.example/v1",
        embedding_base_url="https://models.example/v1",
        search_database_url="postgresql://ro@db.example/neondb",
    )

    assert env._search.available_modes == ["hybrid", "lexical", "vector"]
    assert env._search.get_params() == {
        "backend": "neon",
        "table": "neon_rag",
        "schema": "benchmax_corpus",
    }


def test_environment_bundle_carries_explicit_search_database_url() -> None:
    secret = "postgresql://benchmax_ro:very-secret@db.example/neondb"
    env = main.NeonRagEnv(
        judge_base_url="https://models.example/v1",
        embedding_base_url="https://models.example/v1",
        search_database_url=secret,
    )

    payload = cloudpickle.dumps(env)

    assert secret.encode() in payload


def test_deployable_bundle_excludes_castform_and_ingestion() -> None:
    secret = "postgresql://benchmax_ro:very-secret@db.example/neondb"
    with (
        patch.dict(sys.modules, _RUNTIME_MODULES),
        patch.object(sys, "path", [str(_MAIN_PATH.parent), *sys.path]),
    ):
        bundle = dump_bundle(
            main.NeonRagEnv,
            constructor_args={
                "judge_base_url": "https://models.example/v1",
                "embedding_base_url": "https://models.example/v1",
                "search_database_url": secret,
            },
            pip_dependencies=main.RUNTIME_DEPENDENCIES,
        )
    env = load_bundle(bundle)

    assert env._search.available_modes == ["hybrid", "lexical", "vector"]
    assert secret.encode() in bundle.pickled
    assert b"castform.rag" not in bundle.pickled
    assert b"NeonChunkSource" not in bundle.pickled


def test_data_action_runs_the_complete_pipeline(monkeypatch) -> None:
    calls: list[tuple[str, bool, int]] = []
    monkeypatch.setattr(main, "ensure_session", lambda: None)
    monkeypatch.setattr(
        main,
        "prepare_data",
        lambda *, data_preparation_database_url, force, question_count: calls.append(
            (data_preparation_database_url, force, question_count)
        ),
    )

    assert (
        main.main(
            [
                "data",
                "--force",
                "--question-count",
                "12",
                "--neon-data-preparation-database-url",
                "postgresql://rw@db/neondb",
            ]
        )
        == 0
    )
    assert calls == [("postgresql://rw@db/neondb", True, 12)]


def test_validation_uploads_assets_and_runs_remote_rollout(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundle = object()
    assets = SimpleNamespace(
        env_cls_path="env/neon/env-cls.pkl",
        env_metadata_path="env/neon/env-metadata.json",
        dataset_path="datasets/neon",
    )
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(main, "require_dataset_files", lambda: dataset_files)
    monkeypatch.setattr(
        main,
        "_constructor_args",
        lambda search_database_url: {
            "judge_base_url": "https://models.example/v1",
            "embedding_base_url": "https://models.example/v1",
            "search_database_url": search_database_url,
        },
    )
    monkeypatch.setattr(main, "ensure_session", lambda: None)
    monkeypatch.setattr(main, "dump_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        main,
        "upload_assets",
        lambda **kwargs: calls.append(("upload", kwargs)) or assets,
    )
    monkeypatch.setattr(
        main,
        "validate",
        lambda env, uploaded: calls.append(("validate", uploaded)) or report,
    )

    assert (
        main.main(
            [
                "validate",
                "--neon-search-database-url",
                "postgresql://ro@db/neondb",
            ]
        )
        == 0
    )
    assert calls == [
        (
            "upload",
            {
                "bundle": bundle,
                "dataset_files": dataset_files,
                "run_name": "neon-rag",
            },
        ),
        ("validate", assets),
    ]


def test_launch_prepares_data_before_upload_and_validation(monkeypatch) -> None:
    dataset_files = {"train.jsonl": object(), "eval.jsonl": object()}
    bundle = object()
    assets = SimpleNamespace(
        env_cls_path="env/neon/env-cls.pkl",
        env_metadata_path="env/neon/env-metadata.json",
        dataset_path="datasets/neon",
    )
    report = SimpleNamespace(ok=True)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(main, "ensure_session", lambda: calls.append(("session", None)))
    monkeypatch.setattr(
        main,
        "prepare_data",
        lambda **kwargs: calls.append(("prepare", kwargs)) or dataset_files,
    )
    monkeypatch.setattr(main, "dump_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        main,
        "upload_assets",
        lambda **kwargs: calls.append(("upload", kwargs)) or assets,
    )
    monkeypatch.setattr(
        main,
        "validate",
        lambda env, uploaded: calls.append(("validate", uploaded)) or report,
    )
    monkeypatch.setattr(
        main,
        "launch",
        lambda uploaded, *, assume_yes: calls.append(("launch", (uploaded, assume_yes))),
    )

    assert (
        main.main(
            [
                "launch",
                "--force",
                "--yes",
                "--question-count",
                "24",
                "--neon-data-preparation-database-url",
                "postgresql://prepare@db/neondb",
                "--neon-search-database-url",
                "postgresql://search@db/neondb",
            ]
        )
        == 0
    )

    assert calls[0] == ("session", None)
    assert calls[1] == (
        "prepare",
        {
            "data_preparation_database_url": "postgresql://prepare@db/neondb",
            "force": True,
            "question_count": 24,
        },
    )
    assert calls[-1] == ("launch", (assets, True))


@pytest.mark.parametrize(
    ("argv", "missing_flag"),
    [
        (["data"], "--neon-data-preparation-database-url"),
        (["validate"], "--neon-search-database-url"),
        (
            [
                "launch",
                "--neon-search-database-url",
                "postgresql://search@db/neondb",
            ],
            "--neon-data-preparation-database-url",
        ),
    ],
)
def test_actions_report_missing_database_urls(argv, missing_flag, capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main.run_cli(argv)

    assert f"{argv[0]} requires {missing_flag}" in capsys.readouterr().err
