"""Unit tests for castform.platform.upload_training_run."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Optional

import pytest

from benchmax.bundle import Bundle, BundleMetadata
from castform.platform import (
    UploadedTrainingRun,
    upload_training_run,
)


def _bundle(*, pickled: bytes = b"caller-selected-pickle") -> Bundle:
    return Bundle(
        pickled=pickled,
        metadata=BundleMetadata(
            pip_dependencies=["example-dependency==1.2.3"],
            python_version="3.12",
            benchmax_version="0.1.0",
            env_class_source="class ExampleEnv: ...\n",
        ),
    )


class FakeStorageClient:
    """In-memory StorageClient stand-in. Records calls; returns synthetic blob paths."""

    def __init__(self):
        self.uploads: list[tuple[str, bytes]] = []

    def upload_local_file(
        self, path: str, file_path: Path, *, expires_in_minutes: Optional[int] = None
    ) -> dict:
        # Verify the file actually exists at upload time (it lives in a tempdir
        # that gets deleted on context exit — catches lifetime bugs).
        assert Path(file_path).exists(), f"File missing at upload: {file_path}"
        self.uploads.append((path, Path(file_path).read_bytes()))
        return {
            "blobPath": f"blob://{path}",
            "uploadUrl": f"https://example.invalid/{path}",
            "expiresAt": "2099-01-01T00:00:00Z",
            "willOverwrite": False,
        }


def test_upload_training_run_returns_paths_matching_launch_kwargs():
    """Field names must spread cleanly into TrainerClient.launch_training_run."""
    storage = FakeStorageClient()
    result = upload_training_run(
        bundle=_bundle(),
        train_dataset=[{"prompt": "p", "ground_truth": "g"}],
        eval_dataset=[{"prompt": "p2", "ground_truth": "g2"}],
        run_name="test-run",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert isinstance(result, UploadedTrainingRun)
    # Spread test — field names must match the launch_training_run signature.
    spread = dataclasses.asdict(result)
    assert set(spread.keys()) == {
        "env_cls_path",
        "env_metadata_path",
        "train_dataset_path",
        "eval_dataset_path",
    }


def test_upload_training_run_can_upload_bundle_without_datasets():
    storage = FakeStorageClient()

    result = upload_training_run(
        bundle=_bundle(),
        run_name="harbor-managed",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert [path for path, _ in storage.uploads] == [
        f"envs/harbor-managed/{hashlib.sha256(_bundle().pickled).hexdigest()[:16]}/env-cls.pkl",
        f"envs/harbor-managed/{hashlib.sha256(_bundle().pickled).hexdigest()[:16]}/env-metadata.json",
    ]
    assert result.train_dataset_path is None
    assert result.eval_dataset_path is None


def test_upload_training_run_uploads_only_the_supplied_dataset_split():
    storage = FakeStorageClient()

    result = upload_training_run(
        bundle=_bundle(),
        train_dataset=[{"prompt": "p"}],
        run_name="train-only",
        dataset_prefix="fixed/dataset",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert [path for path, _ in storage.uploads if path.startswith("fixed/")] == [
        "fixed/dataset/train.jsonl"
    ]
    assert result.train_dataset_path == "blob://fixed/dataset/train.jsonl"
    assert result.eval_dataset_path is None


def test_upload_training_run_rejects_dataset_prefix_without_datasets():
    storage = FakeStorageClient()

    with pytest.raises(ValueError, match="dataset_prefix requires"):
        upload_training_run(
            bundle=_bundle(),
            run_name="bundle-only",
            dataset_prefix="unused/datasets",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []


def test_upload_training_run_uses_hashed_envs_and_datasets_layout():
    storage = FakeStorageClient()
    upload_training_run(
        bundle=_bundle(),
        train_dataset=[{"prompt": "p"}],
        eval_dataset=[{"prompt": "p"}],
        run_name="run-abc",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    assert len(paths) == 4
    env_paths = [p for p in paths if p.startswith("envs/")]
    dataset_paths = [p for p in paths if p.startswith("datasets/")]
    assert len(env_paths) == 2
    assert len(dataset_paths) == 2

    # All env files sit under a single envs/<run>/<16-hex>/ prefix.
    env_dirs = {p.rsplit("/", 1)[0] for p in env_paths}
    assert len(env_dirs) == 1
    (env_dir,) = env_dirs
    parts = env_dir.split("/")
    assert parts[0] == "envs"
    assert parts[1] == "run-abc"
    assert len(parts[2]) == 16  # hash slice
    assert {p.rsplit("/", 1)[1] for p in env_paths} == {
        "env-cls.pkl",
        "env-metadata.json",
    }

    # All dataset files sit under a single datasets/<run>/<8-hex>/ prefix.
    ds_dirs = {p.rsplit("/", 1)[0] for p in dataset_paths}
    assert len(ds_dirs) == 1
    (ds_dir,) = ds_dirs
    parts = ds_dir.split("/")
    assert parts[0] == "datasets"
    assert parts[1] == "run-abc"
    assert len(parts[2]) == 8
    assert {p.rsplit("/", 1)[1] for p in dataset_paths} == {
        "train.jsonl",
        "eval.jsonl",
    }


def test_upload_training_run_respects_env_prefix_override():
    storage = FakeStorageClient()
    upload_training_run(
        bundle=_bundle(),
        train_dataset=[],
        eval_dataset=[],
        run_name="run-x",
        env_prefix="custom/env/path",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    env_paths = [p for p in paths if not p.startswith("datasets/")]
    assert set(env_paths) == {
        "custom/env/path/env-cls.pkl",
        "custom/env/path/env-metadata.json",
    }
    # Datasets still use the default layout.
    assert all(
        p.startswith("datasets/run-x/") for p in paths if p.startswith("datasets/")
    )


def test_upload_training_run_respects_dataset_prefix_override():
    storage = FakeStorageClient()
    upload_training_run(
        bundle=_bundle(),
        train_dataset=[],
        eval_dataset=[],
        run_name="run-y",
        dataset_prefix="custom/data/path",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    ds_paths = [p for p in paths if not p.startswith("envs/")]
    assert set(ds_paths) == {
        "custom/data/path/train.jsonl",
        "custom/data/path/eval.jsonl",
    }


def test_upload_training_run_writes_jsonl_one_object_per_line():
    """The train/eval files must be valid JSONL."""
    captured: dict[str, bytes] = {}

    class CapturingStorage:
        def upload_local_file(self, path, file_path, **kwargs):
            captured[path] = Path(file_path).read_bytes()
            return {
                "blobPath": f"blob://{path}",
                "uploadUrl": "",
                "expiresAt": "",
                "willOverwrite": False,
            }

    upload_training_run(
        bundle=_bundle(),
        train_dataset=[{"a": 1}, {"a": 2}],
        eval_dataset=[{"b": 3}],
        run_name="jsonl-test",
        dataset_prefix="fixed/ds",
        storage_client=CapturingStorage(),  # type: ignore[arg-type]
    )

    train_lines = captured["fixed/ds/train.jsonl"].decode().splitlines()
    eval_lines = captured["fixed/ds/eval.jsonl"].decode().splitlines()

    import json

    assert [json.loads(line) for line in train_lines] == [{"a": 1}, {"a": 2}]
    assert [json.loads(line) for line in eval_lines] == [{"b": 3}]


def test_upload_training_run_api_key_optional_resolves_via_seam(monkeypatch):
    """api_key is optional: with neither api_key nor storage_client, the built
    StorageClient gets api_key=None and resolves the bearer per request via the
    seam (ACT_AS_TOKEN_PATH / PLATFORM_API_KEY) — no upfront guard."""
    captured: dict[str, Any] = {}

    def _fake_storage_client(*, api_key, base_url):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return FakeStorageClient()

    monkeypatch.setattr(
        "castform.platform.training_run.StorageClient", _fake_storage_client
    )

    result = upload_training_run(
        bundle=_bundle(),
        train_dataset=[{"a": 1}],
        eval_dataset=[{"a": 1}],
        run_name="test",
    )

    assert captured["api_key"] is None  # no guard; bearer resolves at request time
    assert isinstance(result, UploadedTrainingRun)


def test_upload_training_run_rejects_unsafe_run_name():
    """A run_name with a URL-breaking char fails loud before any upload."""
    storage = FakeStorageClient()
    with pytest.raises(ValueError, match="Invalid storage path segment"):
        upload_training_run(
            bundle=_bundle(),
            train_dataset=[{"a": 1}],
            eval_dataset=[{"a": 1}],
            run_name="rag-is-eval-fixed?",
            storage_client=storage,  # type: ignore[arg-type]
        )
    assert storage.uploads == []  # nothing uploaded


def test_upload_training_run_rejects_unsafe_prefix_override():
    storage = FakeStorageClient()
    with pytest.raises(ValueError, match="Invalid storage path segment"):
        upload_training_run(
            bundle=_bundle(),
            train_dataset=[],
            eval_dataset=[],
            run_name="ok",
            env_prefix="custom/bad name/env",
            storage_client=storage,  # type: ignore[arg-type]
        )


def test_upload_training_run_uploads_supplied_bundle_exactly():
    """Castform uploads the caller's artifact without rebuilding or altering it."""
    storage = FakeStorageClient()
    bundle = _bundle(pickled=b"not-even-a-pickle\x00\xff")

    result = upload_training_run(
        bundle=bundle,
        train_dataset=[],
        eval_dataset=[],
        run_name="exact-bundle",
        env_prefix="fixed/env",
        dataset_prefix="fixed/dataset",
        storage_client=storage,  # type: ignore[arg-type]
    )

    uploaded = dict(storage.uploads)
    assert uploaded["fixed/env/env-cls.pkl"] == bundle.pickled
    assert uploaded["fixed/env/env-metadata.json"] == bundle.metadata.to_json_bytes()
    assert result.env_cls_path == "blob://fixed/env/env-cls.pkl"
    assert result.env_metadata_path == "blob://fixed/env/env-metadata.json"


def test_upload_training_run_hashes_the_supplied_bundle_bytes():
    storage = FakeStorageClient()
    bundle = _bundle(pickled=b"exact hash input")

    upload_training_run(
        bundle=bundle,
        train_dataset=[],
        eval_dataset=[],
        run_name="hash-input",
        storage_client=storage,  # type: ignore[arg-type]
    )

    expected_hash = hashlib.sha256(bundle.pickled).hexdigest()[:16]
    paths = [path for path, _ in storage.uploads]
    assert f"envs/hash-input/{expected_hash}/env-cls.pkl" in paths
    assert f"envs/hash-input/{expected_hash}/env-metadata.json" in paths
