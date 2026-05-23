"""Unit tests for benchmax.platform.upload_training_run."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Messages, ToolDefinition
from benchmax.platform import (
    StorageClient,
    UploadedTrainingRun,
    upload_training_run,
)

# MinimalEnv is defined in this test module; cloudpickle would otherwise
# pickle it by-reference and fail on a fresh worker. dump_bundle now
# enforces this, so the test module itself must be registered for by-value.
_TEST_MODULE = sys.modules[__name__]


class MinimalEnv(BaseEnv):
    """Minimal valid BaseEnv subclass for bundling tests."""

    system_prompt = "test"

    def __init__(self, greeting: str = "hello"):
        self.greeting = greeting

    async def list_tools(self) -> List[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return None

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"score": 0.0}


class FakeStorageClient:
    """In-memory StorageClient stand-in. Records calls; returns synthetic blob paths."""

    def __init__(self):
        self.uploads: list[tuple[str, Path]] = []

    def upload_local_file(
        self, path: str, file_path: Path, *, expires_in_minutes: Optional[int] = None
    ) -> dict:
        self.uploads.append((path, Path(file_path)))
        # Verify the file actually exists at upload time (it lives in a tempdir
        # that gets deleted on context exit — catches lifetime bugs).
        assert Path(file_path).exists(), f"File missing at upload: {file_path}"
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
        env_class=MinimalEnv,
        train_dataset=[{"prompt": "p", "ground_truth": "g"}],
        eval_dataset=[{"prompt": "p2", "ground_truth": "g2"}],
        run_name="test-run",
        storage_client=storage,  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
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


def test_upload_training_run_uses_hashed_envs_and_datasets_layout():
    storage = FakeStorageClient()
    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[{"prompt": "p"}],
        eval_dataset=[{"prompt": "p"}],
        run_name="run-abc",
        storage_client=storage,  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
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
        env_class=MinimalEnv,
        train_dataset=[],
        eval_dataset=[],
        run_name="run-x",
        env_prefix="custom/env/path",
        storage_client=storage,  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
    )

    paths = [path for path, _ in storage.uploads]
    env_paths = [p for p in paths if not p.startswith("datasets/")]
    assert set(env_paths) == {
        "custom/env/path/env-cls.pkl",
        "custom/env/path/env-metadata.json",
    }
    # Datasets still use the default layout.
    assert all(p.startswith("datasets/run-x/") for p in paths if p.startswith("datasets/"))


def test_upload_training_run_respects_dataset_prefix_override():
    storage = FakeStorageClient()
    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[],
        eval_dataset=[],
        run_name="run-y",
        dataset_prefix="custom/data/path",
        storage_client=storage,  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
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
            return {"blobPath": f"blob://{path}", "uploadUrl": "", "expiresAt": "", "willOverwrite": False}

    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[{"a": 1}, {"a": 2}],
        eval_dataset=[{"b": 3}],
        run_name="jsonl-test",
        dataset_prefix="fixed/ds",
        storage_client=CapturingStorage(),  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
    )

    train_lines = captured["fixed/ds/train.jsonl"].decode().splitlines()
    eval_lines = captured["fixed/ds/eval.jsonl"].decode().splitlines()

    import json
    assert [json.loads(line) for line in train_lines] == [{"a": 1}, {"a": 2}]
    assert [json.loads(line) for line in eval_lines] == [{"b": 3}]


def test_upload_training_run_requires_api_key_or_storage_client():
    with pytest.raises(ValueError, match="api_key or storage_client"):
        upload_training_run(
            env_class=MinimalEnv,
            train_dataset=[],
            eval_dataset=[],
            run_name="test",
        )


def test_upload_training_run_passes_constructor_args_through_to_bundle():
    """constructor_args should reach the bundled metadata."""
    captured: dict[str, bytes] = {}

    class CapturingStorage:
        def upload_local_file(self, path, file_path, **kwargs):
            captured[path] = Path(file_path).read_bytes()
            return {"blobPath": f"blob://{path}", "uploadUrl": "", "expiresAt": "", "willOverwrite": False}

    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[],
        eval_dataset=[],
        run_name="ctor-args",
        env_prefix="fixed/env",
        constructor_args={"greeting": "hola"},
        storage_client=CapturingStorage(),  # type: ignore[arg-type]
        local_modules=[_TEST_MODULE],
    )

    import cloudpickle
    _, ctor_args = cloudpickle.loads(captured["fixed/env/env-cls.pkl"])
    assert ctor_args == {"greeting": "hola"}
