"""Unit tests for benchmax.platform.upload_training_run."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional
from pathlib import Path

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Completion, ToolDefinition
from benchmax.platform import (
    StorageClient,
    UploadedTrainingRun,
    upload_training_run,
)


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
        self, rollout_id: str, completion: Completion, ground_truth: Any, **kwargs: Any
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
        name="test-run",
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


def test_upload_training_run_uploads_four_files_with_correct_storage_paths():
    storage = FakeStorageClient()
    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[{"prompt": "p"}],
        eval_dataset=[{"prompt": "p"}],
        name="run-abc",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    assert paths == [
        "training-runs/run-abc/env-cls.bmxp",
        "training-runs/run-abc/env-metadata.json",
        "training-runs/run-abc/train.jsonl",
        "training-runs/run-abc/eval.jsonl",
    ]


def test_upload_training_run_respects_storage_prefix_override():
    storage = FakeStorageClient()
    upload_training_run(
        env_class=MinimalEnv,
        train_dataset=[],
        eval_dataset=[],
        name="run-x",
        storage_prefix="custom/path",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    assert all(p.startswith("custom/path/run-x/") for p in paths)


def test_upload_training_run_writes_jsonl_one_object_per_line(tmp_path: Path, monkeypatch):
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
        name="jsonl-test",
        storage_client=CapturingStorage(),  # type: ignore[arg-type]
    )

    train_lines = captured["training-runs/jsonl-test/train.jsonl"].decode().splitlines()
    eval_lines = captured["training-runs/jsonl-test/eval.jsonl"].decode().splitlines()

    import json
    assert [json.loads(line) for line in train_lines] == [{"a": 1}, {"a": 2}]
    assert [json.loads(line) for line in eval_lines] == [{"b": 3}]


def test_upload_training_run_requires_api_key_or_storage_client():
    with pytest.raises(ValueError, match="api_key or storage_client"):
        upload_training_run(
            env_class=MinimalEnv,
            train_dataset=[],
            eval_dataset=[],
            name="test",
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
        name="ctor-args",
        constructor_args={"greeting": "hola"},
        storage_client=CapturingStorage(),  # type: ignore[arg-type]
    )

    import json
    metadata = json.loads(captured["training-runs/ctor-args/env-metadata.json"].decode())
    assert metadata["constructor_args"] == {"greeting": "hola"}
