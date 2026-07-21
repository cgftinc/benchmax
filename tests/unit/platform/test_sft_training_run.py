"""Unit tests for benchmax.platform.upload_sft_run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from benchmax.platform import UploadedSftRun, upload_sft_run
from benchmax.sft.dataset import SftDataset, SftRow, canonical_jsonl


def _dataset(rows: list[dict[str, Any]], path: str = "train.jsonl") -> SftDataset:
    return SftDataset(
        path=path,
        rows=[SftRow(path, i + 1, r) for i, r in enumerate(rows)],
        load_issues=[],
    )


class FakeStorageClient:
    """In-memory StorageClient stand-in. Records calls; returns synthetic blob paths."""

    def __init__(self):
        self.uploads: list[tuple[str, Path]] = []

    def upload_local_file(
        self, path: str, file_path: Path, *, expires_in_minutes: Optional[int] = None
    ) -> dict:
        self.uploads.append((path, Path(file_path)))
        assert Path(file_path).exists(), f"File missing at upload: {file_path}"
        return {
            "blobPath": f"blob://{path}",
            "uploadUrl": f"https://example.invalid/{path}",
            "expiresAt": "2099-01-01T00:00:00Z",
            "willOverwrite": False,
        }


class CapturingStorage:
    """Records the raw bytes written for each uploaded path."""

    def __init__(self):
        self.captured: dict[str, bytes] = {}

    def upload_local_file(self, path, file_path, **kwargs):
        self.captured[path] = Path(file_path).read_bytes()
        return {
            "blobPath": f"blob://{path}",
            "uploadUrl": "",
            "expiresAt": "",
            "willOverwrite": False,
        }


def test_upload_sft_run_eval_none_uploads_only_train():
    storage = FakeStorageClient()
    train = _dataset([{"messages": [{"role": "user", "content": "hi"}]}])

    result = upload_sft_run(
        train=train,
        eval=None,
        run_name="sft-run",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert isinstance(result, UploadedSftRun)
    assert result.eval_dataset_path is None
    paths = [p for p, _ in storage.uploads]
    assert len(paths) == 1
    assert paths[0].endswith("/train.jsonl")


def test_upload_sft_run_eval_present_uploads_both():
    storage = FakeStorageClient()
    train = _dataset([{"messages": [{"role": "user", "content": "hi"}]}])
    eval_ds = _dataset(
        [{"messages": [{"role": "user", "content": "bye"}]}], path="eval.jsonl"
    )

    result = upload_sft_run(
        train=train,
        eval=eval_ds,
        run_name="sft-run",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert result.train_dataset_path is not None
    assert result.eval_dataset_path is not None
    paths = {p for p, _ in storage.uploads}
    assert any(p.endswith("/train.jsonl") for p in paths)
    assert any(p.endswith("/eval.jsonl") for p in paths)
    assert len(paths) == 2


def test_upload_sft_run_output_is_byte_identical_to_canonical_jsonl():
    """The upload helper must serialize via canonical_jsonl and nothing else."""
    storage = CapturingStorage()
    train = _dataset(
        [
            {"messages": [{"role": "user", "content": "hi"}], "tools": [{"a": 1}]},
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a", "weight": 1},
                ]
            },
        ]
    )
    eval_ds = _dataset(
        [{"messages": [{"role": "user", "content": "eval-row"}]}], path="eval.jsonl"
    )

    upload_sft_run(
        train=train,
        eval=eval_ds,
        run_name="byte-check",
        dataset_prefix="fixed/ds",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert storage.captured["fixed/ds/train.jsonl"] == canonical_jsonl(train)
    assert storage.captured["fixed/ds/eval.jsonl"] == canonical_jsonl(eval_ds)


def test_upload_sft_run_respects_dataset_prefix_override():
    storage = FakeStorageClient()
    train = _dataset([{"messages": [{"role": "user", "content": "hi"}]}])

    upload_sft_run(
        train=train,
        eval=None,
        run_name="run-y",
        dataset_prefix="custom/data/path",
        storage_client=storage,  # type: ignore[arg-type]
    )

    paths = [path for path, _ in storage.uploads]
    assert paths == ["custom/data/path/train.jsonl"]


def test_upload_sft_run_api_key_optional_resolves_via_seam(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_storage_client(*, api_key, base_url):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return FakeStorageClient()

    monkeypatch.setattr(
        "benchmax.platform.training_run.StorageClient", _fake_storage_client
    )

    train = _dataset([{"messages": [{"role": "user", "content": "hi"}]}])
    result = upload_sft_run(train=train, eval=None, run_name="test")

    assert captured["api_key"] is None
    assert isinstance(result, UploadedSftRun)
