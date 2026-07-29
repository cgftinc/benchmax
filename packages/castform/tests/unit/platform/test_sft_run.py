"""Unit tests for castform.platform.upload_sft_run — the upload half of the
canonicalize -> validate -> upload boundary."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from benchmax.sft import SftDataset, SftIssue, SftRow, load_sft_dataset
from castform.platform import (
    SftDatasetInvalidError,
    UploadedSftRun,
    upload_sft_run,
)


def _row(text: str = "hello") -> dict:
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"re: {text}"},
        ]
    }


def _write_dataset(tmp_path: Path, name: str, rows: list[dict]) -> SftDataset:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return load_sft_dataset(path)


class FakeStorageClient:
    """In-memory StorageClient stand-in. Records calls; returns synthetic blob paths."""

    def __init__(self):
        self.uploads: list[tuple[str, bytes]] = []

    def upload_local_file(
        self, path: str, file_path: Path, *, expires_in_minutes: int | None = None
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


# ---------------------------------------------------------------------------
# Happy path: layout, spread compatibility, optional eval
# ---------------------------------------------------------------------------


def test_upload_sft_run_returns_paths_matching_launch_kwargs(tmp_path):
    """Field names must spread cleanly into TrainerClient.launch_sft_run."""
    storage = FakeStorageClient()
    result = upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
        eval=_write_dataset(tmp_path, "eval.jsonl", [_row("eval")]),
        run_name="test-run",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert isinstance(result, UploadedSftRun)
    assert set(dataclasses.asdict(result)) == {
        "train_dataset_path",
        "eval_dataset_path",
    }


def test_upload_sft_run_uses_hashed_dataset_layout(tmp_path):
    storage = FakeStorageClient()
    result = upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
        eval=_write_dataset(tmp_path, "eval.jsonl", [_row("eval")]),
        run_name="test-run",
        storage_client=storage,  # type: ignore[arg-type]
    )

    keys = [key for key, _ in storage.uploads]
    assert len(keys) == 2
    prefix = keys[0].rsplit("/", 1)[0]
    assert prefix.startswith("datasets/test-run/")
    assert keys == [f"{prefix}/train.jsonl", f"{prefix}/eval.jsonl"]
    assert result.train_dataset_path == f"blob://{prefix}/train.jsonl"
    assert result.eval_dataset_path == f"blob://{prefix}/eval.jsonl"


def test_upload_sft_run_uploads_nothing_for_eval_when_omitted(tmp_path):
    storage = FakeStorageClient()
    result = upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
        run_name="train-only",
        storage_client=storage,  # type: ignore[arg-type]
    )

    keys = [key for key, _ in storage.uploads]
    assert len(keys) == 1
    assert keys[0].endswith("/train.jsonl")
    assert result.eval_dataset_path is None


def test_upload_sft_run_hashes_train_only_and_train_plus_eval_differently(tmp_path):
    """The prefix hash covers the bytes actually uploaded, so adding an eval
    split cannot land on top of an earlier train-only run."""
    train = _write_dataset(tmp_path, "train.jsonl", [_row()])
    eval_dataset = _write_dataset(tmp_path, "eval.jsonl", [_row("eval")])

    train_only = FakeStorageClient()
    upload_sft_run(
        train=train,
        run_name="same-name",
        storage_client=train_only,  # type: ignore[arg-type]
    )
    with_eval = FakeStorageClient()
    upload_sft_run(
        train=train,
        eval=eval_dataset,
        run_name="same-name",
        storage_client=with_eval,  # type: ignore[arg-type]
    )

    assert train_only.uploads[0][0] != with_eval.uploads[0][0]


def test_upload_sft_run_uploads_canonical_jsonl_bytes(tmp_path):
    """Rows go through canonical_jsonl only — never re-serialized here."""
    storage = FakeStorageClient()
    upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row("a"), _row("b")]),
        run_name="canonical",
        storage_client=storage,  # type: ignore[arg-type]
    )

    _, content = storage.uploads[0]
    lines = content.decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [_row("a"), _row("b")]


def test_upload_sft_run_respects_dataset_prefix_override(tmp_path):
    storage = FakeStorageClient()
    upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
        run_name="ignored-for-paths",
        dataset_prefix="datasets/custom/place",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert [key for key, _ in storage.uploads] == ["datasets/custom/place/train.jsonl"]


def test_upload_sft_run_rejects_unsafe_run_name(tmp_path):
    storage = FakeStorageClient()
    with pytest.raises(ValueError, match="Invalid storage path segment"):
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
            run_name="bad name?",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []


# ---------------------------------------------------------------------------
# Enforcement: refusal happens before any storage mutation
# ---------------------------------------------------------------------------


def test_upload_sft_run_refuses_schema_invalid_rows_before_any_upload(tmp_path):
    """A row that fails the schema stops the upload with nothing written."""
    storage = FakeStorageClient()
    # No assistant turn — validate_row reports an error-severity issue.
    invalid = {"messages": [{"role": "user", "content": "unanswered"}]}

    with pytest.raises(SftDatasetInvalidError) as exc_info:
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", [invalid]),
            run_name="invalid-rows",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []
    assert not exc_info.value.report.ok
    assert any(issue.severity == "error" for issue in exc_info.value.report.issues)


def test_upload_sft_run_refuses_partially_loaded_dataset_before_any_upload(tmp_path):
    """A file with one unparseable line never uploads its readable rows."""
    storage = FakeStorageClient()
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_row()) + "\n{ not json\n", encoding="utf-8")

    with pytest.raises(SftDatasetInvalidError) as exc_info:
        upload_sft_run(
            train=load_sft_dataset(path),
            run_name="partial",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []
    assert any("invalid JSON" in issue.message for issue in exc_info.value.report.issues)


def test_upload_sft_run_refuses_empty_training_data_before_any_upload(tmp_path):
    storage = FakeStorageClient()

    with pytest.raises(SftDatasetInvalidError, match="no rows") as exc_info:
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", []),
            run_name="empty",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []
    assert exc_info.value.report.train_row_count == 0


def test_upload_sft_run_refuses_a_hand_built_dataset_that_skipped_validation(tmp_path):
    """The gate is defensive: an SftDataset assembled in code, never passed
    through load_sft_dataset or validate_sft_dataset, is still refused."""
    storage = FakeStorageClient()
    hand_built = SftDataset(
        path=str(tmp_path / "in-memory.jsonl"),
        rows=[SftRow("in-memory.jsonl", 1, {"messages": [], "extra": 1})],
        load_issues=[SftIssue("in-memory.jsonl", 2, "error", "truncated read")],
    )

    with pytest.raises(SftDatasetInvalidError):
        upload_sft_run(
            train=hand_built,
            run_name="hand-built",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []


def test_upload_sft_run_refuses_before_constructing_a_storage_client(monkeypatch, tmp_path):
    """With no storage_client passed, the refusal must still precede every
    storage interaction — including building the client that would perform it."""
    import castform.platform.training_run as training_run

    def explode(*args, **kwargs):
        raise AssertionError("StorageClient constructed before the validation gate")

    monkeypatch.setattr(training_run, "StorageClient", explode)

    with pytest.raises(SftDatasetInvalidError):
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", []),
            run_name="no-client",
        )


def test_upload_sft_run_refusal_names_the_blocking_issues(tmp_path):
    """The message is actionable on its own — a caller that only prints the
    exception still learns which line to fix."""
    storage = FakeStorageClient()
    invalid = {"messages": [{"role": "user", "content": "unanswered"}]}

    with pytest.raises(SftDatasetInvalidError) as exc_info:
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", [_row(), invalid]),
            run_name="named-issues",
            storage_client=storage,  # type: ignore[arg-type]
        )

    message = str(exc_info.value)
    assert "named-issues" in message
    assert "train.jsonl:2" in message
    assert "no trained assistant turn" in message


def test_upload_sft_run_refuses_an_invalid_eval_split_before_any_upload(tmp_path):
    """A valid train split does not buy a pass for a broken eval split — and
    the train file must not be uploaded on its own."""
    storage = FakeStorageClient()
    invalid = {"messages": [{"role": "user", "content": "unanswered"}]}

    with pytest.raises(SftDatasetInvalidError):
        upload_sft_run(
            train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
            eval=_write_dataset(tmp_path, "eval.jsonl", [invalid]),
            run_name="bad-eval",
            storage_client=storage,  # type: ignore[arg-type]
        )

    assert storage.uploads == []


def test_upload_sft_run_accepts_an_empty_eval_split(tmp_path):
    """An empty eval dataset is a notice, not an error — it must not block."""
    storage = FakeStorageClient()
    result = upload_sft_run(
        train=_write_dataset(tmp_path, "train.jsonl", [_row()]),
        eval=_write_dataset(tmp_path, "eval.jsonl", []),
        run_name="empty-eval",
        storage_client=storage,  # type: ignore[arg-type]
    )

    assert len(storage.uploads) == 2
    assert result.eval_dataset_path is not None
