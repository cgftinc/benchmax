"""Slice 1.6 offline: `castform data upload` (mocked StorageClient)."""

from __future__ import annotations

import argparse

from castform.cli import data


class _FakeStorage:
    def __init__(self, result=None, raise_exc=None):
        self.result = result or {"blobPath": "datasets/cli/x.jsonl"}
        self.raise_exc = raise_exc
        self.uploaded = None

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def upload_local_file(self, path, file_path):
        self.uploaded = (path, file_path)
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def _ns(file, **kw):
    base = dict(file=file, path=None, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_upload_ok(monkeypatch, tmp_path, capsys):
    f = tmp_path / "train.jsonl"
    f.write_text('{"a": 1}\n')
    fake = _FakeStorage(result={"blobPath": "datasets/cli/train.jsonl"})
    monkeypatch.setattr(data, "StorageClient", lambda: fake)
    assert data._cmd_data_upload(_ns(str(f))) == 0
    assert fake.uploaded[0] == "datasets/cli/train.jsonl"  # default path
    assert "datasets/cli/train.jsonl" in capsys.readouterr().out


def test_upload_custom_path(monkeypatch, tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("{}\n")
    fake = _FakeStorage()
    monkeypatch.setattr(data, "StorageClient", lambda: fake)
    assert data._cmd_data_upload(_ns(str(f), path="datasets/custom/d.jsonl")) == 0
    assert fake.uploaded[0] == "datasets/custom/d.jsonl"


def test_upload_missing_file(monkeypatch, capsys):
    assert data._cmd_data_upload(_ns("/tmp/does-not-exist-xyz.jsonl")) == 1
    assert "not found" in capsys.readouterr().err


def test_upload_unsupported_type(monkeypatch, tmp_path, capsys):
    f = tmp_path / "bad.txt"
    f.write_text("x")
    fake = _FakeStorage(raise_exc=ValueError("Unsupported file type: .txt"))
    monkeypatch.setattr(data, "StorageClient", lambda: fake)
    assert data._cmd_data_upload(_ns(str(f))) == 1
    assert "Unsupported file type" in capsys.readouterr().err
