"""Tests for CheckpointBase shared I/O."""

import json

from benchmax.utils.checkpoint import CheckpointBase


class TestCheckpointBase:
    def test_manifest_roundtrip(self, tmp_path):
        cp = CheckpointBase(tmp_path / "ckpt")
        assert cp.load_manifest() is None

        cp.save_manifest({"version": 1, "model": "test"})
        loaded = cp.load_manifest()
        assert loaded == {"version": 1, "model": "test"}

    def test_manifest_is_atomic(self, tmp_path):
        cp = CheckpointBase(tmp_path / "ckpt")
        cp.save_manifest({"key": "value"})
        assert not cp.manifest_path.with_suffix(".tmp").exists()

    def test_jsonl_roundtrip(self, tmp_path):
        cp = CheckpointBase(tmp_path / "ckpt")
        items = [{"a": 1}, {"b": 2}, {"c": 3}]
        cp.append_jsonl("data.jsonl", items)

        loaded = cp.load_jsonl("data.jsonl")
        assert loaded == items

    def test_jsonl_incremental_append(self, tmp_path):
        cp = CheckpointBase(tmp_path / "ckpt")
        cp.append_jsonl("data.jsonl", [{"batch": 1}])
        cp.append_jsonl("data.jsonl", [{"batch": 2}])

        loaded = cp.load_jsonl("data.jsonl")
        assert len(loaded) == 2

    def test_jsonl_truncation_recovery(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        cp = CheckpointBase(ckpt_dir)
        cp.append_jsonl("data.jsonl", [{"good": 1}, {"good": 2}])

        # Simulate crash: append truncated line
        path = ckpt_dir / "data.jsonl"
        with path.open("a") as fh:
            fh.write('{"truncat')

        loaded = cp.load_jsonl("data.jsonl")
        assert len(loaded) == 2
        assert loaded[0] == {"good": 1}
        assert loaded[1] == {"good": 2}

    def test_load_jsonl_missing_file(self, tmp_path):
        cp = CheckpointBase(tmp_path / "ckpt")
        assert cp.load_jsonl("missing.jsonl") == []

    def test_cleanup(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        cp = CheckpointBase(ckpt_dir)
        cp.save_manifest({"test": True})
        assert ckpt_dir.exists()

        cp.cleanup()
        assert not ckpt_dir.exists()

    def test_cleanup_nonexistent_dir(self, tmp_path):
        cp = CheckpointBase(tmp_path / "nonexistent")
        cp.cleanup()  # should not raise

    def test_mkdir_on_first_write(self, tmp_path):
        ckpt_dir = tmp_path / "nested" / "deep" / "ckpt"
        cp = CheckpointBase(ckpt_dir)
        assert not ckpt_dir.exists()

        cp.save_manifest({"test": True})
        assert ckpt_dir.exists()
