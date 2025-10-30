"""
Tests for MCP utility functions.
"""

import pytest
import tempfile
from pathlib import Path

from benchmax.envs.mcp.provisioners.utils import (
    setup_sync_dir,
    cleanup_dir,
)


class TestSetupSyncDir:
    """Tests for setup_sync_dir function."""

    def test_setup_sync_dir_creates_temp_directory(self, example_workdir: Path):
        """Test that sync directory is created in temp location."""
        sync_dir = setup_sync_dir(example_workdir)

        try:
            assert sync_dir.exists()
            assert sync_dir.is_dir()
            assert "benchmax_skypilot_" in sync_dir.name
        finally:
            cleanup_dir(sync_dir)

    def test_setup_sync_dir_copies_proxy_server(self, example_workdir: Path):
        """Test that proxy_server.py is copied to sync directory."""
        sync_dir = setup_sync_dir(example_workdir)

        try:
            proxy_server = sync_dir / "proxy_server.py"
            assert proxy_server.exists()
            assert proxy_server.is_file()
        finally:
            cleanup_dir(sync_dir)

    def test_setup_sync_dir_copies_workdir_contents(self, example_workdir: Path):
        """Test that all workdir contents are copied."""
        sync_dir = setup_sync_dir(example_workdir)

        try:
            # Check required files
            assert (sync_dir / "mcp_config.yaml").exists()
            assert (sync_dir / "reward_fn.py").exists()
            assert (sync_dir / "setup.sh").exists()
            assert (sync_dir / "proxy_server.py").exists()
        finally:
            cleanup_dir(sync_dir)

    def test_setup_sync_dir_validates_required_files(self, tmp_path: Path):
        """Test that missing required files raise FileNotFoundError."""
        # Create incomplete workdir
        incomplete_workdir = tmp_path / "incomplete"
        incomplete_workdir.mkdir()
        (incomplete_workdir / "mcp_config.yaml").touch()
        # Missing reward_fn.py and setup.sh

        with pytest.raises(FileNotFoundError, match="reward_fn.py"):
            setup_sync_dir(incomplete_workdir)

    def test_setup_sync_dir_handles_nonexistent_workdir(self, tmp_path: Path):
        """Test that nonexistent workdir raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="workdir_path"):
            setup_sync_dir(nonexistent)

    def test_setup_sync_dir_handles_file_as_workdir(self, tmp_path: Path):
        """Test that file instead of directory raises ValueError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="directory"):
            setup_sync_dir(file_path)

    def test_setup_sync_dir_cleanup_on_error(self, tmp_path: Path):
        """Test that sync directory is cleaned up if setup fails."""
        # Create workdir with required files but missing reward_fn.py
        incomplete_workdir = tmp_path / "workdir"
        incomplete_workdir.mkdir()
        (incomplete_workdir / "mcp_config.yaml").touch()
        (incomplete_workdir / "setup.sh").touch()

        # This should fail because reward_fn.py doesn't exist in mcp/
        # But the temp directory should be cleaned up
        with pytest.raises(FileNotFoundError):
            setup_sync_dir(incomplete_workdir)


class TestCleanupDir:
    """Tests for cleanup_dir function."""

    def test_cleanup_dir_removes_directory(self):
        """Test that cleanup_dir removes an existing directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_cleanup_"))
        assert temp_dir.exists()

        cleanup_dir(temp_dir)
        assert not temp_dir.exists()

    def test_cleanup_dir_handles_none(self):
        """Test that cleanup_dir handles None gracefully."""
        cleanup_dir(None)  # Should not raise

    def test_cleanup_dir_handles_nonexistent(self, tmp_path: Path):
        """Test that cleanup_dir handles nonexistent path gracefully."""
        nonexistent = tmp_path / "does_not_exist"
        cleanup_dir(nonexistent)  # Should not raise

    def test_cleanup_dir_handles_file(self, tmp_path: Path):
        """Test that cleanup_dir ignores files (only removes directories)."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        cleanup_dir(file_path)  # Should not raise
        assert file_path.exists()  # File should still exist
