"""
Shared fixtures
"""

import os
import tempfile
import uuid
import pytest
from pathlib import Path
import importlib.util


@pytest.fixture
def unique_rollout_id() -> str:
    """Generate a unique rollout ID for testing."""
    return f"test-rollout-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_sync_dir(tmp_path: Path) -> Path:
    """Temporary directory for mocking syncdir (unit tests only)."""
    sync_dir = tmp_path / "sync"
    os.mkdir(sync_dir)
    return sync_dir


@pytest.fixture(scope="session")
def session_tmp_path() -> Path:
    """Temporary directory for test session."""
    return Path(tempfile.mkdtemp(prefix="benchmax_test_session_"))


@pytest.fixture(scope="session")
def example_workdir() -> Path:
    """Path to example MCP workdir inside benchmax.envs.mcp."""
    # Locate the mcp package dynamically
    spec = importlib.util.find_spec("benchmax.envs.mcp")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate benchmax.envs.mcp package")

    # The directory containing __init__.py
    mcp_pkg_dir = Path(spec.submodule_search_locations[0])

    # Workdir is relative to that
    workdir = mcp_pkg_dir / "example_workdir"

    if not workdir.exists():
        raise FileNotFoundError(f"Expected example_workdir not found at: {workdir}")

    return workdir
