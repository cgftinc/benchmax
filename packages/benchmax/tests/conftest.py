"""
Shared fixtures
"""

import os
import tempfile
import uuid
from pathlib import Path

import pytest

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
