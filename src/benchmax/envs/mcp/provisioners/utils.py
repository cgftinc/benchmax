"""
Utility functions for server provisioners
"""

import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional


def setup_sync_dir(workdir_path: Path) -> Path:
    """
    Set up a temporary directory to sync the workdir contents.

    This creates a temp directory and copies:
    1. proxy_server.py from the mcp/ directory
    2. env tracking helper for reward logging
    3. All contents of the provided workdir_path

    Args:
        workdir_path: Path to workdir containing mcp_config.yaml, setup.sh, etc.

    Returns:
        Path to the temporary sync directory.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If workdir_path is not a directory.
    """
    sync_dir = Path(tempfile.mkdtemp(prefix="benchmax_skypilot_"))

    try:
        # Copy proxy_server.py (located in the mcp/ directory)
        src_server_path = Path(__file__).parent.parent / "proxy_server.py"
        if not src_server_path.exists():
            raise FileNotFoundError(
                f"Expected proxy_server.py at {src_server_path}, but not found."
            )
        shutil.copy(src_server_path, sync_dir / "proxy_server.py")

        # Copy shared env tracking helper for reward_fn logging.
        src_tracking_path = Path(__file__).parents[2] / "tracking.py"
        if not src_tracking_path.exists():
            raise FileNotFoundError(
                f"Expected tracking helper at {src_tracking_path}, but not found."
            )
        shutil.copy(src_tracking_path, sync_dir / "env_tracking.py")

        # Validate workdir exists and is a directory
        if not workdir_path.exists():
            raise FileNotFoundError(
                f"Expected workdir_path at {workdir_path}, but not found."
            )
        if not workdir_path.is_dir():
            raise ValueError(
                f"Expected workdir_path at {workdir_path} to be a directory."
            )

        # Validate required files in the workdir using regex patterns
        required_patterns = {
            r"^reward_fn\.py$": "reward_fn.py",
            r"^setup\.sh$": "setup.sh",
            r"^mcp_config\.(yaml|yml)$": "mcp_config.yaml or mcp_config.yml",
        }

        workdir_files = {f.name for f in workdir_path.iterdir() if f.is_file()}

        for pattern, description in required_patterns.items():
            pattern_re = re.compile(pattern)
            if not any(pattern_re.match(filename) for filename in workdir_files):
                raise FileNotFoundError(
                    f"Required file matching '{description}' not found in workdir_path '{workdir_path}'."
                )

        # Copy all contents of the workdir
        shutil.copytree(workdir_path, sync_dir, dirs_exist_ok=True)

    except Exception:
        shutil.rmtree(sync_dir, ignore_errors=True)
        raise

    return sync_dir


def cleanup_dir(path: Optional[Path]) -> None:
    """
    Recursively delete a directory if it exists.

    Args:
        path: Path to directory to delete. If None or doesn't exist, no-op.
    """
    if path and path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def get_setup_command() -> str:
    """Generate setup command for installing dependencies."""
    return """
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
UV_VENV_CLEAR=1 uv venv ~/venv && source ~/venv/bin/activate
uv pip install fastmcp~=2.12.0 pyyaml psutil expt-logger
bash setup.sh
"""


def get_run_command(ports: List[str]) -> str:
    """Generate command to start multiple proxy servers on different ports."""
    commands = []
    for port in ports:
        cmd = f"source ~/venv/bin/activate && python proxy_server.py --port {port} --base-dir ../workspace &"
        commands.append(cmd)
    commands.append("wait")  # Wait for all background processes
    return "\n".join(commands)
