"""
Test that ProxyServer._server_reset() properly kills the spawned MCP subprocess.

ProxyServer._server_reset() calls client.close() to shut down the MCP
subprocess. This test verifies the subprocess is actually terminated.
"""

import os
import sys
import types
import asyncio
import psutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import proxy_server directly to avoid benchmax.envs.mcp.__init__ pulling in
# skypilot (which fails due to local sky config issues unrelated to this test).
import importlib.util

# Mock the reward_fn module before importing proxy_server
sys.modules["reward_fn"] = types.ModuleType("reward_fn")
sys.modules["reward_fn"].reward_functions = {}  # type: ignore[attr-defined]

_spec = importlib.util.spec_from_file_location(
    "proxy_server",
    Path(__file__).resolve().parents[4]
    / "src"
    / "benchmax"
    / "envs"
    / "mcp"
    / "proxy_server.py",
)
assert _spec and _spec.loader
_proxy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proxy_mod)

ProxyServer = _proxy_mod.ProxyServer


def _find_demo_server_pids() -> set[int]:
    """Find PIDs of running demo_mcp_server.py processes owned by us."""
    pids = set()
    my_uid = os.getuid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("demo_mcp_server.py" in arg for arg in cmdline):
                if proc.uids().real == my_uid:
                    pids.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
async def test_server_reset_kills_subprocess(example_workdir: Path):
    """ProxyServer._server_reset() should not leave zombie MCP subprocesses."""
    before = _find_demo_server_pids()

    server = ProxyServer(base_dir=example_workdir / "workspace")
    server.workspace = example_workdir / "workspace" / "test"
    server.workspace.mkdir(parents=True, exist_ok=True)

    # Build MCP config the same way _setup() does via load_config + Client()
    config = {
        "mcpServers": {
            "test_server": {
                "command": "python",
                "args": [str(example_workdir / "demo_mcp_server.py")],
                "cwd": str(server.workspace),
            }
        }
    }
    from fastmcp import Client

    server.client = Client(config)
    await server.client._connect()

    spawned = _find_demo_server_pids() - before
    assert spawned, "Expected a demo_mcp_server.py subprocess after connect"

    # Patch asyncio.create_task so the do_reset() coroutine (which calls
    # os.execv) is never actually scheduled — it would re-exec the test runner.
    # Also patch shutil.rmtree to skip workspace cleanup.
    with (
        patch.object(_proxy_mod.asyncio, "create_task", new=MagicMock()),
        patch.object(_proxy_mod.shutil, "rmtree"),
    ):
        await server._server_reset()

    await asyncio.sleep(1)

    still_alive = {pid for pid in spawned if psutil.pid_exists(pid)}
    # cleanup so we don't leave processes around regardless
    for pid in still_alive:
        try:
            os.kill(pid, 9)
        except OSError:
            pass

    assert not still_alive, (
        f"Zombie subprocess(es) {still_alive} survived _server_reset()"
    )
