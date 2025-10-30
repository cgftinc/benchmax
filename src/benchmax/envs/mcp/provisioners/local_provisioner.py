"""
Local provisioner for spawning proxy servers on localhost.
"""

import os
import signal
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
from .base_provisioner import BaseProvisioner
from .utils import setup_sync_dir, cleanup_dir, get_setup_command

logger = logging.getLogger(__name__)


class LocalProvisioner(BaseProvisioner):
    """
    Provisioner for spawning MCP servers locally as subprocesses.

    Use this when you want to quickly start multiple servers on your
    local machine. Useful for:
    - Development and testing without cloud infrastructure
    - Running small-scale parallel rollouts
    - Rapid iteration on tool or environment changes

    Example:
        provisioner = LocalProvisioner(
            workdir_path=Path("path/to/workdir"),
            num_servers=4,
            base_port=8080,
            api_secret="your-token"  # Optional, auto-generated if not provided
        )
    """

    def __init__(
        self,
        workdir_path: Path | str,
        num_servers: int = 1,
        base_port: int = 8080,
    ):
        """
        Initialize local provisioner.

        Args:
            workdir_path: Path to workdir containing mcp_config.yaml, reward_func.py, etc.
            num_servers: Number of servers to spawn.
            base_port: Starting port number. Each server gets base_port + index.
        """
        if num_servers < 1:
            raise ValueError("num_servers must be at least 1")
        if base_port < 1024 or base_port > 65535:
            raise ValueError("base_port must be between 1024 and 65535")
        if base_port + num_servers > 65536:
            raise ValueError(
                "Port range exceeds max port (base_port + num_servers > 65536)"
            )

        self._workdir_path = Path(workdir_path)
        self._num_servers = num_servers
        self._base_port = base_port

        self._processes: List[subprocess.Popen] = []
        self._sync_dir: Path | None = None
        self._is_provisioned = False

        logger.info(
            f"LocalProvisioner configured: {num_servers} servers, "
            f"ports {base_port}-{base_port + num_servers - 1}, "
        )

    @property
    def num_servers(self) -> int:
        """
        Total number of servers
        """
        return self._num_servers

    async def provision_servers(self, api_secret: str) -> List[str]:
        """
        Spawn proxy server subprocesses and return their addresses.

        Args:
            api_secret: Secret for server authentication.

        Returns:
            List of server addresses in "host:port" format.
        """
        if self._is_provisioned:
            raise RuntimeError("Servers already provisioned. Call teardown() first.")

        logger.info("Setting up local servers in temporary sync directory")

        # Setup synchronized directory
        try:
            self._sync_dir = setup_sync_dir(self._workdir_path)
            logger.debug(f"Created sync directory: {self._sync_dir}")
        except Exception as e:
            logger.error(f"Failed to setup sync directory: {e}")
            raise

        # Install dependencies (blocking)
        logger.info("Installing dependencies in sync directory")
        setup_cmd = get_setup_command()
        try:
            await self._spawn_process(
                cmd=setup_cmd,
                cwd=self._sync_dir,
                desc="dependency setup",
                wait=True,
                env_vars={"UV_PROJECT_ENVIRONMENT": "~/.venv"},
            )
        except Exception as e:
            cleanup_dir(self._sync_dir)
            self._sync_dir = None
            raise RuntimeError("Failed to run setup for local provisioner") from e

        # Launch servers (non-blocking)
        logger.info(
            f"Starting {self._num_servers} local servers "
            f"on ports {self._base_port}-{self._base_port + self._num_servers - 1}"
        )
        addresses: List[str] = []

        try:
            for i in range(self._num_servers):
                port = self._base_port + i
                address = f"localhost:{port}"
                # Pass API token via environment variable to each server
                cmd = f"source ~/venv/bin/activate && uv run proxy_server.py --port {port} --base-dir workspace"
                env = {"API_SECRET": api_secret}

                proc = await self._spawn_process(
                    cmd=cmd,
                    cwd=self._sync_dir,
                    desc=f"server {port}",
                    wait=False,
                    env_vars=env,
                )
                if proc is None:
                    raise RuntimeError("Failed to start a subprocess for the server.")
                self._processes.append(proc)
                addresses.append(address)
                logger.debug(f"Started server on {address} (PID: {proc.pid})")

        except Exception as e:
            logger.error("Failed to start one or more servers")
            await self._cleanup_processes()
            cleanup_dir(self._sync_dir)
            self._sync_dir = None
            raise RuntimeError(f"Failed to provision servers: {e}") from e

        self._is_provisioned = True
        logger.info(f"Successfully started {len(addresses)} local server(s)")
        return addresses

    async def teardown(self) -> None:
        """
        Terminate all spawned server processes and cleanup.
        """
        if not self._is_provisioned:
            logger.debug("LocalProvisioner teardown (nothing to clean up)")
            cleanup_dir(self._sync_dir)
            self._sync_dir = None
            return

        logger.info(f"Terminating {len(self._processes)} local server processes")
        await self._cleanup_processes()

        cleanup_dir(self._sync_dir)
        self._sync_dir = None
        self._is_provisioned = False

        logger.info("LocalProvisioner teardown complete")

    async def _spawn_process(
        self,
        cmd: str,
        cwd: Path,
        desc: str,
        wait: bool,
        env_vars: Optional[dict] = None,
    ) -> subprocess.Popen | None:
        """
        Spawn a subprocess.

        Args:
            cmd: Command to run.
            cwd: Working directory.
            desc: Description for logging.
            wait: If True, block until process completes.
                If False, return Popen immediately.
            env_vars: Optional environment variables to pass to the subprocess.

        Returns:
            subprocess.Popen object if wait=False, else None.
        """
        logger.debug(f"Running {desc}: {cmd}")

        # Merge environment variables
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Capture output only when waiting, otherwise redirect to DEVNULL
        stdout_dest = subprocess.PIPE if wait else subprocess.DEVNULL
        stderr_dest = subprocess.PIPE if wait else subprocess.DEVNULL

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=True,
            executable="/bin/bash",
            stdout=stdout_dest,
            stderr=stderr_dest,
            text=True,
            env=env,
            start_new_session=True,  # Create new process group
        )

        if wait:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                error_msg = stderr.strip() if stderr else "No error output available"
                error_msg += f"\n{stdout.strip()}"
                logger.error(
                    f"{desc} failed with return code {proc.returncode}: {error_msg}"
                )
                raise RuntimeError(f"{desc} failed")
            logger.debug(f"{desc.capitalize()} completed successfully")
            return None

        return proc

    async def _cleanup_processes(self) -> None:
        """Terminate and wait for all processes."""

        async def terminate_process(proc):
            """Terminate a single process."""
            if proc.poll() is None:
                try:
                    logger.debug(f"Terminating process group {proc.pid}")
                    # Kill entire process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    logger.debug(f"Process {proc.pid} already terminated")
                except Exception as e:
                    logger.error(f"Error terminating process {proc.pid}: {e}")

        async def kill_process(proc):
            """Force kill a single process."""
            if proc.poll() is None:
                try:
                    logger.warning(
                        f"Process {proc.pid} didn't respond to SIGTERM, sending SIGKILL"
                    )
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.error(f"Error killing process {proc.pid}: {e}")

        async def wait_process(proc):
            """Wait for a single process to terminate."""
            try:
                await asyncio.to_thread(proc.wait, timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {proc.pid} did not terminate in time")
            except Exception as e:
                logger.error(f"Error waiting for process {proc.pid}: {e}")

        if not self._processes:
            return

        # Send SIGTERM to all processes concurrently
        await asyncio.gather(
            *[terminate_process(proc) for proc in self._processes],
            return_exceptions=True,
        )

        await asyncio.sleep(0.5)

        # Send SIGKILL to any still-running processes
        await asyncio.gather(
            *[kill_process(proc) for proc in self._processes], return_exceptions=True
        )

        await asyncio.sleep(0.1)

        # Wait for all processes to finish
        await asyncio.gather(
            *[wait_process(proc) for proc in self._processes], return_exceptions=True
        )

        self._processes.clear()
