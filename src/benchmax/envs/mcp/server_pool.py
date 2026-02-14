"""
Server pool for managing MCP server lifecycle and allocation.
"""

import asyncio
import logging
from dataclasses import dataclass
import random
from typing import Coroutine, Dict, List, Optional, Set
import aiohttp
try:
    from fastmcp import Client as FastMCPClient
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "fastmcp is required for MCP environments. Install with: pip install 'benchmax[mcp]'"
    ) from e

from benchmax.envs.mcp.utils import generate_jwt_token, get_auth_headers

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """Information about a server in the pool."""

    address: str
    mcp_client: Optional[FastMCPClient]
    status: str  # "pending" | "available" | "assigned" | "recovering" | "failed"


class ServerAcquisitionError(Exception):
    """Raised when no healthy server can be acquired after all retries."""

    pass


class ServerPool:
    """
    Manages pool of servers: allocation, lifecycle, and recovery.

    Responsibilities:
    - Track available and assigned servers
    - Allocate servers to rollouts
    - Monitor server health
    - Recover failed servers
    - Clean shutdown
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        api_secret: str,
        health_check_timeout: int | float,
        initial_health_check_interval: int | float,
        max_health_check_interval: int | float,
        max_health_check_attempts: int,
        backoff_factor: float,
    ):
        """
        Initialize server pool with configuration.

        Args:
            health_check_timeout: Timeout in seconds for each health check request.
            initial_health_check_interval: Starting delay (in seconds) before first retry.
            max_health_check_interval: Upper bound for backoff delay (in seconds).
            max_health_check_attempts: Maximum number of health check attempts before giving up.
            http_session: Shared aiohttp session for HTTP requests.
            api_secret: API secret for server authentication.
            backoff_factor: Exponential backoff multiplier applied to delay after each failed attempt.
        """
        self._http_session = http_session
        self._api_secret = api_secret
        self._health_check_timeout = health_check_timeout
        self._initial_health_check_interval = initial_health_check_interval
        self._max_health_check_interval = max_health_check_interval
        self._max_health_check_attempts = max_health_check_attempts
        self._backoff_factor = backoff_factor

        self._unassigned_servers: List[ServerInfo] = []
        self._rollout_to_server: Dict[str, ServerInfo] = {}

        self._lock = asyncio.Lock()
        self._server_available = asyncio.Condition(self._lock)
        self._shutdown_event = asyncio.Event()

        self._recovery_tasks: Set[asyncio.Task] = set()

    # ===== Server Tracking =====

    async def get_server(self, rollout_id: str) -> Optional[ServerInfo]:
        """
        Get the server assigned to a rollout, if any.

        Args:
            rollout_id: ID of the rollout.

        Returns:
            ServerInfo if assigned, None otherwise.
        """
        async with self._lock:
            return self._rollout_to_server.get(rollout_id)

    # ===== Server Allocation =====

    async def acquire_server(
        self, rollout_id: str, max_attempts: int = 10
    ) -> ServerInfo:
        """
        Acquire an available server for a rollout. Blocks until available.

        Args:
            rollout_id: ID of the rollout requesting a server.
            max_attempts: Number of retry.

        Returns:
            ServerInfo for the assigned server.
        """
        for _ in range(max_attempts):
            # Retry until a healthy server is returned
            async with self._server_available:
                # Wait until a server is available
                while not self._unassigned_servers:
                    logger.debug(
                        f"[{rollout_id}] Waiting for available server "
                        f"(pool: 0 available, {len(self._rollout_to_server)} assigned)"
                    )
                    await self._server_available.wait()

                # Acquire the first available server
                server = self._unassigned_servers.pop(0)

                # Check if server is still responsive
                is_healthy = await self._check_server_health(server.address)
                if not is_healthy:
                    asyncio.create_task(self._release_server_impl(server))
                    # retry
                    continue

                # Connect to the MCP server
                if server.mcp_client and server.mcp_client.is_connected():
                    try:
                        await server.mcp_client._disconnect()
                    except Exception:
                        pass  # ok if disconnect fails

                logger.debug(f"Server {server.address} connecting MCP client")
                server.mcp_client = FastMCPClient(
                    f"http://{server.address}/mcp",
                    auth=generate_jwt_token(
                        api_secret=self._api_secret,
                        rollout_id=rollout_id,
                        expiration_seconds=60 * 30,  # 30 minutes
                    ),
                )

                try:
                    await server.mcp_client._connect()
                except Exception:
                    asyncio.create_task(self._release_server_impl(server))
                    # retry
                    continue

                server.status = "assigned"
                self._rollout_to_server[rollout_id] = server

                logger.debug(
                    f"Server {server.address} assigned to [{rollout_id}] "
                    f"(pool: {len(self._unassigned_servers)} available, {len(self._rollout_to_server)} assigned)"
                )

                return server

        raise ServerAcquisitionError(
            f"Failed to acquire a healthy server for rollout [{rollout_id}] "
            f"after {max_attempts} attempts."
        )

    async def add_server_once_ready(self, address: str) -> None:
        """
        Add server to the pool once it is ready.

        This method waits for the server to become healthy and then
        adds it to the available pool.

        Args:
            address: Server address in "host:port" format.
        """
        try:
            logger.debug(f"Server {address} queued for health check")

            # Create a placeholder ServerInfo
            server_info = ServerInfo(address=address, mcp_client=None, status="pending")

            # Track the recovery task
            self._create_recovery_task(self._recover_server(server_info))

        except asyncio.CancelledError:
            logger.debug(f"Server {address} addition cancelled")
        except Exception as e:
            logger.error(f"Failed to queue server {address}: {str(e)}")
            raise

    async def release_server(self, rollout_id: str) -> None:
        """
        Release a server after rollout completes. Triggers reset and recovery.

        Args:
            rollout_id: ID of the rollout releasing the server.
        """
        logger.debug(f"[{rollout_id}] Releasing server")

        async with self._lock:
            server = self._rollout_to_server.pop(rollout_id, None)
            if server is None:
                logger.debug(f"[{rollout_id}] No server to release")
                return

        await self._release_server_impl(server, rollout_id=rollout_id)

    async def _release_server_impl(
        self, server: ServerInfo, rollout_id: Optional[str] = None
    ):
        try:
            logger.debug(
                f"Server {server.address} released from [{rollout_id}], resetting..."
            )
            server.status = "recovering"

            # Disconnect MCP client if still available and connected
            if server.mcp_client and server.mcp_client.is_connected():
                await server.mcp_client._disconnect()

            # Reset and recover the server
            await self._reset_server(server.address)

            # Let the reset kick in before attempting health check
            await asyncio.sleep(0.5)

            # Track recovery tasks
            self._create_recovery_task(self._recover_server(server))
        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Server release cancelled")

    async def report_server_failure(self, rollout_id: str, reason: str) -> None:
        """
        Report that an assigned server has failed. Triggers recovery.

        Args:
            rollout_id: ID of the rollout reporting the failure.
            reason: Description of the failure.
        """
        logger.warning(f"[{rollout_id}] Server failure reported: {reason}")
        try:
            async with self._lock:
                server = self._rollout_to_server.pop(rollout_id, None)
                if server is None:
                    logger.debug(
                        f"[{rollout_id}] No server assigned for failure report"
                    )
                    return

                logger.warning(
                    f"Server {server.address} failed during [{rollout_id}], initiating recovery"
                )
                server.status = "recovering"

            # Check if server is still healthy
            if await self._check_server_health(server.address):
                # Server is up, reset it
                logger.debug(f"Server {server.address} still responsive, resetting")
                # Disconnect MCP client if still available and connected
                if server.mcp_client and server.mcp_client.is_connected():
                    await server.mcp_client._disconnect()
                await self._reset_server(server.address)
                # Let the reset kick in before attempting health check
                await asyncio.sleep(0.5)
            else:
                logger.debug(
                    f"Server {server.address} unresponsive, waiting for recovery"
                )

            # Wait for recovery
            self._create_recovery_task(self._recover_server(server))
        except asyncio.CancelledError:
            logger.debug(f"[{rollout_id}] Failure recovery cancelled")

    async def add_server_to_available_pool(self, server: ServerInfo):
        """
        Add a server to the available pool and notify waiting tasks.

        Args:
            server: ServerInfo to add to the pool.
        """
        async with self._server_available:
            server.status = "available"
            self._unassigned_servers.append(server)
            self._server_available.notify_all()
            logger.debug(
                f"Server {server.address} ready "
                f"(pool: {len(self._unassigned_servers)} available, {len(self._rollout_to_server)} assigned)"
            )

    # ===== Server Shutdown =====

    async def shutdown(self, client_disconnect_timeout: int | float = 20) -> None:
        """Shutdown pool and cancel all recovery tasks."""
        logger.info("Shutting down server pool...")

        # Signal shutdown to all waiting tasks
        self._shutdown_event.set()

        # Cancel and wait for all tracked recovery tasks
        if self._recovery_tasks:
            logger.debug(f"Cancelling {len(self._recovery_tasks)} recovery tasks")
            for t in list(self._recovery_tasks):
                t.cancel()
            await asyncio.gather(*list(self._recovery_tasks), return_exceptions=True)
            self._recovery_tasks.clear()

        # Disconnect all MCP clients
        all_servers = list(self._unassigned_servers) + list(
            self._rollout_to_server.values()
        )
        disconnect_coros = []
        for server in all_servers:
            if server.mcp_client and server.mcp_client.is_connected():

                async def _safe_disconnect(client: FastMCPClient):
                    try:
                        await asyncio.wait_for(
                            client._disconnect(), timeout=client_disconnect_timeout
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.debug(f"Ignoring error during disconnect: {str(e)}")

                disconnect_coros.append(_safe_disconnect(server.mcp_client))

        if disconnect_coros:
            logger.debug(f"Disconnecting {len(disconnect_coros)} MCP clients")
            await asyncio.gather(*disconnect_coros, return_exceptions=True)

        # Clear all server tracking
        async with self._lock:
            self._unassigned_servers.clear()
            self._rollout_to_server.clear()

        logger.info("Server pool shutdown complete")

    # ===== Internal Methods =====

    async def _reset_server(self, address: str) -> None:
        """
        Reset a server via its reset endpoint.

        Args:
            address: Server address in "host:port" format.
        """
        reset_url = f"http://{address}/reset"
        headers = get_auth_headers(self._api_secret)
        timeout = aiohttp.ClientTimeout(total=self._health_check_timeout)

        try:
            async with self._http_session.post(
                reset_url, headers=headers, timeout=timeout
            ) as response:
                if response.status == 200:
                    logger.debug(f"Server {address} reset successful")
                else:
                    logger.warning(
                        f"Server {address} reset returned status {response.status}"
                    )
        except Exception as e:
            logger.warning(f"Server {address} reset failed: {str(e)}")

    async def _recover_server(self, server: ServerInfo) -> None:
        """
        Wait for server to come back online and return it to the pool.

        Args:
            server: ServerInfo for the server to recover.
        """
        try:
            server.mcp_client = None

            logger.debug(f"Server {server.address} recovery started")
            await self._wait_till_server_online(server.address)

            server.mcp_client = None
            server.status = "available"

            # Add server back to available pool
            await self.add_server_to_available_pool(server)

        except asyncio.CancelledError:
            logger.debug(f"Server {server.address} recovery cancelled")
        except Exception as e:
            logger.error(
                f"Server {server.address} recovery failed permanently {e} "
                f"(pool: {len(self._unassigned_servers)} available, "
                f"{len(self._rollout_to_server)} assigned)",
                exc_info=True,
            )
            server.status = "failed"

    async def _check_server_health(self, address: str) -> bool:
        """
        Check if a server is healthy.

        Args:
            address: Server address in "host:port" format.

        Returns:
            True if server responds to health check, False otherwise.
        """
        health_url = f"http://{address}/health"
        timeout = aiohttp.ClientTimeout(total=self._health_check_timeout)
        try:
            async with self._http_session.get(health_url, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    async def _wait_till_server_online(self, address: str) -> None:
        """
        Wait for server to become healthy using expoential backoff

        Args:
            address: Server address in "host:port" format.

        Raises:
            asyncio.CancelledError: If shutdown is initiated.
            TimeoutError: If max health check attempts exceeded.
        """
        for attempt in range(1, self._max_health_check_attempts + 1):
            if self._shutdown_event.is_set():
                raise asyncio.CancelledError("Shutdown initiated")

            is_alive = await self._check_server_health(address)
            if is_alive:
                logger.debug(
                    f"Server {address} health check passed (attempt {attempt})"
                )
                return

            if attempt % 5 == 0:
                logger.debug(
                    f"Server {address} health check attempt {attempt}/{self._max_health_check_attempts}"
                )

            # Exponential backoff with jitter
            delay = min(
                self._max_health_check_interval,
                self._initial_health_check_interval
                * (self._backoff_factor ** (attempt - 1)),
            )
            jitter = random.uniform(0, delay * 0.25)
            await asyncio.sleep(delay + jitter)

        raise TimeoutError(
            f"Server {address} failed to become healthy after {self._max_health_check_attempts} attempts"
        )

    def _create_recovery_task(self, coro: Coroutine) -> asyncio.Task:
        """
        Create a recovery task and track it so we can cancel/await on shutdown.

        Args:
            coro: Coroutine to run as a task.

        Returns:
            The created task.
        """
        task = asyncio.create_task(coro)

        def _on_done(t: asyncio.Task):
            # Remove from the set so shutdown won't try to wait for it forever
            self._recovery_tasks.discard(t)
            # Ensure exceptions are observed
            try:
                exc = t.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    logger.debug(f"Recovery task finished with exception: {exc}")
            except asyncio.CancelledError:
                # Cancelled tasks raise when calling .exception(); ignore
                pass
            except Exception as e:
                # Defensive: if exception() itself fails, log and continue
                logger.warning(f"Failed to inspect recovery task: {str(e)}")

        task.add_done_callback(_on_done)
        self._recovery_tasks.add(task)
        return task
