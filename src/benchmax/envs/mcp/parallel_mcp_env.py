"""
Parallel MCP Environment for managing distributed tool execution.
"""

import asyncio
import logging
import uuid
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import warnings
import aiohttp
from mcp.types import TextContent
try:
    from fastmcp.exceptions import ToolError
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "fastmcp is required for MCP environments. Install with: pip install 'benchmax[mcp]'"
    ) from e

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition
from .server_pool import ServerPool
from .provisioners.base_provisioner import BaseProvisioner
from .utils import (
    apply_fastmcp_patch,
    convert_tool_definitions,
    get_auth_headers,
    upload_form,
    download_file,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp").setLevel(logging.CRITICAL)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

apply_fastmcp_patch()


class ParallelMcpEnv(BaseEnv):
    """
    Parallel MCP Environment for distributed tool execution.

    Manages a pool of MCP servers to run multiple rollouts concurrently,
    with automatic server health monitoring and recovery.

    Workdir should contain:
    - reward_fn.py       # defines reward computation functions
    - mcp_config.yaml    # MCP server and tool configuration
    - optional tool files and static resources needed for tool execution

    Example usage:
    - Initialize rollout
    - Run tools concurrently across servers
    - Compute rewards for rollout outputs
    """

    def __init__(
        self,
        workdir_path: str | Path,
        provisioner: BaseProvisioner,
        api_secret: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        output_parsers: Optional[Dict[str, Callable[[str], Any]]] = None,
        health_check_timeout: int | float = 5,
        initial_health_check_interval: int | float = 2,
        max_health_check_interval: int | float = 15,
        max_health_check_attempts: int = 20,
        backoff_factor: float = 2.0,
        provision_at_init: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the parallel MCP environment.

        Args:
            workdir_path: Path to workdir containing mcp_config.yaml, setup.sh,
                         reward_func.py, and other server-side files.
            provisioner: Provisioner instance for managing server lifecycle.
            api_secret: API secret for JWT signing. If not provided, generates a random one.
            allowed_tools: Optional whitelist of tool names. If provided, only
                          these tools will be available.
            output_parsers: Optional dict mapping tool names to parser functions
                           that process tool output.
            health_check_timeout: Timeout in seconds for health check requests.
            health_check_interval: Interval in seconds between health check retries.
            max_health_check_attempts: Maximum number of health check attempts
                                      before marking a server as failed.
            provision_at_init: Whether to launch a server at the point of initialization
            **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__(**kwargs)

        self._workdir_path = Path(workdir_path).absolute()
        self._provisioner = provisioner
        self._allowed_tools = allowed_tools
        self._output_parsers = output_parsers

        # Generate or use provided API secret for JWT signing
        self._api_secret = api_secret or uuid.uuid4().hex

        # Cached tool definitions
        self._tool_definitions: Optional[List[ToolDefinition]] = None

        # Set server pool configs
        self._health_check_timeout = health_check_timeout
        self._initial_health_check_interval = initial_health_check_interval
        self._max_health_check_interval = max_health_check_interval
        self._max_health_check_attempts = max_health_check_attempts
        self._backoff_factor = backoff_factor

        # Will initialize aiohttp and server pool later as they require event loop
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._server_pool: Optional[ServerPool] = None

        # Track if servers have been provisioned
        self._servers_provisioned = False
        self._provision_lock = asyncio.Lock()

        logger.debug(f"ParallelMcpEnv initialized with workdir: {self._workdir_path}")

        if provision_at_init:
            asyncio.create_task(self._ensure_servers_provisioned())

    # ===== Server Management =====

    async def _ensure_servers_provisioned(self) -> None:
        """Ensure servers are provisioned before use."""
        if self._servers_provisioned:
            return

        async with self._provision_lock:
            # Double-check after acquiring lock
            if self._servers_provisioned:
                return

            logger.info("Provisioning servers...")
            addresses = await self._provisioner.provision_servers(self._api_secret)

            # Shared HTTP session
            self._http_session = aiohttp.ClientSession()

            # Server pool for managing server lifecycle
            self._server_pool = ServerPool(
                http_session=self._http_session,
                api_secret=self._api_secret,
                health_check_timeout=self._health_check_timeout,
                initial_health_check_interval=self._initial_health_check_interval,
                max_health_check_interval=self._max_health_check_interval,
                max_health_check_attempts=self._max_health_check_attempts,
                backoff_factor=self._backoff_factor,
            )

            # Add all servers to the pool
            for address in addresses:
                asyncio.create_task(self._server_pool.add_server_once_ready(address))

            logger.info(f"Initiated provisioning of {len(addresses)} servers")
            self._servers_provisioned = True

    @property
    def num_servers(self) -> int:
        """Total number of servers"""
        return self._provisioner.num_servers

    async def shutdown(self) -> None:
        """Tear down all resources and cleanup."""
        logger.info("Shutting down ParallelMcpEnv...")
        try:
            # Shutdown server pool
            if self._server_pool:
                await self._server_pool.shutdown()

            # Close HTTP session
            if self._http_session:
                await self._http_session.close()

            # Teardown provisioner
            await self._provisioner.teardown()

            logger.info("ParallelMcpEnv shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise

    # ===== Tool Discovery =====

    async def list_tools(self) -> List[ToolDefinition]:
        """
        List available tools from MCP server.

        Returns:
            List of ToolDefinition objects describing available tools.
        """
        if self._tool_definitions is not None:
            logger.debug(
                f"Returning cached tool definitions ({len(self._tool_definitions)} tools)"
            )
            return self._tool_definitions

        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        # Acquire a temporary server to list tools
        temp_rollout_id = "initial_list_tool"
        server_info = await self._server_pool.acquire_server(temp_rollout_id)

        try:
            if server_info.mcp_client is None:
                raise RuntimeError(
                    "MCP client is not present to list tools. This should not happen."
                )

            tools = await server_info.mcp_client.list_tools()
            self._tool_definitions = convert_tool_definitions(
                tools, self._allowed_tools
            )

            tool_names = [t.name for t in self._tool_definitions]
            logger.info(
                f"Discovered {len(self._tool_definitions)} tools: {', '.join(tool_names)}"
            )

            return self._tool_definitions
        finally:
            # Return server directly to the pool (no state was set)
            if server_info:
                await self._server_pool.add_server_to_available_pool(server_info)

    # ===== Rollout Lifecycle =====

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """
        Initialize a rollout by acquiring a server from the pool.

        Args:
            rollout_id: Unique identifier for this rollout.
            **rollout_args: Additional rollout arguments (currently unused).

        Raises:
            RuntimeError: If rollout_id is already initialized.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        # Check if rollout_id already has a server
        existing_server = await self._server_pool.get_server(rollout_id)
        if existing_server is not None:
            raise RuntimeError(f"Rollout '{rollout_id}' is already initialized")

        assigned_server = await self._server_pool.acquire_server(rollout_id)
        logger.info(f"[{rollout_id}] Initialized on server {assigned_server.address}")

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> str:
        """
        Execute a tool in the rollout's workspace.

        Args:
            rollout_id: ID of the rollout to run the tool in.
            tool_name: Name of the tool to execute.
            **tool_args: Arguments to pass to the tool.

        Returns:
            Tool execution result as a string.

        Raises:
            RuntimeError: If rollout is not initialized or server is unavailable.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        server_info = await self._server_pool.get_server(rollout_id)
        if server_info is None:
            raise RuntimeError(f"Rollout '{rollout_id}' not initialized")

        if server_info.mcp_client is None or not server_info.mcp_client.is_connected():
            await self._server_pool.report_server_failure(
                rollout_id, reason="MCP client disconnected"
            )
            raise RuntimeError(
                f"[{rollout_id}] MCP client unavailable on {server_info.address}"
            )

        logger.debug(
            f"[{rollout_id}] Running tool '{tool_name}' with args: {list(tool_args.keys())}"
        )

        try:
            content_list = (
                await server_info.mcp_client.call_tool(
                    tool_name, tool_args, timeout=datetime.timedelta(seconds=30)
                )
            ).content

            text_content = []
            # Process content based on type
            for content in content_list:
                if isinstance(content, TextContent):
                    text_content.append(content.text)

            combined_text = "\n".join(text_content)

            # Apply output parser if available
            if (
                self._output_parsers
                and tool_name in self._output_parsers
                and isinstance(combined_text, str)
            ):
                combined_text = self._output_parsers[tool_name](combined_text)

            logger.debug(f"[{rollout_id}] Tool '{tool_name}' completed successfully")
            return combined_text

        except ToolError as e:
            logger.warning(
                f"[{rollout_id}] Failed tool call '{tool_name}' encountered ToolError: {str(e)}"
            )
            return str(e)
        except Exception as e:
            exc_type = type(e).__name__
            logger.warning(
                f"[{rollout_id}] Failed tool call '{tool_name}' encountered {exc_type}: {str(e)}"
            )
            return str(e)

    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        """
        Compute reward and cleanup rollout.

        Raises:
            RuntimeError: If rollout is not initialized or computation fails.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        if not self._http_session:
            raise RuntimeError("Http session has not been initialized")

        server_info = await self._server_pool.get_server(rollout_id)
        if server_info is None:
            raise RuntimeError(f"Rollout '{rollout_id}' not initialized")

        # Generate JWT token with rollout_id claim
        headers = get_auth_headers(self._api_secret, rollout_id)
        headers["Content-Type"] = "application/json"

        compute_reward_url = f"http://{server_info.address}/compute_reward"
        payload = {
            "completion": completion or "",
            "ground_truth": ground_truth or "",
            **kwargs,
        }

        logger.debug(f"[{rollout_id}] Computing reward")

        try:
            async with self._http_session.post(
                compute_reward_url, headers=headers, json=payload
            ) as response:
                text = await response.text()
                if response.status == 200:
                    try:
                        result: Dict[str, float] = await response.json()
                        logger.info(f"[{rollout_id}] Completed with rewards: {result}")
                        await self.release_rollout(rollout_id)
                        return result
                    except Exception:
                        raise RuntimeError("Failed to parse reward response as JSON")

                # Try parsing structured JSON error
                try:
                    error_data = await response.json()
                    error_msg = (
                        f"Error {error_data.get('error')}: {error_data.get('detail')}"
                    )
                except Exception:
                    error_msg = text.strip()

                raise RuntimeError(
                    f"Reward computation failed with status {response.status}: {error_msg}"
                )

        except Exception as e:
            logger.error(f"[{rollout_id}] Failed to compute reward: {str(e)}")
            await self._server_pool.report_server_failure(
                rollout_id, reason=f"Reward computation failed: {str(e)}"
            )
            raise

    async def release_rollout(self, rollout_id: str):
        """
        Releases a rollout from the server pool which triggers the server to reset.
        Mostly used internally by compute_reward to clean up the rollout.

        Args:
            rollout_id: ID of the rollout.
        """
        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        await self._server_pool.release_server(rollout_id)

    # ===== Workspace File Management =====

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """
        Upload a file to the workspace for a specific rollout.

        Args:
            rollout_id: ID of the rollout.
            src_path: Local path to the file to upload.
            dst_filename: Optional destination filename. If None, uses src_path.name.

        Raises:
            RuntimeError: If rollout is not initialized or upload fails.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        if not self._http_session:
            raise RuntimeError("Http session has not been initialized")

        server_info = await self._server_pool.get_server(rollout_id)
        if server_info is None:
            raise RuntimeError(f"Rollout '{rollout_id}' not initialized")

        upload_url = f"http://{server_info.address}/upload"
        filename = dst_filename or src_path.name

        logger.debug(f"[{rollout_id}] Uploading file: {src_path} -> {filename}")

        try:
            with open(src_path, "rb") as f:
                file_bytes = f.read()
            await upload_form(
                self._http_session,
                upload_url,
                api_secret=self._api_secret,
                file_bytes=file_bytes,
                filename=filename,
            )
            logger.debug(
                f"[{rollout_id}] File uploaded: {filename} ({len(file_bytes)} bytes)"
            )
        except Exception as e:
            logger.error(f"[{rollout_id}] Failed to upload {src_path}: {str(e)}")
            raise

    async def copy_content_to_workspace(
        self,
        rollout_id: str,
        src_content: str | bytes,
        dst_filename: str,
        encoding: str = "utf-8",
    ) -> None:
        """
        Upload text or binary content as a file in the workspace.

        Args:
            rollout_id: ID of the rollout.
            src_content: Content to upload (string or bytes).
            dst_filename: Destination filename in the workspace.
            encoding: Encoding to use if src_content is a string.

        Raises:
            RuntimeError: If rollout is not initialized or upload fails.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        if not self._http_session:
            raise RuntimeError("Http session has not been initialized")

        server_info = await self._server_pool.get_server(rollout_id)
        if server_info is None:
            raise RuntimeError(f"Rollout '{rollout_id}' not initialized")

        upload_url = f"http://{server_info.address}/upload"

        logger.debug(f"[{rollout_id}] Uploading content to: {dst_filename}")

        try:
            if isinstance(src_content, str):
                file_bytes = src_content.encode(encoding)
                content_type = "text/plain"
            else:
                file_bytes = src_content
                content_type = "application/octet-stream"

            await upload_form(
                self._http_session,
                upload_url,
                api_secret=self._api_secret,
                file_bytes=file_bytes,
                filename=dst_filename,
                content_type=content_type,
            )
            logger.debug(
                f"[{rollout_id}] Content uploaded: {dst_filename} ({len(file_bytes)} bytes)"
            )
        except Exception as e:
            logger.error(
                f"[{rollout_id}] Failed to upload content to {dst_filename}: {str(e)}"
            )
            raise

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """
        Download a file from the workspace for a specific rollout.

        Args:
            rollout_id: ID of the rollout.
            src_filename: Filename in the workspace to download.
            dst_path: Local path to save the downloaded file.

        Raises:
            RuntimeError: If rollout is not initialized or download fails.
        """
        # Ensure servers are provisioned
        await self._ensure_servers_provisioned()

        # Check that server pool is initialized
        if not self._server_pool:
            raise RuntimeError("Server pool failed to initialize.")

        if not self._http_session:
            raise RuntimeError("Http session has not been initialized")

        server_info = await self._server_pool.get_server(rollout_id)
        if server_info is None:
            raise RuntimeError(f"Rollout '{rollout_id}' not initialized")

        download_url = f"http://{server_info.address}/download"
        params = {"file_path": src_filename}

        logger.debug(f"[{rollout_id}] Downloading file: {src_filename} -> {dst_path}")

        try:
            await download_file(
                self._http_session,
                download_url,
                api_secret=self._api_secret,
                params=params,
                dst_path=dst_path,
            )
            logger.debug(f"[{rollout_id}] File downloaded: {src_filename}")
        except Exception as e:
            logger.error(f"[{rollout_id}] Failed to download {src_filename}: {str(e)}")
            raise
