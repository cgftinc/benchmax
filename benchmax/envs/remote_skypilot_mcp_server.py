import asyncio
import aiohttp
import uuid
import yaml
import tempfile
from typing import List, Any, Optional, Dict
from pathlib import Path
from fastmcp import Client as FastMCPClient
from mcp import Tool
import sky
import logging

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition

logger = logging.getLogger(__name__)


class RemoteSkypilotMcpEnv(BaseEnv):
    """Remote MCP Environment for managing tool execution and rollouts with a remote MCP server.
    Currently only supports running on Skypilot containers.
    """
    def __init__(
        self,
        skypilot_yaml_path: str,
        num_nodes: int = 1,
        allowed_tools: Optional[List[str]] = None,
        cluster_name: str = "benchmax-env-cluster",
        health_check_timeout: int = 300,  # 5 minutes
        health_check_interval: int = 10,  # 10 seconds
    ) -> None:
        """Initialize the environment with configuration and pool settings."""
        super().__init__()
        self._allowed_tools = allowed_tools or []
        self._cluster_name = cluster_name
        self._health_check_timeout = health_check_timeout
        self._health_check_interval = health_check_interval

        # Generate API token for worker authentication
        self._api_token = uuid.uuid4().hex
        
        # Worker management
        self._client_pool: Dict[str, FastMCPClient] = {}
        self._available_workers: asyncio.Queue[str] = asyncio.Queue()
        self._rollout_to_worker: Dict[str, str] = {}
        self._worker_init_tasks: List[asyncio.Task] = []
        
        # HTTP session for file operations
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # Cached tool definitions
        self._tool_definitions: Optional[List[ToolDefinition]] = None

        # Launch workers and start initialization
        self._launch_workers(skypilot_yaml_path, num_nodes)


    def _launch_workers(self, skypilot_yaml_path: str, num_nodes: int) -> None:
        """Launch SkyPilot workers synchronously with modified YAML."""
        # Create modified YAML with API token
        task = sky.Task.from_yaml(skypilot_yaml_path)
        task.num_nodes = num_nodes
        task.update_envs({"API_TOKEN": self._api_token})
        _, handle = sky.launch(
            task, 
            cluster_name=self._cluster_name, 
            detach_run=True, 
            detach_setup=True, 
            retry_until_up=True
        )   
        if handle is None:
            raise RuntimeError("Failed to launch SkyPilot task.")
        
        worker_ips = [
            external_ip for _, external_ip in handle.stable_internal_external_ips
        ]
        logger.info(f"Launched workers with IPs: {worker_ips}")
        
        # Start background initialization for each worker
        for worker_ip in worker_ips:
            task = asyncio.create_task(self._init_worker(worker_ip))
            self._worker_init_tasks.append(task)

    async def _init_worker(self, worker_ip: str) -> None:
        """Initialize a single worker: health check + FastMCP client + add to pool."""
        try:
            # Health check
            await self._wait_for_worker_health(worker_ip)
            
            # Initialize FastMCP client
            mcp_url = f"http://{worker_ip}:8080/mcp"
            client = FastMCPClient(mcp_url)
            await client._connect()
            self._client_pool[worker_ip] = client
            
            # Add to available pool
            await self._available_workers.put(worker_ip)
            logger.info(f"Worker {worker_ip} initialized and added to pool")
            
        except Exception as e:
            logger.error(f"Failed to initialize worker {worker_ip}: {e}")
            # Don't re-raise - let other workers continue

    async def _wait_for_worker_health(self, worker_ip: str) -> None:
        """Wait for worker to pass health check."""
        health_url = f"http://{worker_ip}:8080/health"
        start_time = asyncio.get_event_loop().time()
        
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self._health_check_timeout:
                raise TimeoutError(f"Health check timeout for worker {worker_ip}")
            
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with self._http_session.get(health_url, timeout=timeout) as response:
                    if response.status == 200:
                        logger.info(f"Worker {worker_ip} is healthy")
                        return
                    else:
                        logger.debug(f"Worker {worker_ip} health check returned {response.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(f"Health check failed for {worker_ip}: {e}")

            await asyncio.sleep(self._health_check_interval)

    async def _get_available_worker(self) -> str:
        """Get an available worker, blocking until one is ready."""
        return await self._available_workers.get()

    async def _release_worker(self, worker_ip: str) -> None:
        """Return a worker to the available pool."""
        await self._available_workers.put(worker_ip)

    async def _call_worker_reset(self, worker_ip: str, preserve_as: Optional[str] = None) -> None:
        """Call the reset endpoint on a specific worker."""
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        
        reset_url = f"http://{worker_ip}:8080/reset"
        headers = {"Authorization": self._api_token}
        params = {}
        if preserve_as:
            params["preserve_as"] = preserve_as
        
        try:
            async with self._http_session.post(reset_url, headers=headers, params=params) as response:
                if response.status == 200:
                    logger.info(f"Reset successful for worker {worker_ip}")
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Reset failed for worker {worker_ip}: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Reset request failed for worker {worker_ip}: {e}")

    def _convert_and_filter_tools(self, tools: List[Tool]) -> List[ToolDefinition]:
        """Convert Tool objects to ToolDefinition objects and filter based on allowed list."""
        tool_definitions = [
            ToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema
            )
            for tool in tools
        ]
        
        if not self._allowed_tools:
            return tool_definitions
        
        return [tool for tool in tool_definitions if tool.name in self._allowed_tools]

    # ---- Public API Methods ----
    
    async def shutdown(self) -> None:
        """Clean up resources - stop all tasks and close clients."""
        # Cancel worker initialization tasks
        for task in self._worker_init_tasks:
            if not task.done():
                task.cancel()
        
        if self._worker_init_tasks:
            results = await asyncio.gather(*self._worker_init_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    logger.error(f"Error in worker init task {i}: {result}")

        # Close FastMCP clients
        if self._client_pool:
            close_tasks = [client.close() for client in self._client_pool.values()]
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    worker_ip = list(self._client_pool.keys())[i]
                    logger.error(f"Error closing FastMCP client for {worker_ip}: {result}")

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        # Tear down SkyPilot cluster
        sky.down(cluster_name=self._cluster_name)

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools, using cached definitions if available."""
        if self._tool_definitions is not None:
            return self._tool_definitions

        # Get any available worker to fetch tools
        worker_ip = await self._get_available_worker()
        try:
            client = self._client_pool[worker_ip]
            tools = await client.list_tools()
            self._tool_definitions = self._convert_and_filter_tools(tools)
            return self._tool_definitions
        finally:
            await self._release_worker(worker_ip)

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize resources for a new rollout - assigns a worker to the rollout."""
        if rollout_id in self._rollout_to_worker:
            raise ValueError(f"Rollout {rollout_id} is already initialized")
        
        # Get an available worker (blocks until one is ready)
        worker_ip = await self._get_available_worker()
        
        # Assign worker to rollout
        self._rollout_to_worker[rollout_id] = worker_ip
        logger.info(f"Rollout {rollout_id} assigned to worker {worker_ip}")

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute a tool in the context of a specific rollout."""
        if rollout_id not in self._rollout_to_worker:
            raise ValueError(f"Rollout {rollout_id} is not initialized. Call init_rollout() first.")
        
        worker_ip = self._rollout_to_worker[rollout_id]
        client = self._client_pool[worker_ip]
        
        try:
            result = await client.call_tool(tool_name, tool_args)
            logger.debug(f"Tool {tool_name} executed successfully for rollout {rollout_id}")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name} on rollout {rollout_id}: {e}")
            raise

    async def cleanup_rollout(self, rollout_id: str, keep_workspace: bool = False) -> None:
        """Clean up rollout resources and return worker to pool."""
        if rollout_id not in self._rollout_to_worker:
            raise ValueError(f"Rollout {rollout_id} is not initialized")
        
        worker_ip = self._rollout_to_worker[rollout_id]
        
        try:
            # Call reset endpoint
            preserve_as = rollout_id if keep_workspace else None
            await self._call_worker_reset(worker_ip, preserve_as=preserve_as)
            logger.info(f"Rollout {rollout_id} cleaned up (keep_workspace={keep_workspace})")
        finally:
            # Always release worker back to pool and remove assignment
            del self._rollout_to_worker[rollout_id]
            await self._release_worker(worker_ip)

    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """Copy a file to the workspace for a specific rollout."""
        if rollout_id not in self._rollout_to_worker:
            raise ValueError(f"Rollout {rollout_id} is not initialized")
        
        worker_ip = self._rollout_to_worker[rollout_id]
        upload_url = f"http://{worker_ip}:8080/upload"
        headers = {"Authorization": self._api_token}
        
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        
        # Prepare file for upload
        filename = dst_filename or src_path.name
        
        try:
            with open(src_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=filename)
                
                async with self._http_session.post(upload_url, headers=headers, data=data) as response:
                    if response.status == 200:
                        logger.info(f"File {src_path} uploaded as {filename} for rollout {rollout_id}")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Upload failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to workspace for rollout {rollout_id}: {e}")
            raise

    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """Copy a file from the workspace for a specific rollout."""
        if rollout_id not in self._rollout_to_worker:
            raise ValueError(f"Rollout {rollout_id} is not initialized")
        
        worker_ip = self._rollout_to_worker[rollout_id]
        download_url = f"http://{worker_ip}:8080/download"
        headers = {"Authorization": self._api_token}
        params = {"file_path": src_filename}
        
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        
        try:
            async with self._http_session.get(download_url, headers=headers, params=params) as response:
                if response.status == 200:
                    # Ensure destination directory exists
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write file content
                    with open(dst_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info(f"File {src_filename} downloaded from rollout {rollout_id} to {dst_path}")
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Download failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Failed to copy {src_filename} from workspace for rollout {rollout_id}: {e}")
            raise


async def main():
    env = RemoteSkypilotMcpEnv(skypilot_yaml_path="benchmax/envs/skypilot/skypilot_fastmcp.yaml")
    
    try:
        # Wait a bit for workers to come online, then list tools
        await asyncio.sleep(30)  # Give workers time to initialize
        tools = await env.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Example rollout workflow
        rollout_id = "test-rollout-001"
        await env.init_rollout(rollout_id)
        print(f"Initialized rollout: {rollout_id}")
        
        # Run some tool (example)
        # result = await env.run_tool(rollout_id, "some_tool", arg1="value1")
        
        # Cleanup
        await env.cleanup_rollout(rollout_id, keep_workspace=True)
        print(f"Cleaned up rollout: {rollout_id}")
        
    finally:
        await env.shutdown()


if __name__ == "__main__":
    asyncio.run(main())