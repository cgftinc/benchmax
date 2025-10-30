"""
Unit tests for ParallelMcpEnv — fully isolated and async-compatible.
Covers initialization, tool discovery, rollout lifecycle, reward computation,
workspace file management, and edge cases.
"""

import aiohttp
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, AsyncGenerator, Dict

from benchmax.envs.mcp.parallel_mcp_env import ParallelMcpEnv
from mcp.types import TextContent
from fastmcp.exceptions import ToolError

from benchmax.envs.mcp.server_pool import ServerPool


# ------------------------
# Mocked dependencies
# ------------------------


@pytest.fixture
def mock_provisioner() -> MagicMock:
    """Return a mock provisioner with async methods."""
    provisioner = MagicMock()
    provisioner.provision_servers = AsyncMock(
        return_value=["127.0.0.1:8000", "127.0.0.1:8001"]
    )
    provisioner.teardown = AsyncMock()
    return provisioner


class MockServerPool(ServerPool):
    def __init__(self):
        self._servers: Dict[str, Any] = {}  # rollout_id -> server_info

    async def add_server_once_ready(self, address: str) -> None:
        pass

    async def shutdown(self, client_disconnect_timeout: int | float = 20) -> None:
        pass

    async def acquire_server(self, rollout_id: str, max_attempts: int = 5) -> Any:
        if rollout_id not in self._servers:
            self._servers[rollout_id] = MockServerInfo()
        return self._servers[rollout_id]

    async def get_server(self, rollout_id: str) -> Any:
        return self._servers.get(rollout_id, None)

    async def release_server(self, rollout_id: str) -> None:
        pass

    async def add_server_to_available_pool(self, server: Any) -> None:
        pass

    async def report_server_failure(self, rollout_id: str, reason: str) -> None:
        pass


class MockMcpClient:
    def is_connected(self) -> bool:
        return True

    async def list_tools(self) -> list:
        return []

    async def call_tool(self, tool_name: str, tool_args: dict, timeout=None) -> Any:
        return type(
            "ToolResponse", (), {"content": [TextContent(type="text", text="mock")]}
        )()


class MockServerInfo:
    def __init__(self) -> None:
        self.address = "127.0.0.1"
        self.mcp_client = MockMcpClient()


@pytest.fixture
def mock_server_pool() -> MockServerPool:
    """Return a mock server pool that tracks rollouts properly."""
    return MockServerPool()


# ------------------------
# ParallelMcpEnv fixture
# ------------------------


@pytest.fixture
async def parallel_mcp_env(
    example_workdir: Path, mock_provisioner: MagicMock, mock_server_pool: MockServerPool
) -> AsyncGenerator[ParallelMcpEnv, None]:
    """Return a ParallelMcpEnv instance with mocked provisioner and server pool."""
    env = ParallelMcpEnv(
        workdir_path=example_workdir,
        provisioner=mock_provisioner,
        provision_at_init=False,
    )
    # Inject mocked server pool
    env._server_pool = mock_server_pool
    env._servers_provisioned = True
    env._http_session = aiohttp.ClientSession()
    yield env
    await env.shutdown()


# ------------------------
# Test Initialization
# ------------------------


class TestParallelMcpEnvInit:
    def test_init_sets_attributes(self, parallel_mcp_env: ParallelMcpEnv) -> None:
        env = parallel_mcp_env
        assert env._workdir_path.exists() or isinstance(env._workdir_path, Path)
        assert env._provisioner is not None
        assert env._server_pool is not None
        assert env._api_secret is not None


# ------------------------
# Test Tool Discovery
# ------------------------


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_tools_returns_mocked_tools(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        tools = await parallel_mcp_env.list_tools()
        assert isinstance(tools, list)


# ------------------------
# Test Rollout Lifecycle
# ------------------------


class TestRolloutLifecycle:
    @pytest.mark.asyncio
    async def test_init_rollout_acquires_server(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_1"
        await env.init_rollout(rollout_id)
        assert env._server_pool
        server_info = await env._server_pool.get_server(rollout_id)
        assert server_info is not None
        assert server_info.address == "127.0.0.1"

    @pytest.mark.asyncio
    async def test_run_tool_returns_mock_content(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_2"
        await env.init_rollout(rollout_id)
        result = await env.run_tool(rollout_id, "dummy_tool", arg1=123)
        assert isinstance(result, str)
        assert "mock" in result

    @pytest.mark.asyncio
    async def test_run_tool_raises_tool_error(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        """If MCP client raises ToolError, run_tool should return the error string."""
        rollout_id = "rollout_3"
        await parallel_mcp_env.init_rollout(rollout_id)
        assert parallel_mcp_env._server_pool
        server_info = await parallel_mcp_env._server_pool.get_server(rollout_id)

        # Patch call_tool to raise ToolError
        assert server_info and server_info.mcp_client
        server_info.mcp_client.call_tool = AsyncMock(side_effect=ToolError("failure"))

        result = await parallel_mcp_env.run_tool(rollout_id, "dummy_tool")
        assert "failure" in result


# ------------------------
# Test Reward Computation
# ------------------------


class TestComputeReward:
    @pytest.mark.asyncio
    async def test_compute_reward_returns_json(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_4"
        await env.init_rollout(rollout_id)

        mock_response = AsyncMock()
        mock_response.__aenter__.return_value.status = 200
        mock_response.__aenter__.return_value.json = AsyncMock(
            return_value={"reward": 1.0}
        )

        with patch(
            "benchmax.envs.mcp.parallel_mcp_env.get_auth_headers",
            return_value={"Authorization": "Bearer mock"},
        ):
            with patch(
                "benchmax.envs.mcp.parallel_mcp_env.aiohttp.ClientSession.post",
                return_value=mock_response,
            ):
                result = await env.compute_reward(
                    rollout_id, completion="output", ground_truth="truth"
                )
                assert isinstance(result, dict)
                assert result["reward"] == 1.0

    @pytest.mark.asyncio
    async def test_compute_reward_http_failure_raises(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_5"
        await env.init_rollout(rollout_id)

        mock_response = AsyncMock()
        mock_response.__aenter__.return_value.status = 500
        mock_response.__aenter__.return_value.json = AsyncMock(
            return_value={
                "error": "internal_error",
                "detail": "Failed to compute reward.",
            }
        )

        with patch(
            "benchmax.envs.mcp.parallel_mcp_env.get_auth_headers",
            return_value={"Authorization": "Bearer mock"},
        ):
            with patch(
                "benchmax.envs.mcp.parallel_mcp_env.aiohttp.ClientSession.post",
                return_value=mock_response,
            ):
                with pytest.raises(RuntimeError, match="Reward computation failed"):
                    await env.compute_reward(
                        rollout_id, completion="output", ground_truth="truth"
                    )


# ------------------------
# Test Upload / Download
# ------------------------


class TestWorkspaceFiles:
    @pytest.mark.asyncio
    async def test_copy_to_workspace(
        self, parallel_mcp_env: ParallelMcpEnv, tmp_path: Path
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_6"
        await env.init_rollout(rollout_id)

        file_path = tmp_path / "test.txt"
        file_path.write_text("hello")

        with patch(
            "benchmax.envs.mcp.parallel_mcp_env.upload_form", new_callable=AsyncMock
        ) as mock_upload:
            await env.copy_to_workspace(rollout_id, file_path)
            mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_copy_from_workspace(
        self, parallel_mcp_env: ParallelMcpEnv, tmp_path: Path
    ) -> None:
        env = parallel_mcp_env
        rollout_id = "rollout_7"
        await env.init_rollout(rollout_id)

        dst_path = tmp_path / "out.txt"

        with patch(
            "benchmax.envs.mcp.parallel_mcp_env.download_file", new_callable=AsyncMock
        ) as mock_download:
            await env.copy_from_workspace(rollout_id, "remote.txt", dst_path)
            mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_copy_content_to_workspace(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        rollout_id = "rollout_8"
        await parallel_mcp_env.init_rollout(rollout_id)

        with patch(
            "benchmax.envs.mcp.parallel_mcp_env.upload_form", new_callable=AsyncMock
        ) as mock_upload:
            # string content
            await parallel_mcp_env.copy_content_to_workspace(
                rollout_id, "hello", "file.txt"
            )
            _, named_args = mock_upload.call_args

            file_bytes_arg = named_args["file_bytes"]
            dst_filename_arg = named_args["filename"]
            assert file_bytes_arg == b"hello"
            assert dst_filename_arg == "file.txt"

            # bytes content
            mock_upload.reset_mock()
            await parallel_mcp_env.copy_content_to_workspace(
                rollout_id, b"\x01\x02", "file.bin"
            )
            _, named_args = mock_upload.call_args

            file_bytes_arg = named_args["file_bytes"]
            dst_filename_arg = named_args["filename"]
            assert file_bytes_arg == b"\x01\x02"
            assert dst_filename_arg == "file.bin"


# ------------------------
# Test Edge Cases
# ------------------------


class TestRunToolEdgeCases:
    @pytest.mark.asyncio
    async def test_run_tool_before_init_raises(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        rollout_id = "not_initialized"
        with pytest.raises(
            RuntimeError, match=f"Rollout '{rollout_id}' not initialized"
        ):
            await parallel_mcp_env.run_tool(rollout_id, "dummy_tool")

    @pytest.mark.asyncio
    async def test_run_tool_applies_output_parser(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        rollout_id = "rollout_parser"
        await parallel_mcp_env.init_rollout(rollout_id)

        parallel_mcp_env._output_parsers = {"dummy_tool": lambda x: x.upper()}
        result = await parallel_mcp_env.run_tool(rollout_id, "dummy_tool")
        assert result == "MOCK"

    @pytest.mark.asyncio
    async def test_allowed_tools_filter(self, parallel_mcp_env: ParallelMcpEnv) -> None:
        rollout_id = "rollout_allowed"
        await parallel_mcp_env.init_rollout(rollout_id)

        # patch MCP client's list_tools to return some tool names
        tools_data = [MagicMock(name="ToolA"), MagicMock(name="ToolB")]
        for t in tools_data:
            t.name = t._mock_name

        assert parallel_mcp_env._server_pool
        server_info = await parallel_mcp_env._server_pool.get_server(rollout_id)
        assert server_info and server_info.mcp_client
        server_info.mcp_client.list_tools = AsyncMock(return_value=tools_data)

        parallel_mcp_env._allowed_tools = ["ToolA"]
        tool_defs = await parallel_mcp_env.list_tools()
        assert all(t.name in ["ToolA"] for t in tool_defs)


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_resources(
        self, parallel_mcp_env: ParallelMcpEnv
    ) -> None:
        env = parallel_mcp_env

        assert env._server_pool
        assert env._http_session

        # Ensure async methods are AsyncMock
        env._server_pool.shutdown = AsyncMock()
        env._http_session.close = AsyncMock()
        env._provisioner.teardown = AsyncMock()

        await env.shutdown()

        env._server_pool.shutdown.assert_awaited_once()
        env._http_session.close.assert_awaited_once()
        env._provisioner.teardown.assert_awaited_once()
