"""
Unit tests for ServerPool class.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from aiohttp import ClientSession
from fastmcp import Client as FastMCPClient

from benchmax.envs.mcp.server_pool import ServerPool, ServerInfo


# ===== Fixtures =====


@pytest.fixture
def mock_http_session() -> MagicMock:
    """Create a mock aiohttp ClientSession."""
    session = MagicMock(spec=ClientSession)
    return session


@pytest.fixture
def server_pool(mock_http_session: MagicMock) -> ServerPool:
    """Create a ServerPool instance with fast timeouts for testing."""
    return ServerPool(
        http_session=mock_http_session,
        api_secret="test-token",
        health_check_timeout=1,
        initial_health_check_interval=0.1,
        max_health_check_interval=2,
        max_health_check_attempts=5,
        backoff_factor=2,
    )


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """Create a mock FastMCPClient."""
    client = MagicMock(spec=FastMCPClient)
    client.is_connected = MagicMock(return_value=True)
    client._connect = AsyncMock()
    client._disconnect = AsyncMock()
    return client


def create_mock_response(status: int = 200) -> MagicMock:
    """Helper to create a mock HTTP response."""
    response = MagicMock()
    response.status = status
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


# ===== Test Internal Methods with HTTP Mocking =====


class TestInternalHealthCheck:
    """Test _check_server_health with HTTP mocking."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test health check returns True for 200 response."""
        mock_http_session.get.return_value = create_mock_response(200)

        result = await server_pool._check_server_health("localhost:8000")

        assert result is True
        mock_http_session.get.assert_called_once()
        call_args = mock_http_session.get.call_args
        assert call_args[0][0] == "http://localhost:8000/health"

    @pytest.mark.asyncio
    async def test_health_check_non_200(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test health check returns False for non-200 response."""
        mock_http_session.get.return_value = create_mock_response(500)

        result = await server_pool._check_server_health("localhost:8000")

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test health check returns False on timeout."""
        mock_http_session.get.side_effect = asyncio.TimeoutError()

        result = await server_pool._check_server_health("localhost:8000")

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test health check returns False on connection error."""
        mock_http_session.get.side_effect = Exception("Connection refused")

        result = await server_pool._check_server_health("localhost:8000")

        assert result is False


class TestInternalWaitTillOnline:
    """Test _wait_till_server_online with HTTP mocking."""

    @pytest.mark.asyncio
    async def test_wait_success_first_attempt(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test wait succeeds when server is immediately healthy."""
        mock_http_session.get.return_value = create_mock_response(200)

        await server_pool._wait_till_server_online("localhost:8000")

        assert mock_http_session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_wait_success_after_retries(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test wait succeeds after several failed attempts."""
        responses = [
            create_mock_response(500),
            create_mock_response(500),
            create_mock_response(200),
        ]
        mock_http_session.get.side_effect = responses

        await server_pool._wait_till_server_online("localhost:8000")

        assert mock_http_session.get.call_count == 3

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    async def test_wait_timeout_after_max_attempts(
        self,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test wait raises TimeoutError after max attempts."""
        mock_http_session.get.return_value = create_mock_response(500)

        with pytest.raises(TimeoutError, match="failed to become healthy"):
            await server_pool._wait_till_server_online("localhost:8000")

        assert (
            mock_http_session.get.call_count == server_pool._max_health_check_attempts
        )

    @pytest.mark.asyncio
    async def test_wait_cancelled_on_shutdown(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test wait raises CancelledError when shutdown is initiated."""
        mock_http_session.get.return_value = create_mock_response(500)
        server_pool._shutdown_event.set()

        with pytest.raises(asyncio.CancelledError, match="Shutdown initiated"):
            await server_pool._wait_till_server_online("localhost:8000")


class TestInternalResetServer:
    """Test _reset_server with HTTP mocking."""

    @pytest.mark.asyncio
    async def test_reset_success(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test successful server reset."""
        mock_http_session.post.return_value = create_mock_response(200)

        await server_pool._reset_server("localhost:8000")

        mock_http_session.post.assert_called_once()
        call_args = mock_http_session.post.call_args
        assert call_args[0][0] == "http://localhost:8000/reset"

    @pytest.mark.asyncio
    async def test_reset_non_200_handled(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test reset handles non-200 response gracefully."""
        mock_http_session.post.return_value = create_mock_response(500)

        # Should not raise
        await server_pool._reset_server("localhost:8000")

        mock_http_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_timeout_handled(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test reset handles timeout gracefully."""
        mock_http_session.post.side_effect = asyncio.TimeoutError()

        # Should not raise
        await server_pool._reset_server("localhost:8000")

        mock_http_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_connection_error_handled(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test reset handles connection error gracefully."""
        mock_http_session.post.side_effect = Exception("Connection refused")

        # Should not raise
        await server_pool._reset_server("localhost:8000")

        mock_http_session.post.assert_called_once()


class TestInternalRecoverServer:
    """Test _recover_server with HTTP and MCP mocking."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_recover_success_flow(
        self,
        mock_client_class: Mock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test full recovery flow: wait online → connect MCP → add to pool."""
        mock_http_session.get.return_value = create_mock_response(200)

        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client_class.return_value = mock_client

        server = ServerInfo(
            address="localhost:8000", mcp_client=None, status="recovering"
        )

        await server_pool._recover_server(server)

        # Verify server was added to pool
        assert len(server_pool._unassigned_servers) == 1
        assert server_pool._unassigned_servers[0] == server
        assert server.status == "available"
        assert server.mcp_client is not None
        mock_client._connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    async def test_recover_failure_marks_failed(
        self,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test recovery failure marks server as failed."""
        mock_http_session.get.return_value = create_mock_response(500)

        server = ServerInfo(
            address="localhost:8000", mcp_client=None, status="recovering"
        )

        await server_pool._recover_server(server)

        # Server should be marked as failed, not added to pool
        assert server.status == "failed"
        assert len(server_pool._unassigned_servers) == 0

    @pytest.mark.asyncio
    async def test_recover_cancelled(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test recovery handles cancellation gracefully."""

        # Create a mock response that takes time to return
        async def delayed_health_check() -> MagicMock:
            await asyncio.sleep(1)
            return create_mock_response(200)

        # Mock the context manager properly
        mock_response = MagicMock()
        mock_response.__aenter__ = delayed_health_check
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_http_session.get.return_value = mock_response

        server = ServerInfo(
            address="localhost:8000", mcp_client=None, status="recovering"
        )

        task = asyncio.create_task(server_pool._recover_server(server))
        await asyncio.sleep(0.05)
        task.cancel()

        # Wait for the task to complete (should handle cancellation gracefully)
        await task

        # Server should not be added to pool since recovery was cancelled
        assert len(server_pool._unassigned_servers) == 0
        # Server status might still be recovering or failed
        assert server.status in ["recovering", "failed"]


# ===== Test Public Methods =====


class TestServerAddition:
    """Test server addition methods."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_add_server_once_ready_triggers_recovery(
        self,
        mock_client_class: Mock,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test add_server_once_ready triggers recovery task."""
        mock_http_session.get.return_value = create_mock_response(200)

        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client_class.return_value = mock_client

        await server_pool.add_server_once_ready("localhost:8000")

        # Give recovery task time to complete
        await asyncio.wait_for(
            asyncio.gather(*list(server_pool._recovery_tasks), return_exceptions=True),
            timeout=2,
        )

        # Server should be in pool
        assert len(server_pool._unassigned_servers) == 1
        assert server_pool._unassigned_servers[0].address == "localhost:8000"
        assert server_pool._unassigned_servers[0].status == "available"

    @pytest.mark.asyncio
    async def test_add_server_to_available_pool_direct(
        self, server_pool: ServerPool
    ) -> None:
        """Test add_server_to_available_pool directly adds server bypassing recovery."""
        server = ServerInfo(address="localhost:8000", mcp_client=None, status="pending")

        await server_pool.add_server_to_available_pool(server)

        assert len(server_pool._unassigned_servers) == 1
        assert server_pool._unassigned_servers[0] == server
        assert server.status == "available"

    @pytest.mark.asyncio
    async def test_add_multiple_servers(self, server_pool: ServerPool) -> None:
        """Test adding multiple servers to pool."""
        server1 = ServerInfo(
            address="localhost:8000", mcp_client=None, status="pending"
        )
        server2 = ServerInfo(
            address="localhost:8001", mcp_client=None, status="pending"
        )

        await server_pool.add_server_to_available_pool(server1)
        await server_pool.add_server_to_available_pool(server2)

        assert len(server_pool._unassigned_servers) == 2


class TestServerAcquisitionAndTracking:
    """Test server acquisition and tracking methods."""

    @pytest.mark.asyncio
    async def test_acquire_server_immediate(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test acquire_server returns available server immediately."""
        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)

        acquired = await server_pool.acquire_server("rollout-1")

        assert acquired == server
        assert acquired.status == "assigned"
        assert len(server_pool._unassigned_servers) == 0
        assert len(server_pool._rollout_to_server) == 1

    @pytest.mark.asyncio
    async def test_acquire_server_blocks_when_empty(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test acquire_server blocks when pool is empty, unblocks when server available."""
        acquired = None

        async def acquire_task() -> None:
            nonlocal acquired
            acquired = await server_pool.acquire_server("rollout-1")

        # Start acquisition task (should block)
        task = asyncio.create_task(acquire_task())
        await asyncio.sleep(0.1)

        # Task should still be running (blocked)
        assert not task.done()
        assert acquired is None

        # Add a server
        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)

        # Task should complete
        await asyncio.sleep(0.1)
        assert task.done()
        assert acquired is not None
        assert acquired == server
        assert acquired.status == "assigned"

    @pytest.mark.asyncio
    async def test_get_server_returns_assigned(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test get_server returns correct server for rollout."""
        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)

        await server_pool.acquire_server("rollout-1")
        result = await server_pool.get_server("rollout-1")

        assert result == server

    @pytest.mark.asyncio
    async def test_get_server_returns_none_if_not_assigned(
        self, server_pool: ServerPool
    ) -> None:
        """Test get_server returns None if rollout has no server."""
        result = await server_pool.get_server("rollout-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_rollouts_acquire_different_servers(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test multiple rollouts can acquire different servers."""
        server1 = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        server2 = ServerInfo(
            address="localhost:8001", mcp_client=mock_mcp_client, status="available"
        )

        await server_pool.add_server_to_available_pool(server1)
        await server_pool.add_server_to_available_pool(server2)

        acquired1 = await server_pool.acquire_server("rollout-1")
        acquired2 = await server_pool.acquire_server("rollout-2")

        assert acquired1 != acquired2
        assert len(server_pool._rollout_to_server) == 2
        assert len(server_pool._unassigned_servers) == 0


class TestServerRelease:
    """Test server release functionality."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_release_server_full_flow(
        self,
        mock_client_class: Mock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test release_server disconnects, resets, and recovers server."""
        mock_http_session.post.return_value = create_mock_response(200)
        mock_http_session.get.return_value = create_mock_response(200)

        # Mock MCP client
        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client._disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)
        mock_client_class.return_value = mock_client

        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)
        await server_pool.acquire_server("rollout-1")

        await server_pool.release_server("rollout-1")

        # Verify disconnect was called
        mock_client._disconnect.assert_called_once()

        # Verify reset was attempted
        await asyncio.sleep(0.1)
        mock_http_session.post.assert_called_once()
        call_args = mock_http_session.post.call_args
        assert call_args[0][0] == "http://localhost:8000/reset"

        # Wait for full recovery cycle
        await asyncio.wait_for(
            asyncio.gather(*list(server_pool._recovery_tasks), return_exceptions=True),
            timeout=2,
        )

        # Verify server is back in pool
        assert len(server_pool._unassigned_servers) == 1
        assert server_pool._unassigned_servers[0].address == "localhost:8000"
        assert server_pool._unassigned_servers[0].status == "available"

        # Verify server removed from assignments
        assert len(server_pool._rollout_to_server) == 0

        # Verify MCP client was reconnected
        assert mock_client._connect.call_count >= 1

    @pytest.mark.asyncio
    async def test_release_server_not_connected(
        self,
        server_pool: ServerPool,
        mock_mcp_client: MagicMock,
        mock_http_session: MagicMock,
    ) -> None:
        """Test release_server handles disconnected MCP client."""
        mock_http_session.post.return_value = create_mock_response(200)
        mock_http_session.get.return_value = create_mock_response(200)
        mock_mcp_client.is_connected.return_value = False

        with patch.object(server_pool, "_recover_server", new_callable=AsyncMock):
            server = ServerInfo(
                address="localhost:8000", mcp_client=mock_mcp_client, status="available"
            )
            await server_pool.add_server_to_available_pool(server)
            await server_pool.acquire_server("rollout-1")

            await server_pool.release_server("rollout-1")

            # Disconnect should not be called if not connected
            mock_mcp_client._disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_nonexistent_rollout(self, server_pool: ServerPool) -> None:
        """Test release_server handles non-existent rollout gracefully."""
        # Should not raise
        await server_pool.release_server("nonexistent-rollout")


class TestServerFailure:
    """Test server failure reporting and recovery."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_report_failure_responsive_server(
        self,
        mock_client_class: Mock,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test report_server_failure with responsive server resets it."""
        mock_http_session.get.return_value = create_mock_response(200)
        mock_http_session.post.return_value = create_mock_response(200)

        # Mock MCP client
        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client._disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)
        mock_client_class.return_value = mock_client

        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)
        await server_pool.acquire_server("rollout-1")

        await server_pool.report_server_failure("rollout-1", "Test failure")

        # Should check health
        await asyncio.sleep(0.1)
        assert mock_http_session.get.call_count >= 1

        # Should reset
        mock_http_session.post.assert_called_once()

        # Wait for recovery
        await asyncio.wait_for(
            asyncio.gather(*list(server_pool._recovery_tasks), return_exceptions=True),
            timeout=2,
        )

        # Verify server is back in pool
        assert len(server_pool._unassigned_servers) == 1
        assert server_pool._unassigned_servers[0].status == "available"

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    async def test_report_failure_unresponsive_server(
        self,
        mock_client_class: Mock,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test report_server_failure with unresponsive server skips reset."""
        mock_http_session.get.return_value = create_mock_response(500)

        # Mock MCP client
        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)
        mock_client_class.return_value = mock_client

        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)
        await server_pool.acquire_server("rollout-1")

        await server_pool.report_server_failure("rollout-1", "Test failure")

        # Should check health (initial check in report_server_failure)
        await asyncio.sleep(0.1)
        assert mock_http_session.get.call_count >= 1

        # Should NOT reset (server unresponsive)
        mock_http_session.post.assert_not_called()

        # Server should be in recovery but will eventually fail
        # since health checks keep failing. Recovery will keep trying
        # up to max_health_check_attempts times.
        # Wait for all recovery tasks to complete
        initial_task_count = len(server_pool._recovery_tasks)
        if initial_task_count > 0:
            # Wait for recovery tasks to finish (with timeout)
            await asyncio.wait_for(
                asyncio.gather(
                    *list(server_pool._recovery_tasks), return_exceptions=True
                ),
                timeout=2,
            )

        # Server should have failed recovery and NOT be in pool
        assert len(server_pool._unassigned_servers) == 0
        assert server.status == "failed"

        # Recovery process should have made multiple health check attempts
        # Initial check + 10 attempts in _wait_till_server_online
        assert mock_http_session.get.call_count >= 3

    @pytest.mark.asyncio
    async def test_report_failure_nonexistent_rollout(
        self, server_pool: ServerPool
    ) -> None:
        """Test report_server_failure handles non-existent rollout gracefully."""
        # Should not raise
        await server_pool.report_server_failure("nonexistent-rollout", "Test failure")


class TestServerShutdown:
    """Test server pool shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_recovery_tasks(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test shutdown cancels all recovery tasks."""
        mock_http_session.get.return_value = create_mock_response(500)  # Keep retrying

        # Start a recovery task that will keep running
        server = ServerInfo(
            address="localhost:8000", mcp_client=None, status="recovering"
        )
        task = server_pool._create_recovery_task(server_pool._recover_server(server))

        await asyncio.sleep(0.2)
        assert not task.done()

        await server_pool.shutdown()

        # Task should be cancelled
        assert task.cancelled() or task.done()
        assert len(server_pool._recovery_tasks) == 0

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_all_clients(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test shutdown disconnects all MCP clients."""
        server1 = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        server2 = ServerInfo(
            address="localhost:8001", mcp_client=mock_mcp_client, status="available"
        )

        await server_pool.add_server_to_available_pool(server1)
        await server_pool.add_server_to_available_pool(server2)
        await server_pool.acquire_server("rollout-1")

        await server_pool.shutdown()

        # Both clients should be disconnected (called twice)
        assert mock_mcp_client._disconnect.call_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_clears_all_state(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test shutdown clears all server tracking."""
        server1 = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        server2 = ServerInfo(
            address="localhost:8001", mcp_client=mock_mcp_client, status="available"
        )

        await server_pool.add_server_to_available_pool(server1)
        await server_pool.add_server_to_available_pool(server2)
        await server_pool.acquire_server("rollout-1")

        await server_pool.shutdown()

        assert len(server_pool._unassigned_servers) == 0
        assert len(server_pool._rollout_to_server) == 0

    @pytest.mark.asyncio
    async def test_shutdown_handles_disconnect_timeout(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test shutdown handles MCP client disconnect timeouts gracefully."""

        async def slow_disconnect() -> None:
            await asyncio.sleep(2)  # Longer than timeout

        mock_mcp_client._disconnect = slow_disconnect

        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_mcp_client, status="available"
        )
        await server_pool.add_server_to_available_pool(server)

        # Should complete despite timeout
        await server_pool.shutdown(client_disconnect_timeout=1)

        assert len(server_pool._unassigned_servers) == 0


# ===== Integration and Edge Cases =====


class TestConcurrencyAndEdgeCases:
    """Test concurrent operations and edge cases."""

    @pytest.mark.asyncio
    async def test_multiple_rollouts_waiting_for_servers(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test multiple rollouts can wait in queue for servers."""
        acquired = []

        async def acquire_task(rollout_id: str) -> None:
            server = await server_pool.acquire_server(rollout_id)
            acquired.append((rollout_id, server))

        # Start 3 acquisition tasks
        tasks = [
            asyncio.create_task(acquire_task("rollout-1")),
            asyncio.create_task(acquire_task("rollout-2")),
            asyncio.create_task(acquire_task("rollout-3")),
        ]

        await asyncio.sleep(0.1)
        assert all(not t.done() for t in tasks)

        # Add servers one by one
        for i in range(3):
            server = ServerInfo(
                address=f"localhost:800{i}",
                mcp_client=mock_mcp_client,
                status="available",
            )
            await server_pool.add_server_to_available_pool(server)
            await asyncio.sleep(0.1)

        # All tasks should complete
        await asyncio.gather(*tasks)
        assert len(acquired) == 3
        assert all(s.status == "assigned" for _, s in acquired)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_concurrent_acquire_and_release(
        self,
        mock_client_class: Mock,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test concurrent acquire and release operations."""
        mock_http_session.post.return_value = create_mock_response(200)
        mock_http_session.get.return_value = create_mock_response(200)

        # Mock MCP client
        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client._disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)
        mock_client_class.return_value = mock_client

        # Add initial servers
        for i in range(2):
            server = ServerInfo(
                address=f"localhost:800{i}", mcp_client=mock_client, status="available"
            )
            await server_pool.add_server_to_available_pool(server)

        # Concurrent operations
        async def workflow(rollout_id: str) -> None:
            await server_pool.acquire_server(rollout_id)
            await server_pool.release_server(rollout_id)

        tasks = [asyncio.create_task(workflow(f"rollout-{i}")) for i in range(4)]

        # Wait for all workflows with generous timeout for recovery
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

        # All should complete without errors
        assert all(t.done() and not t.exception() for t in tasks)

        # Eventually all servers should be back in pool
        assert len(server_pool._unassigned_servers) >= 2

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_recovery_task_cleanup(
        self,
        mock_client_class: Mock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test recovery tasks are properly tracked and cleaned up."""
        mock_http_session.get.return_value = create_mock_response(200)

        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client_class.return_value = mock_client

        # Add servers which triggers recovery
        await server_pool.add_server_once_ready("localhost:8000")
        await server_pool.add_server_once_ready("localhost:8001")

        await asyncio.wait_for(
            asyncio.gather(*list(server_pool._recovery_tasks), return_exceptions=True),
            timeout=15.0,
        )

        # Tasks should be cleaned up after completion
        assert len(server_pool._recovery_tasks) == 0
        assert len(server_pool._unassigned_servers) == 2

    @pytest.mark.asyncio
    async def test_shutdown_during_recovery(
        self, server_pool: ServerPool, mock_http_session: MagicMock
    ) -> None:
        """Test shutdown while servers are recovering."""

        # Set up slow health checks
        async def slow_health_check() -> MagicMock:
            await asyncio.sleep(0.5)
            return create_mock_response(200)

        # Mock the context manager properly
        mock_response = MagicMock()
        mock_response.__aenter__ = slow_health_check
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_http_session.get.return_value = mock_response

        # Start recovery
        await server_pool.add_server_once_ready("localhost:8000")
        await asyncio.sleep(0.1)

        # Shutdown should cancel the recovery
        await server_pool.shutdown()

        # Pool should be empty
        assert len(server_pool._unassigned_servers) == 0
        assert len(server_pool._recovery_tasks) == 0

    @pytest.mark.asyncio
    async def test_acquire_blocks_multiple_waiters(
        self, server_pool: ServerPool, mock_mcp_client: MagicMock
    ) -> None:
        """Test multiple acquire calls block and resolve in order."""
        results = []

        async def acquire_and_record(rollout_id: str) -> None:
            server = await server_pool.acquire_server(rollout_id)
            results.append((rollout_id, server.address))

        # Start 3 waiters
        tasks = [
            asyncio.create_task(acquire_and_record("rollout-1")),
            asyncio.create_task(acquire_and_record("rollout-2")),
            asyncio.create_task(acquire_and_record("rollout-3")),
        ]

        await asyncio.sleep(0.1)
        assert len(results) == 0  # All should be waiting

        # Add servers
        for i in range(3):
            server = ServerInfo(
                address=f"localhost:800{i}",
                mcp_client=mock_mcp_client,
                status="available",
            )
            await server_pool.add_server_to_available_pool(server)
            await asyncio.sleep(0.05)  # Let one acquire complete

        await asyncio.gather(*tasks)
        assert len(results) == 3
        assert all(
            rid in [r[0] for r in results]
            for rid in ["rollout-1", "rollout-2", "rollout-3"]
        )

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.server_pool.asyncio.sleep")
    @patch("benchmax.envs.mcp.server_pool.FastMCPClient")
    async def test_server_status_transitions(
        self,
        mock_client_class: Mock,
        mock_sleep: AsyncMock,
        server_pool: ServerPool,
        mock_http_session: MagicMock,
    ) -> None:
        """Test server status transitions through lifecycle."""
        mock_http_session.post.return_value = create_mock_response(200)
        mock_http_session.get.return_value = create_mock_response(200)

        # Mock MCP client
        mock_client = MagicMock()
        mock_client._connect = AsyncMock()
        mock_client._disconnect = AsyncMock()
        mock_client.is_connected = MagicMock(return_value=True)
        mock_client_class.return_value = mock_client

        # Start as pending/available
        server = ServerInfo(
            address="localhost:8000", mcp_client=mock_client, status="pending"
        )
        await server_pool.add_server_to_available_pool(server)
        assert server.status == "available"

        # Acquire -> assigned
        await server_pool.acquire_server("rollout-1")
        assert server.status == "assigned"

        # Release -> available
        release_task = asyncio.create_task(server_pool.release_server("rollout-1"))
        await release_task

        # Should be back to available
        assert server.status == "available"
        assert len(server_pool._unassigned_servers) == 1
