import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from benchmax.envs.mcp.provisioners import LocalProvisioner


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestLocalProvisionerInit:
    """Tests for LocalProvisioner initialization."""

    def test_init_requires_workdir_path(self, example_workdir: Path) -> None:
        """Provisioner initializes with given workdir and defaults."""
        LocalProvisioner(workdir_path=example_workdir)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


class TestLocalProvisionerProvision:
    """Tests for LocalProvisioner.provision_servers."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.local_provisioner.setup_sync_dir")
    @patch.object(LocalProvisioner, "_spawn_process", new_callable=AsyncMock)
    async def test_provision_runs_expected_commands(
        self,
        mock_spawn: AsyncMock,
        mock_setup: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Provisioner should prepare sync dir and start subprocesses."""
        mock_setup.return_value = test_sync_dir
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_spawn.return_value = mock_proc

        provisioner = LocalProvisioner(workdir_path=example_workdir, num_servers=2)
        addresses = await provisioner.provision_servers("dummy-api-secret")

        assert len(addresses) == 2
        mock_setup.assert_called_once_with(example_workdir)
        # First call is setup_cmd (wait=True), next calls are servers
        assert mock_spawn.call_count == 3
        assert addresses == ["localhost:8080", "localhost:8081"]

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.local_provisioner.setup_sync_dir")
    async def test_provision_handles_setup_failure(
        self, mock_setup: Mock, example_workdir: Path
    ) -> None:
        """setup_sync_dir errors are propagated."""
        mock_setup.side_effect = OSError("setup failed")
        provisioner = LocalProvisioner(workdir_path=example_workdir)

        with pytest.raises(OSError):
            await provisioner.provision_servers("dummy-api-secret")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestLocalProvisionerTeardown:
    """Tests for LocalProvisioner.teardown."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.local_provisioner.cleanup_dir")
    async def test_teardown_kills_processes_and_cleans_up(
        self, mock_cleanup: Mock, example_workdir: Path, test_sync_dir: Path
    ) -> None:
        """Active processes are killed and sync dir cleaned up."""
        provisioner = LocalProvisioner(workdir_path=example_workdir)

        proc = MagicMock()
        proc.poll.return_value = None
        proc.kill = MagicMock()
        proc.wait = MagicMock()
        provisioner._processes = [proc]
        provisioner._sync_dir = test_sync_dir
        provisioner._is_provisioned = True

        await provisioner.teardown()
        proc.kill.assert_called_once()
        proc.wait.assert_called_once()
        mock_cleanup.assert_called_once_with(test_sync_dir)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.local_provisioner.cleanup_dir")
    async def test_teardown_skips_already_terminated(
        self, mock_cleanup: Mock, example_workdir: Path, test_sync_dir: Path
    ) -> None:
        """Teardown skips processes that are already terminated."""
        provisioner = LocalProvisioner(workdir_path=example_workdir)
        proc = MagicMock()
        proc.poll.return_value = 0  # Already exited
        proc.kill = MagicMock()
        provisioner._processes = [proc]
        provisioner._sync_dir = test_sync_dir
        provisioner._is_provisioned = True

        await provisioner.teardown()
        proc.kill.assert_not_called()
        mock_cleanup.assert_called_once_with(test_sync_dir)
