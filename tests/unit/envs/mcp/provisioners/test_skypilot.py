import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sky
from benchmax.envs.mcp.provisioners import SkypilotProvisioner


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestSkypilotProvisionerInit:
    """Behavioral tests for SkypilotProvisioner initialization."""

    def test_init_accepts_valid_parameters(self, example_workdir: Path) -> None:
        """Provisioner initializes with valid parameters without errors."""
        num_nodes = 2
        servers_per_node = 4

        provisioner = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.AWS(),
            num_nodes=num_nodes,
            servers_per_node=servers_per_node,
            cpus="4+",
            memory="32+",
            base_cluster_name="unit-test",
        )

        assert isinstance(provisioner.cluster_name, str)
        assert "unit-test" in provisioner.cluster_name

        assert provisioner.num_servers == num_nodes * servers_per_node

    def test_init_invalid_num_nodes_raises(self, example_workdir: Path) -> None:
        """Invalid node counts raise ValueError."""
        with pytest.raises(ValueError, match="num_nodes"):
            SkypilotProvisioner(
                workdir_path=example_workdir, cloud=sky.AWS(), num_nodes=0
            )

    def test_init_invalid_servers_per_node_raises(self, example_workdir: Path) -> None:
        """Invalid servers_per_node raises ValueError."""
        with pytest.raises(ValueError, match="servers_per_node"):
            SkypilotProvisioner(
                workdir_path=example_workdir, cloud=sky.GCP(), servers_per_node=0
            )


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


class TestSkypilotProvisionerProvision:
    """Tests for SkypilotProvisioner.provision_servers."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.launch")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.setup_sync_dir")
    async def test_provision_returns_expected_addresses(
        self,
        mock_setup: Mock,
        mock_launch: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Provisioning produces correct addresses from returned node IPs."""
        mock_setup.return_value = test_sync_dir

        mock_handle = MagicMock()
        mock_handle.stable_internal_external_ips = [
            ("10.0.0.1", "1.2.3.4"),
            ("10.0.0.2", "5.6.7.8"),
        ]
        mock_launch.return_value = (None, mock_handle)

        provisioner = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.AWS(),
            num_nodes=2,
            servers_per_node=2,
        )
        result = await provisioner.provision_servers("dummy-api-secret")

        assert sorted(result) == sorted(
            [
                "1.2.3.4:8080",
                "1.2.3.4:8081",
                "5.6.7.8:8080",
                "5.6.7.8:8081",
            ]
        )
        mock_setup.assert_called_once_with(example_workdir)
        mock_launch.assert_called_once()

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.launch")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.setup_sync_dir")
    async def test_provision_configures_task_and_envs(
        self,
        mock_setup: Mock,
        mock_launch: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Launch task is created with expected parameters."""
        mock_setup.return_value = test_sync_dir

        mock_handle = MagicMock()
        mock_handle.stable_internal_external_ips = [("10.0.0.1", "1.2.3.4")]
        mock_launch.return_value = (None, mock_handle)

        provisioner = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.Azure(),
            cpus="8",
            memory="64+",
        )
        await provisioner.provision_servers(api_secret="abc123")

        args, kwargs = mock_launch.call_args
        task = kwargs["task"]
        assert task.envs["API_SECRET"] == "abc123"
        assert kwargs["detach_run"] is True
        assert kwargs["retry_until_up"] is True

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.cleanup_dir")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.launch")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.setup_sync_dir")
    async def test_provision_handles_launch_failure(
        self,
        mock_setup: Mock,
        mock_launch: Mock,
        mock_cleanup: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Failure during sky.launch triggers cleanup and raises RuntimeError."""
        mock_setup.return_value = test_sync_dir
        mock_launch.side_effect = Exception("launch failed")

        provisioner = SkypilotProvisioner(workdir_path=example_workdir, cloud=sky.GCP())

        with pytest.raises(RuntimeError, match="SkyPilot cluster launch failed"):
            await provisioner.provision_servers("dummy-api-secret")

        mock_cleanup.assert_called_once_with(test_sync_dir)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.cleanup_dir")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.launch")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.setup_sync_dir")
    async def test_provision_handles_no_launch_handle_failure(
        self,
        mock_setup: Mock,
        mock_launch: Mock,
        mock_cleanup: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Failure during sky.launch triggers cleanup and raises RuntimeError."""
        mock_setup.return_value = test_sync_dir
        mock_launch.return_value = (None, None)

        provisioner = SkypilotProvisioner(workdir_path=example_workdir, cloud=sky.GCP())

        with pytest.raises(RuntimeError, match="SkyPilot launch returned no handle"):
            await provisioner.provision_servers("dummy-api-secret")

        mock_cleanup.assert_called_once_with(test_sync_dir)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.launch")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.setup_sync_dir")
    async def test_provision_twice_raises_error(
        self,
        mock_setup: Mock,
        mock_launch: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Re-provisioning an existing cluster raises RuntimeError."""
        mock_setup.return_value = test_sync_dir

        mock_handle = MagicMock()
        mock_handle.stable_internal_external_ips = [
            ("10.0.0.1", "1.2.3.4"),
            ("10.0.0.2", "5.6.7.8"),
        ]
        mock_launch.return_value = (None, mock_handle)

        provisioner = SkypilotProvisioner(workdir_path=example_workdir, cloud=sky.AWS())
        await provisioner.provision_servers("dummy-api-secret")

        with pytest.raises(RuntimeError, match="already provisioned"):
            await provisioner.provision_servers("dummy-api-secret")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


class TestSkypilotProvisionerTeardown:
    """Tests for SkypilotProvisioner.teardown."""

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.down")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.cleanup_dir")
    async def test_teardown_invokes_down_and_cleanup(
        self,
        mock_cleanup: Mock,
        mock_down: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Normal teardown calls sky.down and cleans up sync dir."""
        provisioner = SkypilotProvisioner(workdir_path=example_workdir, cloud=sky.AWS())
        provisioner._cluster_provisioned = True
        provisioner._sync_workdir = test_sync_dir

        await provisioner.teardown()

        mock_down.assert_called_once()
        mock_cleanup.assert_called_once_with(test_sync_dir)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.down")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.cleanup_dir")
    async def test_teardown_handles_down_failure(
        self,
        mock_cleanup: Mock,
        mock_down: Mock,
        example_workdir: Path,
        test_sync_dir: Path,
    ) -> None:
        """Even if sky.down fails, cleanup should still run."""
        mock_down.side_effect = Exception("Down failed")
        provisioner = SkypilotProvisioner(
            workdir_path=example_workdir, cloud=sky.Azure()
        )
        provisioner._cluster_provisioned = True
        provisioner._sync_workdir = test_sync_dir

        await provisioner.teardown()
        mock_cleanup.assert_called_once_with(test_sync_dir)

    @pytest.mark.asyncio
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.sky.down")
    @patch("benchmax.envs.mcp.provisioners.skypilot_provisioner.cleanup_dir")
    async def test_teardown_no_sync_dir_still_downs(
        self, mock_cleanup: Mock, mock_down: Mock, example_workdir: Path
    ) -> None:
        """Teardown runs even when no sync directory is set."""
        provisioner = SkypilotProvisioner(workdir_path=example_workdir, cloud=sky.GCP())
        provisioner._cluster_provisioned = True
        provisioner._sync_workdir = None

        await provisioner.teardown()

        mock_down.assert_called_once()
        mock_cleanup.assert_not_called()
