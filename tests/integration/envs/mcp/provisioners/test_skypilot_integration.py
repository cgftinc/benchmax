"""
Integration tests for SkypilotProvisioner.
"""

import pytest
import asyncio
import sky
from pathlib import Path
from benchmax.envs.mcp.provisioners.skypilot_provisioner import SkypilotProvisioner
from tests.integration.envs.mcp.provisioners.utils import wait_for_server_health


class TestEndToEnd:
    """End-to-end integration tests for provisioning and teardown."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.remote
    async def test_single_node_lifecycle(self, example_workdir: Path):
        """Test complete lifecycle of a single-node cluster with all validations."""
        base_name = "test-single-node-cluster"
        api_secret = "single-node-test-secret-32chars!!"

        provisioner = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.Azure(),
            num_nodes=1,
            servers_per_node=2,
            cpus="4+",
            memory="16+",
            base_cluster_name=base_name,
        )

        try:
            # Verify configuration before provisioning
            assert provisioner.cluster_name.startswith(f"{base_name}-")
            assert provisioner._workdir_path.is_absolute()

            # Provision
            addresses = await provisioner.provision_servers(api_secret)
            assert len(addresses) == 2  # 1 node * 2 servers

            # Verify all addresses have correct format
            for addr in addresses:
                assert ":" in addr
                host, port = addr.split(":")
                assert port.isdigit()
                assert 8080 <= int(port) < 8090

            # Verify servers are up and healthy
            health_checks = await asyncio.gather(
                *[wait_for_server_health(addr, timeout=90.0) for addr in addresses]
            )
            assert all(health_checks), "Not all servers became healthy"

            # Check that double-provisioning would result in an error
            with pytest.raises(RuntimeError, match="already provisioned"):
                await provisioner.provision_servers(api_secret)
        finally:
            # Teardown - always attempt cleanup
            await provisioner.teardown()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.remote
    async def test_multi_node_lifecycle(self, example_workdir: Path):
        """Test complete lifecycle of a multi-node cluster with all validations."""
        provisioner1 = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.Azure(),
            num_nodes=2,
            servers_per_node=3,
            cpus=2,
            memory=8,
        )

        provisioner2 = SkypilotProvisioner(
            workdir_path=example_workdir,
            cloud=sky.Azure(),
            num_nodes=1,
            servers_per_node=1,
        )

        api_secret = "multi-node-test-secret-32chars!!"

        try:
            # Verify cluster names are unique
            assert provisioner1.cluster_name != provisioner2.cluster_name
            assert provisioner1.cluster_name.startswith("benchmax-env-cluster-")
            assert provisioner2.cluster_name.startswith("benchmax-env-cluster-")

            # Provision main test cluster
            addresses = await provisioner1.provision_servers(api_secret)
            assert len(addresses) == 6  # 2 nodes * 3 servers

            # Verify addresses are grouped by node
            hosts = [addr.split(":")[0] for addr in addresses]
            unique_hosts = set(hosts)
            assert len(unique_hosts) == 2, "Should have 2 unique node IPs"

            for host in unique_hosts:
                host_servers = [addr for addr in addresses if addr.startswith(host)]
                assert len(host_servers) == 3, f"Host {host} should have 3 servers"

            # Verify all servers are up and healthy
            health_checks = await asyncio.gather(
                *[wait_for_server_health(addr, timeout=90.0) for addr in addresses]
            )
            assert all(health_checks), "Not all servers became healthy"
        finally:
            # Teardown - always attempt cleanup
            await provisioner1.teardown()

class TestValidation:
    """Test parameter validation without provisioning."""
    
    def test_invalid_num_nodes(self):
        """Test validation of num_nodes parameter."""
        with pytest.raises(ValueError, match="at least 1"):
            SkypilotProvisioner(
                workdir_path=".",
                cloud=sky.Azure(),
                num_nodes=0,
                servers_per_node=5,
            )
    
    def test_invalid_servers_per_node_low(self):
        """Test validation of servers_per_node parameter (too low)."""
        with pytest.raises(ValueError, match="between 1 and 100"):
            SkypilotProvisioner(
                workdir_path=".",
                cloud=sky.Azure(),
                num_nodes=1,
                servers_per_node=0,
            )
    
    def test_invalid_servers_per_node_high(self):
        """Test validation of servers_per_node parameter (too high)."""
        with pytest.raises(ValueError, match="between 1 and 100"):
            SkypilotProvisioner(
                workdir_path=".",
                cloud=sky.Azure(),
                num_nodes=1,
                servers_per_node=101,
            )
    
    def test_custom_base_cluster_name(self):
        """Test that custom base cluster name is used."""
        provisioner = SkypilotProvisioner(
            workdir_path=".",
            cloud=sky.Azure(),
            num_nodes=1,
            servers_per_node=1,
            base_cluster_name="my-custom-cluster",
        )
        
        assert provisioner.cluster_name.startswith("my-custom-cluster-")
    
    def test_workdir_path_conversion(self):
        """Test that workdir_path is properly converted to absolute Path."""
        provisioner = SkypilotProvisioner(
            workdir_path="./relative/path",
            cloud=sky.Azure(),
            num_nodes=1,
            servers_per_node=1,
        )
        
        # Should be converted to absolute path
        assert provisioner._workdir_path.is_absolute()
