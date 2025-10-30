"""
Integration tests for LocalProvisioner.
"""

import pytest
import asyncio
from pathlib import Path
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from tests.integration.envs.mcp.provisioners.utils import wait_for_server_health, check_health


class TestEndToEnd:
    """End-to-end integration tests for provisioning and teardown."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_server_lifecycle(self, example_workdir: Path):
        """Test complete lifecycle of a single server: provision, verify, and teardown."""
        provisioner = LocalProvisioner(
            workdir_path=example_workdir,
            num_servers=1,
            base_port=9000,
        )
        
        # Provision
        api_secret = "single-server-test-secret-32chars!!"
        addresses = await provisioner.provision_servers(api_secret)
        assert len(addresses) == 1
        assert addresses[0] == "localhost:9000"
        
        # Verify server is up and healthy
        is_healthy = await wait_for_server_health(addresses[0])
        assert is_healthy, "Server failed to become healthy"
        
        # Verify process is running
        assert len(provisioner._processes) == 1
        assert provisioner._processes[0].poll() is None

        # Check that double-provisioning would result in an error
        with pytest.raises(RuntimeError, match="already provisioned"):
            await provisioner.provision_servers(api_secret)
        
        # Teardown
        await provisioner.teardown()
        
        # Verify cleanup
        assert len(provisioner._processes) == 0
        await asyncio.sleep(0.5)
        assert not await check_health("localhost:9000"), "Server still responding after teardown"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_servers_lifecycle(self, example_workdir: Path):
        """Test complete lifecycle of multiple servers: provision, verify, and teardown."""
        provisioner = LocalProvisioner(
            workdir_path=example_workdir,
            num_servers=5,
            base_port=9100,
        )
        
        # Provision
        api_secret = "single-server-test-secret-32chars!!"
        addresses = await provisioner.provision_servers(api_secret)
        assert len(addresses) == 5
        expected_addresses = [f"localhost:{9100 + i}" for i in range(5)]
        assert addresses == expected_addresses
        
        # Verify all servers are up and healthy
        health_checks = await asyncio.gather(
            *[wait_for_server_health(addr) for addr in addresses]
        )
        assert all(health_checks), "Not all servers became healthy"
        
        # Verify all processes are running
        assert len(provisioner._processes) == 5
        assert all(p.poll() is None for p in provisioner._processes)
        
        # Teardown
        await provisioner.teardown()
        
        # Verify cleanup
        assert len(provisioner._processes) == 0
        await asyncio.sleep(0.5)
        
        # Verify all servers are down
        for addr in addresses:
            assert not await check_health(addr), f"Server {addr} still responding after teardown"


class TestValidation:
    """Test parameter validation."""
    
    def test_invalid_num_servers(self):
        """Test validation of num_servers parameter."""
        with pytest.raises(ValueError, match="at least 1"):
            LocalProvisioner(workdir_path=".", num_servers=0, base_port=8080)
    
    def test_invalid_base_port(self):
        """Test validation of base_port parameter."""
        with pytest.raises(ValueError, match="between 1024 and 65535"):
            LocalProvisioner(workdir_path=".", num_servers=1, base_port=500)
    
    def test_port_range_exceeds_max(self):
        """Test validation of port range."""
        with pytest.raises(ValueError, match="exceeds max port"):
            LocalProvisioner(workdir_path=".", num_servers=100, base_port=65500)
