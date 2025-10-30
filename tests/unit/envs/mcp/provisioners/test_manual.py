import pytest
from benchmax.envs.mcp.provisioners import ManualProvisioner


class TestManualProvisioner:
    """Tests for ManualProvisioner class."""

    def test_init_with_valid_addresses(self):
        """Test initialization with valid addresses."""
        addresses = ["localhost:8080", "192.168.1.10:8080"]
        provisioner = ManualProvisioner(addresses)

        assert provisioner._addresses == addresses

    def test_init_with_empty_list_raises_error(self):
        """Test that empty address list raises ValueError."""
        with pytest.raises(ValueError, match="at least one address"):
            ManualProvisioner([])

    @pytest.mark.asyncio
    async def test_provision_servers_returns_addresses(self):
        """Test that provision_servers returns configured addresses."""
        addresses = ["localhost:8080", "localhost:8081"]
        provisioner = ManualProvisioner(addresses)

        result = await provisioner.provision_servers("dummy-api-secret")

        assert result == addresses

    @pytest.mark.asyncio
    async def test_provision_servers_returns_copy(self):
        """Test that provision_servers returns a copy, not the original list."""
        addresses = ["localhost:8080"]
        provisioner = ManualProvisioner(addresses)

        result = await provisioner.provision_servers("dummy-api-secret")
        result.append("modified")

        # Original should be unchanged
        assert provisioner._addresses == ["localhost:8080"]

    @pytest.mark.asyncio
    async def test_teardown_is_noop(self):
        """Test that teardown completes without error (no-op)."""
        provisioner = ManualProvisioner(["localhost:8080"])
        await provisioner.teardown()  # Should not raise

    @pytest.mark.asyncio
    async def test_multiple_provision_calls(self):
        """Test that provision_servers can be called multiple times."""
        provisioner = ManualProvisioner(["localhost:8080"])

        result1 = await provisioner.provision_servers("dummy-api-secret")
        result2 = await provisioner.provision_servers("dummy-api-secret")

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_provision_after_teardown(self):
        """Test that provision_servers works after teardown."""
        provisioner = ManualProvisioner(["localhost:8080"])

        await provisioner.provision_servers("dummy-api-secret")
        await provisioner.teardown()
        result = await provisioner.provision_servers("dummy-api-secret")

        assert result == ["localhost:8080"]
