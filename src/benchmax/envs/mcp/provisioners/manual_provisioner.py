"""
Manual provisioner for using pre-existing servers.
"""

import logging
from typing import List
from .base_provisioner import BaseProvisioner

logger = logging.getLogger(__name__)


class ManualProvisioner(BaseProvisioner):
    """
    Provisioner for manually specified server addresses.

    Use this when you have already started servers and want to provide
    their addresses directly. Useful for:
    - Debugging with pre-started local servers
    - Testing against persistent test infrastructure
    - Using servers managed by external systems

    Example:
        provisioner = ManualProvisioner([
            "localhost:8080",
            "localhost:8081",
            "192.168.1.10:8080"
        ])
    """

    def __init__(self, addresses: List[str]):
        """
        Initialize with pre-existing server addresses.

        Args:
            addresses: List of server addresses in "host:port" format.
        """
        if not addresses:
            raise ValueError("ManualProvisioner requires at least one address")

        self._addresses = addresses
        logger.info(f"ManualProvisioner configured with {len(addresses)} addresses")

    @property
    def num_servers(self) -> int:
        """
        Total number of servers
        """
        return len(self._addresses)

    async def provision_servers(self, api_secret: str) -> List[str]:
        """
        Return the pre-configured server addresses.

        Args:
            api_secret: Unused in this function. Servers already have set ther api secret.

        Returns:
            The list of addresses provided during initialization.
        """
        logger.info(f"Using {len(self._addresses)} manually configured servers")
        return self._addresses.copy()

    async def teardown(self) -> None:
        """
        No-op teardown since servers are externally managed.

        ManualProvisioner does not start servers, so it does not stop them.
        The user is responsible for managing the server lifecycle.
        """
        logger.info("ManualProvisioner teardown (servers externally managed)")
