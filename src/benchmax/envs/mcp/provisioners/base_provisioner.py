"""
Base provisioner interface for server provisioning strategies.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseProvisioner(ABC):
    """
    Abstract base class for server provisioning strategies.

    A provisioner is responsible for:
    1. Starting/launching servers (returning their addresses)
    2. Cleaning up resources when done
    """

    @property
    @abstractmethod
    def num_servers(self) -> int:
        """
        Total number of servers

        This reports the number of servers that are / will be provisioned.
        """
        pass

    @abstractmethod
    async def provision_servers(self, api_secret: str) -> List[str]:
        """
        Provision servers and return their addresses.

        Args:
            api_secret: Secret for server authentication.

        Returns:
            List of server addresses in "host:port" format.
            Example: ["localhost:8080", "192.168.1.10:8080"]
        """
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """
        Tear down provisioned resources.

        This should clean up any resources created during provisioning,
        such as stopping processes, terminating cloud instances, etc.
        """
        pass
