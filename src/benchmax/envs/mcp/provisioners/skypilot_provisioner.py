"""
SkyPilot provisioner for launching cloud-based server clusters.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional
import sky

from .base_provisioner import BaseProvisioner
from .utils import get_run_command, setup_sync_dir, cleanup_dir, get_setup_command

logger = logging.getLogger(__name__)


class SkypilotProvisioner(BaseProvisioner):
    """
    Provisioner that launches a SkyPilot cluster in the cloud.

    Use this for:
    - Production-scale parallel execution
    - Distributed benchmarking across many nodes
    - Cloud-based compute resource management

    Example:
        import sky
        provisioner = SkypilotProvisioner(
            workdir_path=Path("my_workdir"),
            cloud=sky.Azure(),
            num_nodes=5,
            servers_per_node=4,
            cpus=4,
            memory=16,
        )
        # Will provision 20 total servers (5 nodes * 4 servers/node)
    """

    def __init__(
        self,
        workdir_path: Path | str,
        cloud: sky.clouds.Cloud,
        num_nodes: int = 1,
        servers_per_node: int = 5,
        cpus: Optional[str | int] = "2+",
        memory: Optional[str | int] = "8+",
        base_cluster_name: str = "benchmax-env-cluster",
    ):
        """
        Initialize SkyPilot provisioner.

        Args:
            workdir_path: Path to workdir containing mcp_config.yaml, setup.sh, etc.
            cloud: SkyPilot cloud instance (e.g., sky.AWS(), sky.Azure(), sky.GCP()).
            num_nodes: Number of nodes in the cluster.
            servers_per_node: Number of proxy servers to run on each node.
            cpus: CPU requirement per node (e.g., "2+", 4, "8").
            memory: Memory requirement per node in GB (e.g., "16+", 32).
            base_cluster_name: Base name for the cluster (timestamp will be appended).
        """
        if num_nodes < 1:
            raise ValueError("num_nodes must be at least 1")
        if servers_per_node < 1 or servers_per_node > 100:
            raise ValueError("servers_per_node must be between 1 and 100")

        self._workdir_path = Path(workdir_path).absolute()
        self._cloud = cloud
        self._num_nodes = num_nodes
        self._servers_per_node = servers_per_node
        self._total_num_servers = num_nodes * servers_per_node
        self._cpus = cpus
        self._memory = memory

        # Generate unique cluster name
        unique_suffix = uuid.uuid4().hex[:4]
        self._cluster_name = f"{base_cluster_name}-{unique_suffix}"

        # Internal state
        self._sync_workdir: Optional[Path] = None
        self._cluster_provisioned: bool = False

        logger.info(
            f"SkypilotProvisioner configured: {num_nodes} nodes * {servers_per_node} servers/node = "
            f"{self._total_num_servers} total servers, cluster: '{self._cluster_name}'"
        )

    @property
    def num_servers(self) -> int:
        """
        Total number of servers
        """
        return self._total_num_servers

    async def provision_servers(self, api_secret: str) -> List[str]:
        """
        Launch SkyPilot cluster and return server addresses.

        Args:
            api_secret: API token for server authentication.

        Returns:
            List of server addresses in "host:port" format.
        """
        if self._cluster_provisioned:
            raise RuntimeError("Cluster already provisioned. Call teardown() first.")

        self._cluster_provisioned = True

        logger.info(f"Launching SkyPilot cluster '{self._cluster_name}'...")

        # Setup sync directory with workdir contents + proxy_server.py
        try:
            self._sync_workdir = setup_sync_dir(self._workdir_path)
            logger.debug(f"Synced workdir to temporary directory: {self._sync_workdir}")
        except Exception as e:
            logger.error(f"Failed to setup sync directory: {e}")
            raise

        # Calculate ports
        base_port = 8080
        all_ports = [str(base_port + i) for i in range(self._servers_per_node)]

        env = None if api_secret is None else {"API_SECRET": api_secret}

        # Configure SkyPilot task
        sky_task = sky.Task(
            name="mcp-server",
            run=get_run_command(ports=all_ports),
            setup=get_setup_command(),
            workdir=str(self._sync_workdir),
            num_nodes=self._num_nodes,
            envs=env,
        )

        sky_task.set_resources(
            sky.Resources(
                cloud=self._cloud,
                cpus=self._cpus,
                memory=self._memory,
                ports=all_ports,
            )
        )

        # Launch cluster
        logger.info(
            f"Submitting cluster launch: {self._num_nodes} nodes, "
            f"{self._cpus} CPUs, {self._memory}GB memory per node"
        )
        cluster_handle = None
        try:
            _, handle = sky.launch(
                task=sky_task,
                cluster_name=self._cluster_name,
                detach_run=True,
                detach_setup=True,
                retry_until_up=True,
            )
            cluster_handle = handle
        except Exception as e:
            logger.error(f"Failed to launch SkyPilot cluster: {e}")
            cleanup_dir(self._sync_workdir)
            self._sync_workdir = None
            raise RuntimeError(f"SkyPilot cluster launch failed: {e}") from e

        if cluster_handle is None:
            cleanup_dir(self._sync_workdir)
            self._sync_workdir = None
            raise RuntimeError("SkyPilot launch returned no handle")

        # Collect server addresses
        addresses = []
        for node_idx, (_, node_ip) in enumerate(
            cluster_handle.stable_internal_external_ips
        ):
            for port in all_ports:
                addresses.append(f"{node_ip}:{port}")
            logger.debug(f"Node {node_idx}: {node_ip} with {len(all_ports)} servers")

        logger.info(
            f"Successfully launched cluster '{self._cluster_name}' "
            f"with {len(addresses)} servers across {self._num_nodes} node(s)"
        )
        return addresses

    async def teardown(self) -> None:
        """
        Tear down SkyPilot cluster and clean up resources.
        """
        if self._cluster_provisioned is None:
            logger.warning("teardown() called but no cluster is active.")
            return

        logger.info(f"Tearing down SkyPilot cluster '{self._cluster_name}'...")
        try:
            sky.down(cluster_name=self._cluster_name)
            logger.info(f"Cluster '{self._cluster_name}' torn down successfully")
        except Exception as e:
            logger.error(f"Error tearing down cluster '{self._cluster_name}': {e}")
        finally:
            self._cluster_provisioned = False
            if self._sync_workdir:
                cleanup_dir(self._sync_workdir)
                self._sync_workdir = None
                logger.debug("Cleaned up sync directory")

    @property
    def cluster_name(self) -> str:
        """
        Unique name of the SkyPilot cluster.
        """
        return self._cluster_name
