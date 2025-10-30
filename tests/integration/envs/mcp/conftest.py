"""
Integration fixtures for MCP environment tests.
These may start subprocesses, use real ports, or access the filesystem.
"""

import uuid
import pytest
from pathlib import Path
from typing import Tuple, List
from collections.abc import AsyncGenerator
import sky

from benchmax.envs.mcp import ParallelMcpEnv
from benchmax.envs.mcp.provisioners import (
    LocalProvisioner,
    ManualProvisioner,
    SkypilotProvisioner,
)

# ===== Session-scoped: Provision servers once for all tests =====


@pytest.fixture(scope="session")
async def local_servers_with_secret(
    example_workdir: Path,
) -> AsyncGenerator[Tuple[List[str], str], None]:
    """
    Provision local servers once for the entire test session.
    Returns (addresses, api_secret) tuple.

    These servers are reused across tests for speed.
    """
    api_secret = uuid.uuid4().hex
    provisioner = LocalProvisioner(
        workdir_path=example_workdir,
        num_servers=4,
        base_port=8080,
    )

    addresses = await provisioner.provision_servers(api_secret)
    print(f"\n[Session Setup] Provisioned {len(addresses)} local servers")

    yield addresses, api_secret

    print(f"\n[Session Teardown] Tearing down {len(addresses)} local servers")
    await provisioner.teardown()


@pytest.fixture(scope="session")
async def skypilot_servers_with_secret(
    example_workdir: Path,
) -> AsyncGenerator[Tuple[List[str], str], None]:
    """
    Provision Skypilot servers once for the entire test session.
    Returns (addresses, api_secret) tuple.

    These servers are reused across tests for speed.
    Only provisioned if test actually uses this fixture.
    """

    api_secret = uuid.uuid4().hex
    provisioner = SkypilotProvisioner(
        workdir_path=example_workdir,
        cloud=sky.Azure(),
        num_nodes=2,
        servers_per_node=4,
        base_cluster_name="test-cluster",
        cpus=2,
        memory=8
    )

    addresses = await provisioner.provision_servers(api_secret)
    print(f"\n[Session Setup] Provisioned {len(addresses)} Skypilot servers")

    yield addresses, api_secret

    print(f"\n[Session Teardown] Tearing down {len(addresses)} Skypilot servers")
    await provisioner.teardown()


# ===== Function-scoped: Fresh env for each test =====


@pytest.fixture
async def local_env(
    local_servers_with_secret: Tuple[List[str], str], example_workdir: Path
) -> AsyncGenerator[ParallelMcpEnv, None]:
    """
    Create a fresh ParallelMcpEnv using reused local servers.

    Each test gets a clean env instance, but servers are shared
    across tests for speed. This means server state may be
    contaminated, but that's acceptable for most tests.
    """
    addresses, api_secret = local_servers_with_secret

    manual_provisioner = ManualProvisioner(addresses)
    env = ParallelMcpEnv(
        workdir_path=example_workdir,
        provisioner=manual_provisioner,
        api_secret=api_secret,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()


@pytest.fixture
async def skypilot_env(
    skypilot_servers_with_secret: Tuple[List[str], str], example_workdir: Path
) -> AsyncGenerator[ParallelMcpEnv, None]:
    """
    Create a fresh ParallelMcpEnv using reused Skypilot servers.

    Each test gets a clean env instance, but servers are shared
    across tests for speed. Mark tests using this with @pytest.mark.remote.
    """
    addresses, api_secret = skypilot_servers_with_secret

    manual_provisioner = ManualProvisioner(addresses)
    env = ParallelMcpEnv(
        workdir_path=example_workdir,
        provisioner=manual_provisioner,
        api_secret=api_secret,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()


@pytest.fixture
async def fresh_local_env(example_workdir: Path) -> AsyncGenerator[ParallelMcpEnv, None]:
    """
    Create a fresh ParallelMcpEnv with its own dedicated servers.

    Use this for tests that need clean server state.
    Slower than local_env but provides isolation.
    Mark tests using this with @pytest.mark.slow.
    """
    provisioner = LocalProvisioner(
        workdir_path=example_workdir,
        num_servers=4,
        base_port=9080,  # Different port range to avoid conflicts
    )

    env = ParallelMcpEnv(
        workdir_path=example_workdir,
        provisioner=provisioner,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()


@pytest.fixture
async def fresh_skypilot_env(
    example_workdir: Path,
) -> AsyncGenerator[ParallelMcpEnv, None]:
    """
    Create a fresh ParallelMcpEnv with its own dedicated Skypilot servers.

    Use this for E2E tests that need clean server state on cloud infrastructure.
    Slower and more expensive than skypilot_env.
    Mark tests using this with @pytest.mark.remote and @pytest.mark.slow.
    """
    provisioner = SkypilotProvisioner(
        workdir_path=example_workdir,
        cloud=sky.Azure(),
        num_nodes=2,
        servers_per_node=4,
        base_cluster_name="test-cluster-fresh",
        cpus=2,
        memory=8,
    )

    env = ParallelMcpEnv(
        workdir_path=example_workdir,
        provisioner=provisioner,
        provision_at_init=True,
    )

    yield env

    await env.shutdown()

