from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmax.envs.mcp.parallel_mcp_env import ParallelMcpEnv
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from benchmax.envs.mcp.provisioners.skypilot_provisioner import SkypilotProvisioner
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example

if TYPE_CHECKING:
    import sky

SYSTEM_PROMPT = """Please use the tools provided to do any computation.
Write your complete answer on the final line only, within the xml tags <answer></answer>.\n
"""


class MathEnv(ParallelMcpEnv):
    """Environment for math problems, using local MCP tools."""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(self, workdir_path: Path, provisioner: BaseProvisioner, **kwargs):
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        return make_example(
            prompt_messages=[{"role": "user", "content": example.get("task", "")}],
            task={"ground_truth": example.get("answer", "")},
            system_prompt=cls.system_prompt,
        )


class MathEnvLocal(MathEnv):
    """Import this env to run environment locally"""

    def __init__(self, num_local_servers: int = 5, **kwargs):
        workdir_path = Path(__file__).parent / "workdir"
        provisioner = LocalProvisioner(
            workdir_path=workdir_path, num_servers=num_local_servers
        )
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)


class MathEnvSkypilot(MathEnv):
    """Import this env to run environment on any cloud (i.e. AWS / GCP / Azure) with Skypilot"""

    def __init__(
        self,
        cloud: "sky.clouds.Cloud | None" = None,
        num_nodes: int = 2,
        servers_per_node: int = 5,
        **kwargs,
    ):
        if cloud is None:
            try:
                import sky
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "skypilot is required for MathEnvSkypilot. Install with: pip install 'benchmax[skypilot]'"
                ) from e
            cloud = sky.Azure()
        workdir_path = Path(__file__).parent / "workdir"
        provisioner = SkypilotProvisioner(
            workdir_path=workdir_path,
            cloud=cloud,
            num_nodes=num_nodes,
            servers_per_node=servers_per_node,
        )
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)
