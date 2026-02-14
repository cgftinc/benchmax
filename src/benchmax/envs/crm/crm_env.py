from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from benchmax.envs.mcp.parallel_mcp_env import ParallelMcpEnv
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from benchmax.envs.mcp.provisioners.skypilot_provisioner import SkypilotProvisioner
from benchmax.envs.types import StandardizedExample

if TYPE_CHECKING:
    import sky


SYSTEM_PROMPT = """\
You are an expert in Salesforce and you have access to a Salesforce instance.

# Instructions
- You will be provided a question, the system description, and relevant task context.
- Interact with the Salesforce instance using the tools provided to help you answer the question.
- You should ALWAYS make ONLY ONE tool call at a time. If you want to submit your final answer, just respond with the answer without tool call. If not, you should call some other tool.
- Always end by respond with ONLY the answer, NO full sentence or any explanation.
- If your answer is empty that is there are no records found matching the requirements mentioned, just return 'None' as the response.
- You should be able to get to an answer within 2-3 tool calls, so don't overthink.

Write your complete answer on the final line, within the xml tags <answer></answer>. If there are multiple answers, use comma as a delimiter.
e.g.
For Case IDs, final answer should look like <answer>0XA124XDF</answer>. If there are multiple, it could look like <answer>0XA124XDF, 001XX000003GXXX</answer>
For Months, it could look like <answer>May,July</answer>
If nothing matches, output <answer>None</answer>
"""


class CRMExample(StandardizedExample):
    reward_metric: str


class CRMEnv(ParallelMcpEnv):
    """Environment for CRM tasks using MCP with Salesforce"""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(self, workdir_path: Path, provisioner: BaseProvisioner, **kwargs):
        """Initialize CRMEnv."""
        super().__init__(workdir_path, provisioner, **kwargs)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> CRMExample:
        # convert dataset example into CRMExample (inherit from StandardizedExample)
        task: Optional[str] = example.get("task")
        persona: Optional[str] = example.get("persona")
        metadata: Optional[Dict[str, Any]] = example.get("metadata")
        answer: Optional[List[str]] = example.get("answer")
        query: Optional[str] = example.get("query")
        reward_metric: Optional[str] = example.get("reward_metric")

        if not task or not persona or not query or answer is None or not reward_metric:
            raise ValueError(
                "Example must contain 'task', 'persona', 'query', 'answer', and 'reward_metric' fields"
            )

        prompt = f"{persona}\n{task}\n{query}"
        if metadata and "required" in metadata:
            required_metadata = metadata["required"]
            prompt = f"{persona}\n{task}\n{required_metadata}\n{query}"

        return CRMExample(
            prompt=prompt,
            ground_truth=answer,
            init_rollout_args=None,
            reward_metric=reward_metric,
        )


class CRMEnvLocal(CRMEnv):
    """Import this env to run environment locally"""

    def __init__(self, num_local_servers: int = 5, **kwargs):
        workdir_path = Path(__file__).parent / "workdir"
        provisioner = LocalProvisioner(
            workdir_path=workdir_path, num_servers=num_local_servers
        )
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)


class CRMEnvSkypilot(CRMEnv):
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
                    "skypilot is required for CRMEnvSkypilot. Install with: pip install 'benchmax[skypilot]'"
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
