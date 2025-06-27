from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Adjust import paths to match your project layout
from envs.mcp_sandbox import MCPSandbox
from envs.base_sandbox import ToolDefinition
from verl.tools.base_tool import BaseTool
from verl.schemas import OpenAIFunctionToolSchema


class SandboxToolAdapter(BaseTool):
    """
    Wrap *one* MCPSandbox tool behind the BaseTool interface expected by the
    evaluator/trainer stack.

    - `instance_id` ↔ `rollout_id`: we simply pass the value straight through.
    - Rewards are left as zero; add your own logic later if needed.
    """

    def __init__(self, sandbox: MCPSandbox, tool_def: ToolDefinition):
        self._sandbox = sandbox
        self._tool_def = tool_def
        super().__init__(config={}, tool_schema=None)  # schema supplied lazily below

    # ------------------------------------------------------------------ #
    # BaseTool API
    # ------------------------------------------------------------------ #
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Generate a minimal function-tool schema on-the-fly."""
        return OpenAIFunctionToolSchema(
            function=OpenAIFunctionToolSchema.Function(
                name=self._tool_def.name,
                description=self._tool_def.description or "",
                parameters=self._tool_def.property_schema or {},
            )
        )

    async def create(self, instance_id: Optional[str] = None, **_) -> str:
        """Just echo or mint an ID; nothing to initialise on the sandbox side."""
        return instance_id or str(uuid4())

    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **_
    ) -> Tuple[str, float, Dict]:
        """
        Forward the call to the underlying sandbox tool, injecting `rollout_id`
        so everything downstream can treat it as the same handle.
        """
        response = self._sandbox.run_tool(
            self._tool_def.name,
            parameters,
            rollout_id=instance_id,
            parse_output=True,
        )
        return response, 0.0, {}      # (tool_response, step_reward, metrics)

    async def calc_reward(self, instance_id: str, **_) -> float:
        """No per-step reward in this generic adapter."""
        return 0.0

    async def release(self, instance_id: str, **_) -> None:
        """Nothing to clean up for now."""
        pass


# ---------------------------------------------------------------------- #
# Public helper
# ---------------------------------------------------------------------- #
def sandbox_to_tool_list(sandbox: MCPSandbox) -> List[BaseTool]:
    """
    Convert **all** tools registered inside `sandbox` into `BaseTool` instances.

    Example
    -------
    >>> msb = MCPSandbox("mcp.json")
    >>> tool_objs = sandbox_to_tool_list(msb)
    >>> first_tool_response, *_ = await tool_objs[0].execute("demo-id", {"arg": 1})
    """
    return [
        SandboxToolAdapter(sandbox, tool_def)
        for tool_def in sandbox._tools.values()
    ]
