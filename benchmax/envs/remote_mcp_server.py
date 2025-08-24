from benchmax.envs.base_env import BaseEnv
from typing import List, Any, Optional
from pathlib import Path

from benchmax.envs.remote_host.base_host import RemoteHost
from benchmax.envs.types import ToolDefinition

class RemoteMCPEnv(BaseEnv):
    """Remote MCP Environment for managing tool execution and rollouts with a remote MCP server.
    Currently only supports running on Modal containers.
    """
    def __init__(
        self,
        remote_host: RemoteHost,
        allowed_tools: Optional[List[str]] = None,
        workspace_dir: Optional[Path] = None
    ) -> None:
        """Initialize the environment with configuration and pool settings."""
        super().__init__()
        self.remote_host = remote_host
        self.allowed_tools = allowed_tools or []
        self.workspace_dir = workspace_dir or Path("/tmp/remote_mcp_workspace")

    # ---- Public API Methods ----
    def shutdown(self) -> None:
        """Clean up resources and stop the event loop."""
    
    def list_tools(self) -> List[ToolDefinition]:
        """List available tools, using cached definitions if availabaz5le"""
        pass

    def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute a tool in the context of a specific rollout"""
        pass

    def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        """Initialize resources for a new rollout"""
        pass

    # TODO: Make this a private method that gets called by base class as a "fake" reward computation
    def cleanup_rollout(self, rollout_id: str, keep_workspace=True) -> None:
        """
        """
        pass

    def get_rollout_workspace(self, rollout_id: str, strict_check: bool = False) -> Path:
        """
        """
        pass

    def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        """Copy a file to the workspace for a specific rollout. If dst_filename is None, use the original filename."""
        pass

    def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        """Copy a file from the workspace for a specific rollout"""
        pass
