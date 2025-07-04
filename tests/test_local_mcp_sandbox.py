import pytest
import shutil
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Generator
from envs.local_mcp_sandbox import LocalMCPSandbox, ClientWorkspacePair
from envs.base_sandbox import ToolDefinition

@pytest.fixture
def mcp_config(tmp_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Fixture providing a basic MCP server configuration with temporary workspace"""
    config = {
        "mcpServers": {
            "time": {
                "command": "npx",
                "args": ["-y", "time-mcp"]
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"]
            }
        }
    }
    yield config
    # Cleanup any test workspaces
    workspace_path = tmp_path / "workspaces"
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

@pytest.fixture
def sandbox(mcp_config: Dict[str, Any], tmp_path: Path) -> Generator[LocalMCPSandbox, None, None]:
    """Fixture providing a configured LocalMCPSandbox instance with cleanup"""
    sandbox = LocalMCPSandbox(mcp_config, workspace_dir=tmp_path)
    yield sandbox
    sandbox.shutdown()  # Ensure proper cleanup after test

class TestLocalMCPSandbox:
    def test_init(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test sandbox initialization and configuration"""
        sandbox = LocalMCPSandbox(mcp_config, workspace_dir=tmp_path)
        try:
            assert sandbox._config == mcp_config
            assert sandbox._workspace_dir == tmp_path
            assert sandbox._pool_size == 3  # Default pool size
            assert len(sandbox._pre_warmed_pool) > 0  # Pool should be initialized
            assert isinstance(sandbox._pre_warmed_pool[0], ClientWorkspacePair)
        finally:
            sandbox.shutdown()

    def test_rollout_lifecycle(self, sandbox: LocalMCPSandbox) -> None:
        """Test rollout initialization, workspace management, and cleanup"""
        rollout_id: str = "test_rollout"
        
        # Test init_rollout
        sandbox.init_rollout(rollout_id)
        assert rollout_id in sandbox._active_clients
        
        # Test get_rollout_workspace
        workspace: Path = sandbox.get_rollout_workspace(rollout_id)
        assert isinstance(workspace, Path)
        assert workspace.exists()
        assert workspace.is_dir()
        
        # Test cleanup_rollout
        sandbox.cleanup_rollout(rollout_id)
        assert rollout_id not in sandbox._active_clients
        
        # Test error on invalid rollout
        with pytest.raises(ValueError):
            sandbox.get_rollout_workspace("invalid_rollout")

    def test_tool_definition_caching(self, sandbox: LocalMCPSandbox) -> None:
        """Test that tool definitions are properly cached and contain expected tools"""
        # First call should populate cache
        tools_first: List[ToolDefinition] = sandbox.list_tools()
        
        # Verify specific tools are present
        tool_names = [t.name for t in tools_first]
        
        # Check for specific time tools
        assert "time_current_time" in tool_names
        assert "time_days_in_month" in tool_names
        
        # Check for specific filesystem tools
        assert "filesystem_read_file" in tool_names
        assert "filesystem_write_file" in tool_names
        
        # Second call should use cache
        tools_second: List[ToolDefinition] = sandbox.list_tools()
        
        assert tools_first == tools_second
        assert sandbox._tool_definitions is not None
        assert sandbox._tool_definitions == tools_first

    def test_allowed_tools_filtering(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test that tool filtering works correctly with allowed_tools list"""
        # Test with only time tools
        allowed_time_tools = [
            "time_current_time",
            "time_relative_time",
            "time_days_in_month"
        ]
        
        sandbox = LocalMCPSandbox(mcp_config, allowed_tools=allowed_time_tools, workspace_dir=tmp_path)
        try:
            tools = sandbox.list_tools()
            tool_names = [t.name for t in tools]
            
            # Verify only allowed tools are present
            assert all(name in allowed_time_tools for name in tool_names)
            
            # Verify filesystem tools are filtered out
            assert all(not name.startswith("filesystem_") for name in tool_names)
            
            # Verify count matches
            assert len(tools) == len(allowed_time_tools)
            
        finally:
            sandbox.shutdown()

    def test_run_tool_synchronously(self, sandbox: LocalMCPSandbox) -> None:
        """Test running tools synchronously with actual time tool"""
        result: Optional[str] = sandbox.run_tool("test_rollout", "time_current_time", format="YYYY-MM-DD HH:mm:ss")
        
        # Verify result matches a time-like pattern
        assert result is not None
        import re
        assert re.match(r'.*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*', result), f"Unexpected time format: {result}"

    def test_workspace_cleanup(self, sandbox: LocalMCPSandbox) -> None:
        """Test that workspaces are properly cleaned up using real filesystem operations"""
        rollout_id: str = "test_rollout"
        sandbox.init_rollout(rollout_id)
        workspace: Path = sandbox.get_rollout_workspace(rollout_id)
        
        # Use filesystem_list_directory to verify workspace exists
        result: Optional[str] = sandbox.run_tool("test_rollout", "filesystem_list_directory", path=str(workspace))
        assert result is not None
        
        sandbox.cleanup_rollout(rollout_id)
        assert rollout_id not in sandbox._active_clients

    def test_shutdown(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test proper cleanup during shutdown using real filesystem operations"""
        sandbox = LocalMCPSandbox(mcp_config, pool_size=2, workspace_dir=tmp_path)
        
        # Initialize some rollouts
        sandbox.init_rollout("rollout1")
        sandbox.init_rollout("rollout2")
        
        # Get initial workspace list
        result: Optional[str] = sandbox.run_tool("rollout1", "filesystem_list_allowed_directories")
        assert result is not None
        
        sandbox.shutdown()
        assert len(sandbox._active_clients) == 0
        assert len(sandbox._pre_warmed_pool) == 0
        
    def test_config_loading_from_file(self, tmp_path: Path) -> None:
        """Test loading config from file path"""
        config_file = tmp_path / "config.json"
        config = {
            "mcpServers": {
                "time": {
                    "command": "npx",
                    "args": ["-y", "time-mcp"]
                }
            }
        }
        config_file.write_text(json.dumps(config))
        
        sandbox = LocalMCPSandbox(config_file, workspace_dir=tmp_path)
        try:
            # Get base config without workspace paths
            base_config = {"mcpServers": {
                server: {k: v for k, v in info.items() if k != 'cwd'}
                for server, info in sandbox._config["mcpServers"].items()
            }}
            expected_base_config = {"mcpServers": {
                server: {k: v for k, v in info.items() if k != 'cwd'}
                for server, info in config["mcpServers"].items()
            }}
            assert base_config == expected_base_config
        finally:
            sandbox.shutdown()
            
    def test_mcp_server_with_config_file(self, tmp_path: Path) -> None:
        """Test MCP server functionality using config loaded from file"""
        config_file = tmp_path / "test_config.json"
        config = {
            "mcpServers": {
                "time": {
                    "command": "npx",
                    "args": ["-y", "time-mcp"]
                }
            }
        }
        config_file.write_text(json.dumps(config))
        
        sandbox = None
        try:
            # Create sandbox with config from file
            sandbox = LocalMCPSandbox(config_file, workspace_dir=tmp_path)
            
            # List tools to ensure server is initialized
            tools = sandbox.list_tools()
            assert len(tools) > 0, "No tools available"
            
            # Wait for tool initialization
            tools = sandbox.list_tools()  # Try listing tools again
            tool_names = [tool.name for tool in tools]
            assert "current_time" in tool_names, f"current_time not found in available tools: {tool_names}"
            
            # Test MCP server functionality
            result = sandbox.run_tool("test_rollout", "current_time", format="YYYY-MM-DD HH:mm:ss")
            assert result is not None
            import re
            assert re.match(r'.*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*', result)
            
            # Verify config was loaded correctly
            assert "time" in sandbox._config["mcpServers"]
            assert sandbox._config["mcpServers"]["time"]["command"] == "npx"
            assert sandbox._config["mcpServers"]["time"]["args"] == ["-y", "time-mcp"]
        finally:
            if sandbox is not None:
                sandbox.shutdown()
            # Clean up config file
            config_file.unlink()
