import pytest
import shutil
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Generator
from benchmax.envs.local_mcp_env import LocalMCPEnv, ClientWorkspacePair
from benchmax.envs.base_env import ToolDefinition

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
def benchmax_env(mcp_config: Dict[str, Any], tmp_path: Path) -> Generator[LocalMCPEnv, None, None]:
    """Fixture providing a configured LocalMCPEnv instance with cleanup"""
    benchmax_env = LocalMCPEnv(mcp_config, workspace_dir=tmp_path)
    yield benchmax_env
    benchmax_env.shutdown()  # Ensure proper cleanup after test

class TestLocalMCPEnv:
    def test_init(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test benchmax_env initialization and configuration"""
        benchmax_env = LocalMCPEnv(mcp_config, workspace_dir=tmp_path)
        try:
            assert benchmax_env._config == mcp_config
            assert benchmax_env._workspace_dir == tmp_path
            assert benchmax_env._pool_size == 3  # Default pool size
            assert len(benchmax_env._pre_warmed_pool) > 0  # Pool should be initialized
            assert isinstance(benchmax_env._pre_warmed_pool[0], ClientWorkspacePair)
        finally:
            benchmax_env.shutdown()

    def test_rollout_lifecycle(self, benchmax_env: LocalMCPEnv) -> None:
        """Test rollout initialization, workspace management, and cleanup"""
        rollout_id: str = "test_rollout"
        
        # Test init_rollout
        benchmax_env.init_rollout(rollout_id)
        assert rollout_id in benchmax_env._active_clients
        
        # Test get_rollout_workspace
        workspace: Path = benchmax_env.get_rollout_workspace(rollout_id)
        assert isinstance(workspace, Path)
        assert workspace.exists()
        assert workspace.is_dir()
        
        # Test cleanup_rollout
        benchmax_env.cleanup_rollout(rollout_id)
        assert benchmax_env._active_clients[rollout_id].client is None

        # Test error on invalid rollout
        with pytest.raises(ValueError):
            benchmax_env.get_rollout_workspace("invalid_rollout")

    def test_tool_definition_caching(self, benchmax_env: LocalMCPEnv) -> None:
        """Test that tool definitions are properly cached and contain expected tools"""
        # First call should populate cache
        tools_first: List[ToolDefinition] = benchmax_env.list_tools()
        
        # Verify specific tools are present
        tool_names = [t.name for t in tools_first]
        
        # Check for specific time tools
        assert "time_current_time" in tool_names
        assert "time_days_in_month" in tool_names
        
        # Check for specific filesystem tools
        assert "filesystem_read_file" in tool_names
        assert "filesystem_write_file" in tool_names
        
        # Second call should use cache
        tools_second: List[ToolDefinition] = benchmax_env.list_tools()
        
        assert tools_first == tools_second
        assert benchmax_env._tool_definitions is not None
        assert benchmax_env._tool_definitions == tools_first

    def test_allowed_tools_filtering(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test that tool filtering works correctly with allowed_tools list"""
        # Test with only time tools
        allowed_time_tools = [
            "time_current_time",
            "time_relative_time",
            "time_days_in_month"
        ]
        
        benchmax_env = LocalMCPEnv(mcp_config, allowed_tools=allowed_time_tools, workspace_dir=tmp_path)
        try:
            tools = benchmax_env.list_tools()
            tool_names = [t.name for t in tools]
            
            # Verify only allowed tools are present
            assert all(name in allowed_time_tools for name in tool_names)
            
            # Verify filesystem tools are filtered out
            assert all(not name.startswith("filesystem_") for name in tool_names)
            
            # Verify count matches
            assert len(tools) == len(allowed_time_tools)
            
        finally:
            benchmax_env.shutdown()

    def test_output_parsing(self, benchmax_env: LocalMCPEnv) -> None:
        """Test that tool output parsing works correctly"""
        # Define a mock parser
        def mock_parser(output):
            return f"Parsed: {output}"
        
        # Add the parser to the benchmax_env
        benchmax_env._output_parsers["time_current_time"] = mock_parser
        
        # Mock the tool output
        benchmax_env.init_rollout("test_rollout")
        result = benchmax_env.run_tool("test_rollout", "time_current_time", format="YYYY-MM-DD HH:mm:ss")
        
        # Verify the parser was applied
        assert result.startswith("Parsed: "), f"Unexpected parsed result: {result}"

    def test_run_tool_synchronously(self, benchmax_env: LocalMCPEnv) -> None:
        """Test running tools synchronously with actual time tool"""
        result: Optional[str] = benchmax_env.run_tool("test_rollout", "time_current_time", format="YYYY-MM-DD HH:mm:ss")
        
        # Verify result matches a time-like pattern
        assert result is not None
        import re
        assert re.match(r'.*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*', result), f"Unexpected time format: {result}"

    def test_workspace_cleanup(self, benchmax_env: LocalMCPEnv) -> None:
        """Test that workspaces are properly cleaned up using real filesystem operations"""
        rollout_id: str = "test_rollout"
        benchmax_env.init_rollout(rollout_id)
        workspace: Path = benchmax_env.get_rollout_workspace(rollout_id)
        
        # Use filesystem_list_directory to verify workspace exists
        result: Optional[str] = benchmax_env.run_tool("test_rollout", "filesystem_list_directory", path=str(workspace))
        assert result is not None
        
        benchmax_env.cleanup_rollout(rollout_id)
        assert benchmax_env._active_clients[rollout_id].client is None

    def test_shutdown(self, mcp_config: Dict[str, Any], tmp_path: Path) -> None:
        """Test proper cleanup during shutdown using real filesystem operations"""
        benchmax_env = LocalMCPEnv(mcp_config, pool_size=2, workspace_dir=tmp_path)
        
        # Initialize some rollouts
        benchmax_env.init_rollout("rollout1")
        benchmax_env.init_rollout("rollout2")
        
        # Get initial workspace list
        result: Optional[str] = benchmax_env.run_tool("rollout1", "filesystem_list_allowed_directories")
        assert result is not None
        
        benchmax_env.shutdown()
        assert len(benchmax_env._active_clients) == 0
        assert len(benchmax_env._pre_warmed_pool) == 0
        
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
        
        benchmax_env = LocalMCPEnv(config_file, workspace_dir=tmp_path)
        try:
            # Get base config without workspace paths
            base_config = {"mcpServers": {
                server: {k: v for k, v in info.items() if k != 'cwd'}
                for server, info in benchmax_env._config["mcpServers"].items()
            }}
            expected_base_config = {"mcpServers": {
                server: {k: v for k, v in info.items() if k != 'cwd'}
                for server, info in config["mcpServers"].items()
            }}
            assert base_config == expected_base_config
        finally:
            benchmax_env.shutdown()
            
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
        
        benchmax_env = None
        try:
            # Create benchmax_env with config from file
            benchmax_env = LocalMCPEnv(config_file, workspace_dir=tmp_path)
            
            # List tools to ensure server is initialized
            tools = benchmax_env.list_tools()
            assert len(tools) > 0, "No tools available"
            
            # Wait for tool initialization
            tools = benchmax_env.list_tools()  # Try listing tools again
            tool_names = [tool.name for tool in tools]
            assert "current_time" in tool_names, f"current_time not found in available tools: {tool_names}"
            
            # Test MCP server functionality
            result = benchmax_env.run_tool("test_rollout", "current_time", format="YYYY-MM-DD HH:mm:ss")
            assert result is not None
            import re
            assert re.match(r'.*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*', result)
            
            # Verify config was loaded correctly
            assert "time" in benchmax_env._config["mcpServers"]
            assert benchmax_env._config["mcpServers"]["time"]["command"] == "npx"
            assert benchmax_env._config["mcpServers"]["time"]["args"] == ["-y", "time-mcp"]
        finally:
            if benchmax_env is not None:
                benchmax_env.shutdown()
            # Clean up config file
            config_file.unlink()
