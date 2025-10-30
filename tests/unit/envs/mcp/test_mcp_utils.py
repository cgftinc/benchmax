"""
Tests for MCP utility functions.
"""

import jwt
from mcp import Tool
from benchmax.envs.mcp.utils import (
    convert_tool_definitions,
    generate_jwt_token,
    get_auth_headers,
)
from benchmax.envs.types import ToolDefinition


class TestConvertToolDefinitions:
    """Tests for convert_tool_definitions function."""

    def test_convert_tool_definitions_basic(self):
        """Test basic conversion of MCP tools to ToolDefinitions."""
        mcp_tools = [
            Tool(
                name="read_file",
                description="Read a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            ),
            Tool(
                name="write_file",
                description="Write to a file",
                inputSchema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            ),
        ]

        result = convert_tool_definitions(mcp_tools, allowed_tools=None)

        assert len(result) == 2
        assert all(isinstance(t, ToolDefinition) for t in result)
        assert result[0].name == "read_file"
        assert result[1].name == "write_file"

    def test_convert_tool_definitions_with_filter(self):
        """Test filtering tools with allowed_tools list."""
        mcp_tools = [
            Tool(name="read_file", description="Read", inputSchema={}),
            Tool(name="write_file", description="Write", inputSchema={}),
            Tool(name="execute", description="Execute", inputSchema={}),
        ]

        allowed = ["read_file", "write_file"]
        result = convert_tool_definitions(mcp_tools, allowed_tools=allowed)

        assert len(result) == 2
        assert all(t.name in allowed for t in result)
        assert not any(t.name == "execute" for t in result)

    def test_convert_tool_definitions_empty_description(self):
        """Test handling of tools with no description."""
        mcp_tools = [Tool(name="tool1", description=None, inputSchema={})]

        result = convert_tool_definitions(mcp_tools, allowed_tools=None)

        assert len(result) == 1
        assert result[0].description == ""

    def test_convert_tool_definitions_empty_list(self):
        """Test conversion of empty tool list."""
        result = convert_tool_definitions([], allowed_tools=None)
        assert result == []


class TestGenerateJwtToken:
    """Tests for generate_jwt_token function."""

    def test_generate_jwt_token_contains_standard_claims(self):
        """Ensure generated JWT includes required standard claims."""
        secret = "secret"
        token = generate_jwt_token(secret)
        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )

        assert decoded["iss"] == "mcp-client"
        assert decoded["aud"] == "mcp-proxy-server"
        assert "iat" in decoded
        assert "exp" in decoded
        assert decoded["exp"] > decoded["iat"]

    def test_generate_jwt_token_includes_rollout_id(self):
        """Ensure rollout_id is included if provided."""
        secret = "secret"
        token = generate_jwt_token(secret, rollout_id="rollout-123")
        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )

        assert decoded["rollout_id"] == "rollout-123"

    def test_generate_jwt_token_includes_extra_claims(self):
        """Ensure extra custom claims are included."""
        secret = "secret"
        token = generate_jwt_token(secret, user="test_user", env="staging")
        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )

        assert decoded["user"] == "test_user"
        assert decoded["env"] == "staging"

    def test_generate_jwt_token_expiration_respects_custom_value(self):
        """Ensure custom expiration_seconds is respected."""
        secret = "secret"
        token = generate_jwt_token(secret, expiration_seconds=60)
        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )

        assert (
            abs(decoded["exp"] - decoded["iat"] - 60) <= 1
        )  # small clock drift margin


class TestGetAuthHeaders:
    """Tests for get_auth_headers function."""

    def test_get_auth_headers_contains_bearer_prefix(self):
        """Ensure Authorization header has proper Bearer format."""
        headers = get_auth_headers("secret", rollout_id="r1")
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_get_auth_headers_token_decodes_correctly(self):
        """Ensure token inside header is valid and decodable."""
        secret = "secret"
        headers = get_auth_headers(secret, rollout_id="rollout-xyz")
        token = headers["Authorization"].split("Bearer ")[1]

        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )
        assert decoded["rollout_id"] == "rollout-xyz"

    def test_get_auth_headers_includes_extra_claims(self):
        """Ensure extra claims propagate correctly through get_auth_headers."""
        secret = "secret"
        headers = get_auth_headers(secret, env="prod", user="alice")
        token = headers["Authorization"].split("Bearer ")[1]

        decoded = jwt.decode(
            token, secret, algorithms=["HS256"], audience="mcp-proxy-server"
        )
        assert decoded["env"] == "prod"
        assert decoded["user"] == "alice"
