"""
Utility functions for MCP environment infrastructure.
"""

import jwt
import time
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional, Dict, List, Any
import aiohttp
try:
    from fastmcp import Client
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "fastmcp is required for MCP environments. Install with: pip install 'benchmax[mcp]'"
    ) from e
from mcp import Tool

from benchmax.envs.types import ToolDefinition


def convert_tool_definitions(
    tools: List[Tool], allowed_tools: Optional[List[str]]
) -> List[ToolDefinition]:
    """
    Convert MCP Tool objects to ToolDefinition dataclass.

    Args:
        tools: List of MCP Tool objects.
        allowed_tools: Optional whitelist of tool names. If provided,
                      only tools in this list are included.

    Returns:
        List of ToolDefinition objects.
    """
    tool_definitions = [
        ToolDefinition(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
        )
        for tool in tools
    ]

    if not allowed_tools:
        return tool_definitions

    return [tool for tool in tool_definitions if tool.name in allowed_tools]


def generate_jwt_token(
    api_secret: str,
    rollout_id: Optional[str] = None,
    expiration_seconds: int = 300,
    **extra_claims: Any,
) -> str:
    """
    Generate a JWT token with standard and custom claims.

    Args:
        api_secret: Shared secret for signing (HS256).
        rollout_id: Optional rollout ID to include in claims.
        expiration_seconds: Token validity duration (default: 5 minutes).
        **extra_claims: Additional custom claims to include.

    Returns:
        JWT token string.
    """
    current_time = int(time.time())

    payload = {
        # Standard claims
        "iss": "mcp-client",
        "aud": "mcp-proxy-server",
        "iat": current_time,
        "exp": current_time + expiration_seconds,
        # Custom claims
        **extra_claims,
    }

    # Add rollout_id if provided
    if rollout_id:
        payload["rollout_id"] = rollout_id

    # Sign with HS256
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    return token


def get_auth_headers(
    api_secret: str, rollout_id: Optional[str] = None, **extra_claims: Any
) -> Dict[str, str]:
    """
    Generate Authorization header with JWT token.

    Args:
        api_secret: Shared secret for signing.
        rollout_id: Optional rollout ID to include in claims.
        **extra_claims: Additional custom claims.

    Returns:
        Headers dict with Authorization Bearer token.
    """
    token = generate_jwt_token(api_secret, rollout_id, **extra_claims)
    return {"Authorization": f"Bearer {token}"}


async def upload_form(
    http_session: aiohttp.ClientSession,
    upload_url: str,
    api_secret: str,
    file_bytes: bytes,
    filename: str,
    rollout_id: Optional[str] = None,
    content_type: str = "application/octet-stream",
) -> None:
    """
    Upload a file or content to a remote URL using multipart form with JWT auth.

    Args:
        http_session: aiohttp client session.
        upload_url: URL to upload to.
        api_secret: Shared secret for JWT signing.
        file_bytes: File content as bytes.
        filename: Name for the uploaded file.
        rollout_id: Optional rollout ID for JWT claims.
        content_type: MIME type of the content.

    Raises:
        RuntimeError: If upload fails.
    """
    # Generate JWT token with rollout_id
    headers = get_auth_headers(api_secret, rollout_id)

    # Create multipart form data
    data = aiohttp.FormData()
    data.add_field("file", file_bytes, filename=filename, content_type=content_type)

    async with http_session.post(upload_url, headers=headers, data=data) as response:
        if response.status == 200:
            return
        error_text = await response.text()
        raise RuntimeError(f"Upload failed: {response.status} - {error_text}")


async def download_file(
    http_session: aiohttp.ClientSession,
    download_url: str,
    api_secret: str,
    params: Dict[str, str],
    dst_path: Path,
    rollout_id: Optional[str] = None,
) -> None:
    """
    Download a file from a remote URL and save it locally with JWT auth.

    Args:
        http_session: aiohttp client session.
        download_url: URL to download from.
        api_secret: Shared secret for JWT signing.
        params: Query parameters.
        dst_path: Local path to save the downloaded file.
        rollout_id: Optional rollout ID for JWT claims.

    Raises:
        RuntimeError: If download fails.
    """
    # Generate JWT token with rollout_id
    headers = get_auth_headers(api_secret, rollout_id)

    async with http_session.get(
        download_url, headers=headers, params=params
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            raise RuntimeError(f"Download failed: {response.status} - {error_text}")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "wb") as f:
            async for chunk in response.content.iter_chunked(8192):
                f.write(chunk)


async def _safe_session_runner(self):
    """
    Patched version of FastMCPClient._session_runner that catches exceptions.

    This prevents crashes when servers disconnect or restart, logging errors
    instead of propagating them.
    """
    try:
        async with AsyncExitStack() as stack:
            try:
                await stack.enter_async_context(self._context_manager())
                self._session_state.ready_event.set()
                await self._session_state.stop_event.wait()
            except (aiohttp.ClientError, asyncio.CancelledError) as e:
                # Common expected errors when server disconnects or restarts
                print(f"[INFO] Client session ended: {type(e).__name__}: {e}")
            except Exception as e:
                # Log unexpected errors
                print(f"[WARN] Client session crashed: {e}")
            finally:
                self._session_state.ready_event.set()
    except Exception as e:
        # Catch outer-level async exit errors
        print(f"[WARN] Session runner outer error: {e}")


def apply_fastmcp_patch():
    """
    Apply the monkey patch to FastMCP Client.

    Call this once at module import to enable graceful session handling.
    """
    Client._session_runner = _safe_session_runner
