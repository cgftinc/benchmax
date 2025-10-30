"""
Reward functions for demo calculator MCP server.

Three reward types:
1. Stateless completion check
2. Tool call variable in memory check
3. Workspace log check

All reward functions will receive the same ground truth and completion.
When defining the dataset, you can determine the shape of your ground truth.

In this example, the ground truth is a dictionary of the shape:
{
    'completion': str,
    'variable': { name: str, expected_value: float },
    'log': { filename: str, expected_content: str }
}

Each reward fn extract what they need from the ground_truth dictionary.
Each of these reward shows way of computing reward using completion, calling mcp server tool,
and reaading from the workspace that the MCP is operating in.

"""

from pathlib import Path
from typing import Any, Callable, Dict, Union, Awaitable
from mcp.types import TextContent
from fastmcp import Client
from fastmcp.exceptions import ToolError

RewardFunction = Callable[..., Union[float, Awaitable[float]]]


# -------------------------------
# Reward 0: Stateless completion check
# -------------------------------
async def completion_match_reward(
    completion: str,
    ground_truth: dict,
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any,
) -> float:
    """
    Return 1.0 if completion matches ground_truth['completion'], else 0.0

    Uses: ground_truth['completion'] (str)
    """
    expected = ground_truth.get("completion", "")
    return 1.0 if completion.strip() == expected.strip() else 0.0


# -------------------------------
# Reward 1: Tool call variable in memory check
# -------------------------------
async def variable_in_memory_reward(
    completion: str, ground_truth: dict, mcp_client: Client, workspace: Path, **kwargs
) -> float:
    """
    Reward uses tool call to match in-memory variable value.

    Uses: ground_truth['variable'] = {"name": str, "expected_value": float}
    """
    variable_spec = ground_truth.get("variable", {})
    var_name = variable_spec.get("name")
    expected = variable_spec.get("expected_value")

    if not var_name or expected is None:
        return 0.0

    if not mcp_client or not mcp_client.is_connected():
        return 0.0

    try:
        # Call tool via MCP client
        response = await mcp_client.call_tool("get_variable", {"name": var_name})

        # Extract text from TextContent objects
        text_contents = []
        for content in response.content:
            if isinstance(content, TextContent):
                text_contents.append(content.text)

        combined_text = "\n".join(text_contents).strip()
        value = float(combined_text)

        return 1.0 if value == expected else 0.0

    except ToolError:
        return 0.0
    except Exception:
        return 0.0


# -------------------------------
# Reward 2: Workspace log check
# -------------------------------
async def log_in_workspace_reward(
    completion: str,
    ground_truth: dict,
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any,
) -> float:
    """
    Reward based on workspace file content.

    Uses: ground_truth['log'] = {"filename": str, "expected_content": str}
    """
    log_spec = ground_truth.get("log", {})
    filename = log_spec.get("filename")
    expected = log_spec.get("expected_content", "")

    if not filename:
        return 0.0

    file_path = Path(workspace) / filename
    if not file_path.exists():
        return 0.0

    content = file_path.read_text().strip()
    return 1.0 if content == expected.strip() else 0.0


# -------------------------------
# Export reward functions
# -------------------------------
reward_functions: Dict[str, RewardFunction] = {
    "completion": completion_match_reward,
    "variable": variable_in_memory_reward,
    "log": log_in_workspace_reward,
}
