"""
Demo MCP server for calculator-style workflow.

Tools:
- hello_world: stateless sanity check
- define_variable / get_variable: in-memory calculator variables
- evaluate: arithmetic using stored variables
- append_log / read_log: workspace file I/O
- allocate_memory: stress test
"""

from pathlib import Path
from typing import Dict
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# ----------------------------------------------------------------------
# MCP server setup
# ----------------------------------------------------------------------

mcp = FastMCP("demo-calculator-server")

# In-memory state
_variables: Dict[str, float] = {}

# Memory stress
_leaked_memory = []

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------


@mcp.tool()
async def hello_world(name: str) -> str:
    """Simple stateless greeting."""
    return f"Hello, {name}!"


def is_valid_var_name(name: str) -> bool:
    """Check if a variable name is valid: start with letter/_ and contain only letters, digits, or _"""
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


@mcp.tool()
async def define_variable(name: str, value: float) -> str:
    """Store a named variable in memory."""
    if not is_valid_var_name(name):
        raise ToolError(f"Invalid variable name: '{name}'")
    _variables[name] = value
    return f"Variable '{name}' set to {value}"


@mcp.tool()
async def get_variable(name: str) -> str:
    """Retrieve a variable from memory."""
    if not is_valid_var_name(name):
        raise ToolError(f"Invalid variable name: '{name}'")
    if name not in _variables:
        raise ToolError(f"Variable '{name}' not defined")
    return str(_variables[name])


@mcp.tool()
async def evaluate(expression: str) -> str:
    """
    Evaluate arithmetic expression using stored variables.

    Only numbers and defined variable names are allowed.
    """
    allowed_names = {k: v for k, v in _variables.items()}
    allowed_chars = "0123456789+-*/()., "

    # Split expression manually into potential identifiers and other tokens
    token = ""
    for c in expression + " ":  # add space to flush last token
        if c.isalnum() or c == "_":
            token += c
        else:
            if token:
                if token[0].isalpha() or token[0] == "_":
                    if token not in allowed_names:
                        raise ToolError(f"Undefined variable: '{token}'")
                    if not is_valid_var_name(token):
                        raise ToolError(f"Invalid variable name: '{token}'")
                # numeric token is implicitly allowed
                token = ""
            if c not in allowed_chars and not c.isspace():
                raise ToolError(f"Invalid character in expression: '{c}'")

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        raise ToolError(f"Evaluation error: {str(e)}")


@mcp.tool()
async def append_log(filename: str, message: str) -> str:
    """Append a message to a workspace file."""
    file_path = Path(filename)
    with open(file_path, "a") as f:
        f.write(message + "\n")
    return f"Appended message to {filename}"


@mcp.tool()
async def read_log(filename: str) -> str:
    """Read the content of a workspace file."""
    file_path = Path(filename)
    if not file_path.exists():
        raise ToolError(f"File '{filename}' not found")
    return file_path.read_text()


@mcp.tool()
async def allocate_memory(megabytes: int) -> str:
    """Allocate memory to simulate stress / OOM."""
    global _leaked_memory
    size = megabytes * 1024 * 1024
    _leaked_memory.append(bytearray(size))
    return f"Leaked {megabytes} MB (total allocations: {len(_leaked_memory)})"


if __name__ == "__main__":
    # Run the server
    mcp.run()
