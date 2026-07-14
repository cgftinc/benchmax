import json
from typing import Dict, List

from benchmax.envs import Tool


def parse_hermes_tool_call(text: str) -> List[Dict[str, str]]:
    """
    Parse a tool call from Hermes XML format.
    Example:
        <tool_call>
            {"name": "get_weather", "arguments": {"location": "New York"}}
        </tool_call>
        <tool_call>
            {"name": "get_weather", "arguments": {"location": "New York"}}
        </tool_call>
    """
    import re
    import json

    # Match all tool call XML tags and extract the JSON content
    matches = re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    tool_calls = []

    for match in matches:
        tool_call_json = match.group(1).strip()
        try:
            tool_calls.append(json.loads(tool_call_json))
        except json.JSONDecodeError:
            return []

    return tool_calls if tool_calls else []


def render_tools_prompt(tool_definitions: List[Tool], system_message: str = "") -> str:
    """
    Build the prompt block that advertises the available function tools to the model.

    Parameters
    ----------
    tool_schema : list[dict]
        A list of tool descriptors in the OpenAI Tools / function-calling format.
    system_message : str, optional
        The system message that will be placed at the top of the prompt
        (defaults to the Qwen assistant greeting).

    Returns
    -------
    str
        A fully-rendered prompt string with system message and tool information.
    """
    if not tool_definitions:
        return system_message

    # Header
    lines = [
        system_message,
        "",
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]

    # One line-per-tool JSON dump (compact, no extra spaces)
    for tool in tool_definitions:
        lines.append(json.dumps(tool, separators=(",", ":")))

    lines.extend(
        [
            "</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
        ]
    )

    return "\n".join(lines)
