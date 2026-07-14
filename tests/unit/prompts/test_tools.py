from typing import Any

from benchmax.envs import Tool
from benchmax.prompts.tools import parse_hermes_tool_call, render_tools_prompt


def _tool(name: str) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_parse_hermes_tool_calls():
    text = """
    <tool_call>{"name": "one", "arguments": {"x": 1}}</tool_call>
    <tool_call>{"name": "two", "arguments": {"y": 2}}</tool_call>
    """
    result: list[dict[str, Any]] = parse_hermes_tool_call(text)

    assert [call["name"] for call in result] == ["one", "two"]
    assert parse_hermes_tool_call("") == []


def test_render_tools_prompt_uses_openai_tools_without_conversion():
    tools = [_tool("one"), _tool("two")]
    rendered = render_tools_prompt(tools, system_message="System")

    assert "System" in rendered
    assert '"name":"one"' in rendered
    assert '"name":"two"' in rendered
    assert "<tools>" in rendered
    assert "<tool_call>" in rendered


def test_render_tools_prompt_without_tools_returns_system_message():
    assert render_tools_prompt([], system_message="System") == "System"
