from typing import Dict, Any, List
from benchmax.envs.base_env import ToolDefinition
from benchmax.prompts.tools import (
    mcp2openai,
    parse_hermes_tool_call,
    render_tools_prompt,
)


def test_mcp2openai():
    # Test basic conversion
    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
    )
    result = mcp2openai(tool_def)

    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"
    assert result["function"]["description"] == "A test tool"
    assert result["function"]["parameters"] == {
        "type": "object",
        "properties": {"arg1": {"type": "string"}},
        "required": [],
    }
    assert result["function"]["strict"] is False

    # Test with empty input schema
    tool_def_no_schema = ToolDefinition(
        name="empty_tool", description="Tool with no schema", input_schema=None
    )
    result_no_schema = mcp2openai(tool_def_no_schema)
    assert result_no_schema["function"]["parameters"] == {"required": []}


def test_parse_hermes_tool_call():
    # Test single tool call
    single_call = """<tool_call>{"name": "get_weather", "arguments": {"location": "New York"}}</tool_call>"""
    result: List[Dict[str, Any]] = parse_hermes_tool_call(single_call)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"]["location"] == "New York"

    # Test multiple tool calls
    multiple_calls = """
    <tool_call>{"name": "tool1", "arguments": {"arg1": "value1"}}</tool_call>
    <tool_call>{"name": "tool2", "arguments": {"arg2": "value2"}}</tool_call>
    """
    result: List[Dict[str, Any]] = parse_hermes_tool_call(multiple_calls)
    assert len(result) == 2
    assert result[0]["name"] == "tool1"
    assert result[1]["name"] == "tool2"

    # Test empty string
    assert parse_hermes_tool_call("") == []


def test_render_tools_prompt():
    # Test with empty tool list
    assert render_tools_prompt([]) == ""

    # Test with single tool
    tool_def = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
    )
    result = render_tools_prompt([tool_def], system_message="Test System Message")

    assert "Test System Message" in result
    assert "# Tools" in result
    assert "<tools>" in result
    assert "</tools>" in result
    assert "test_tool" in result
    assert "<tool_call>" in result
    assert "</tool_call>" in result

    # Test with multiple tools
    tool_def2 = ToolDefinition(
        name="another_tool",
        description="Another test tool",
        input_schema={"type": "object", "properties": {"arg2": {"type": "number"}}},
    )
    result_multiple = render_tools_prompt([tool_def, tool_def2])
    assert "test_tool" in result_multiple
    assert "another_tool" in result_multiple
