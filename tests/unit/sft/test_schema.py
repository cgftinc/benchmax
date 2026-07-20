"""Tests for per-row SFT schema validation."""

from __future__ import annotations

from benchmax.sft.schema import validate_row

_VALID_ROW = {
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
}


class TestValidRows:
    def test_minimal_valid_row(self):
        assert validate_row(_VALID_ROW) == []

    def test_valid_row_with_tools_and_weight(self):
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo", "weight": 1},
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
        assert validate_row(row) == []

    def test_valid_row_with_tool_call_and_tool_response(self):
        row = {
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        }
        assert validate_row(row) == []

    def test_valid_multimodal_row(self):
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "text", "text": "what is this"},
                    ],
                },
                {"role": "assistant", "content": "a cat"},
            ]
        }
        assert validate_row(row) == []

    def test_multimodal_image_url_https(self):
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                },
                {"role": "assistant", "content": "a cat"},
            ]
        }
        assert validate_row(row) == []


class TestRowShapeErrors:
    def test_not_a_dict(self):
        assert validate_row([1, 2, 3]) != []

    def test_missing_messages(self):
        errors = validate_row({})
        assert any("'messages' is required" in e for e in errors)

    def test_messages_not_a_list(self):
        errors = validate_row({"messages": "oops"})
        assert any("non-empty list" in e for e in errors)

    def test_messages_empty_list(self):
        errors = validate_row({"messages": []})
        assert any("non-empty list" in e for e in errors)

    def test_tools_not_a_list(self):
        row = {**_VALID_ROW, "tools": {"not": "a list"}}
        errors = validate_row(row)
        assert any("'tools' must be a list" in e for e in errors)


class TestRoleErrors:
    def test_invalid_role(self):
        row = {"messages": [{"role": "narrator", "content": "hi"}]}
        errors = validate_row(row)
        assert any("role must be one of" in e for e in errors)

    def test_tool_role_missing_tool_call_id(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "result"},
            ]
        }
        errors = validate_row(row)
        assert any("tool_call_id" in e for e in errors)


class TestWeightErrors:
    def test_invalid_weight_value(self):
        row = {"messages": [{"role": "assistant", "content": "x", "weight": 0.5}]}
        errors = validate_row(row)
        assert any("weight must be 0 or 1" in e for e in errors)

    def test_all_assistant_turns_masked_is_error(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo", "weight": 0},
            ]
        }
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_no_assistant_message_is_error(self):
        row = {"messages": [{"role": "user", "content": "hi"}]}
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_default_weight_counts_as_trained(self):
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
        assert validate_row(row) == []

    def test_one_trained_one_masked_is_valid(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a", "weight": 0},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "b", "weight": 1},
            ]
        }
        assert validate_row(row) == []


class TestToolCallErrors:
    def test_tool_call_wrong_type(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "oops", "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("type must be 'function'" in e for e in errors)

    def test_tool_call_missing_function(self):
        row = {
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]}
            ]
        }
        errors = validate_row(row)
        assert any("function must be an object" in e for e in errors)

    def test_tool_call_empty_name(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "", "arguments": "{}"}}
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("non-empty string" in e for e in errors)

    def test_tool_call_arguments_not_string(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "f", "arguments": {}}}
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("JSON-encoded string" in e for e in errors)

    def test_tool_call_arguments_invalid_json(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("must be valid JSON" in e for e in errors)

    def test_tool_calls_not_a_list(self):
        row = {"messages": [{"role": "assistant", "content": None, "tool_calls": "oops"}]}
        errors = validate_row(row)
        assert any("tool_calls must be a list" in e for e in errors)


class TestContentPartErrors:
    def test_content_not_str_or_list(self):
        row = {"messages": [{"role": "user", "content": 5}]}
        errors = validate_row(row)
        assert any("string or a list of content parts" in e for e in errors)

    def test_content_part_not_dict(self):
        row = {"messages": [{"role": "user", "content": ["oops"]}]}
        errors = validate_row(row)
        assert any("content[0] must be an object" in e for e in errors)

    def test_content_part_unknown_type(self):
        row = {"messages": [{"role": "user", "content": [{"type": "audio"}]}]}
        errors = validate_row(row)
        assert any("type must be one of" in e for e in errors)

    def test_text_part_missing_text(self):
        row = {"messages": [{"role": "user", "content": [{"type": "text"}]}]}
        errors = validate_row(row)
        assert any("must have a string 'text'" in e for e in errors)

    def test_image_part_missing_url(self):
        row = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
        errors = validate_row(row)
        assert any("image_url.url" in e for e in errors)

    def test_image_part_bad_scheme(self):
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}],
                }
            ]
        }
        errors = validate_row(row)
        assert any("image_url.url" in e for e in errors)


class TestJsonSerializability:
    def test_non_serializable_tool_spec_is_error(self):
        row = {**_VALID_ROW, "tools": [{"unserializable": {1, 2, 3}}]}
        errors = validate_row(row)
        assert any("not JSON-serializable" in e for e in errors)
