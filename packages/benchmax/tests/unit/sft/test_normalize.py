"""Tests for legacy SFT row shape normalization."""

from __future__ import annotations

import json

from benchmax.sft.normalize import normalize_row


class TestCanonicalPassthrough:
    def test_messages_and_tools_preserved_verbatim(self):
        row = {
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        }
        assert normalize_row(row) == row

    def test_weight_preserved(self):
        row = {"messages": [{"role": "assistant", "content": "x", "weight": 0}]}
        assert normalize_row(row)["messages"][0]["weight"] == 0

    def test_multimodal_content_preserved_untouched(self):
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "what is this"},
        ]
        row = {"messages": [{"role": "user", "content": content}]}
        assert normalize_row(row)["messages"][0]["content"] == content

    def test_unrecognized_shape_preserves_every_field(self):
        row = {"tools": [{"type": "function", "function": {"name": "f"}}], "random_field": 1}
        result = normalize_row(row)
        assert "messages" not in result
        assert result["tools"] == row["tools"]
        assert result["random_field"] == 1

    def test_unexpected_top_level_field_survives_when_already_canonical(self):
        row = {
            "messages": [{"role": "assistant", "content": "hi"}],
            "extra_metadata": {"foo": "bar"},
        }
        result = normalize_row(row)
        assert result["extra_metadata"] == {"foo": "bar"}

    def test_unexpected_top_level_field_survives_split_format(self):
        row = {
            "prompt_messages": [{"role": "user", "content": "hi"}],
            "completion": "yo",
            "extra_metadata": {"foo": "bar"},
        }
        result = normalize_row(row)
        assert result["extra_metadata"] == {"foo": "bar"}
        # the legacy keys actually consumed into `messages` are gone, but
        # nothing else is
        assert "prompt_messages" not in result
        assert "completion" not in result


class TestSplitFormat:
    def test_prompt_messages_plus_completion_messages(self):
        row = {
            "prompt_messages": [{"role": "user", "content": "hi"}],
            "completion_messages": [{"role": "assistant", "content": "yo"}],
        }
        result = normalize_row(row)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_bare_prompt_string_with_completion_messages(self):
        row = {
            "prompt": "hi",
            "completion_messages": [{"role": "assistant", "content": "yo"}],
        }
        result = normalize_row(row)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_bare_prompt_and_bare_completion_strings(self):
        row = {"prompt": "hi", "completion": "yo"}
        result = normalize_row(row)
        assert result["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]

    def test_prompt_messages_single_dict_wrapped_in_list(self):
        row = {
            "prompt_messages": {"role": "user", "content": "hi"},
            "completion": "yo",
        }
        result = normalize_row(row)
        assert result["messages"][0] == {"role": "user", "content": "hi"}


class TestFlatToolCallFormat:
    def test_flat_tool_call_normalized_to_nested(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "cat"}}],
                }
            ]
        }
        result = normalize_row(row)
        tc = result["messages"][0]["tool_calls"][0]
        assert tc == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"q": "cat"})},
        }

    def test_nested_tool_call_arguments_dict_stringified(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"q": "cat"}},
                        }
                    ],
                }
            ]
        }
        result = normalize_row(row)
        arguments = result["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, str)
        assert json.loads(arguments) == {"q": "cat"}

    def test_nested_tool_call_string_arguments_untouched(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "cat"}'},
                        }
                    ],
                }
            ]
        }
        result = normalize_row(row)
        assert result["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"q": "cat"}'

    def test_split_format_with_flat_tool_calls_in_completion(self):
        row = {
            "prompt_messages": [{"role": "user", "content": "find a cat"}],
            "completion_messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": "{}"}],
                }
            ],
        }
        result = normalize_row(row)
        tc = result["messages"][1]["tool_calls"][0]
        assert tc["function"]["name"] == "lookup"
        assert tc["type"] == "function"

    def test_flat_tool_call_preserves_extra_fields_on_the_call(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "name": "lookup", "arguments": "{}", "index": 0}
                    ],
                }
            ]
        }
        result = normalize_row(row)
        tc = result["messages"][0]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["id"] == "call_1"

    def test_nested_tool_call_preserves_full_fidelity_extra_fields(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "index": 0,
                            "function": {
                                "name": "lookup",
                                "arguments": {"q": "cat"},
                                "description": "custom",
                            },
                        }
                    ],
                }
            ]
        }
        result = normalize_row(row)
        tc = result["messages"][0]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["function"]["name"] == "lookup"
        assert tc["function"]["description"] == "custom"
        assert json.loads(tc["function"]["arguments"]) == {"q": "cat"}
