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

    def test_boolean_weight_rejected(self):
        # isinstance(True, int) is True and True == 1, so a naive
        # membership/equality check would silently accept a bool.
        row = {"messages": [{"role": "assistant", "content": "x", "weight": True}]}
        errors = validate_row(row)
        assert any("weight must be 0 or 1" in e for e in errors)

    def test_float_weight_rejected(self):
        # 1.0 == 1, so this must be rejected by an exact int type check.
        row = {"messages": [{"role": "assistant", "content": "x", "weight": 1.0}]}
        errors = validate_row(row)
        assert any("weight must be 0 or 1" in e for e in errors)

    def test_weight_on_user_message_is_error(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi", "weight": 1},
                {"role": "assistant", "content": "yo"},
            ]
        }
        errors = validate_row(row)
        assert any("weight is only valid on assistant messages" in e for e in errors)

    def test_weight_on_system_message_is_error(self):
        row = {
            "messages": [
                {"role": "system", "content": "sys", "weight": 0},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        }
        errors = validate_row(row)
        assert any("weight is only valid on assistant messages" in e for e in errors)

    def test_assistant_with_no_content_and_no_tool_calls_not_trained(self):
        row = {"messages": [{"role": "assistant"}]}
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_empty_string_content_not_trained(self):
        row = {"messages": [{"role": "assistant", "content": ""}]}
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_empty_content_list_not_trained(self):
        row = {"messages": [{"role": "assistant", "content": []}]}
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_empty_text_part_not_trained(self):
        # a content-part list whose only part carries "" is semantically the
        # same as bare content: "" — must not count as a trained turn either.
        row = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}]}
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_multiple_empty_text_parts_not_trained(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}, {"type": "text", "text": ""}],
                }
            ]
        }
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_non_empty_text_part_is_trained(self):
        row = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]}
        assert validate_row(row) == []

    def test_empty_text_part_with_non_canonical_content_key_not_trained(self):
        # a `content` key on a text part isn't part of the canonical
        # ChatCompletionContentPartTextParam shape (text-only) — must not be
        # used as a fallback to smuggle a trained turn past an empty `text`.
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "", "content": "fallback"}],
                }
            ]
        }
        errors = validate_row(row)
        assert any("no trained assistant turn" in e for e in errors)

    def test_assistant_with_only_image_part_is_trained(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ],
                }
            ]
        }
        assert validate_row(row) == []

    def test_assistant_with_empty_text_plus_valid_image_is_trained(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ]
        }
        assert validate_row(row) == []

    def test_assistant_with_valid_tool_calls_and_no_content_is_trained(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ]
        }
        assert validate_row(row) == []

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

    def test_tool_call_arguments_deeply_nested_json_is_reported_not_raised(self):
        # deeply nested arguments overflow json's parser stack (RecursionError);
        # validation must report it as invalid JSON, never crash with a traceback
        deep = "[" * 20000 + "]" * 20000
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "f", "arguments": deep}}
                    ],
                }
            ]
        }
        errors = validate_row(row)  # must not raise RecursionError
        assert any("must be valid JSON" in e for e in errors)

    def test_tool_calls_not_a_list(self):
        row = {"messages": [{"role": "assistant", "content": None, "tool_calls": "oops"}]}
        errors = validate_row(row)
        assert any("tool_calls must be a list" in e for e in errors)

    def test_tool_call_missing_id(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}],
                }
            ]
        }
        errors = validate_row(row)
        assert any("tool_calls[0].id" in e for e in errors)

    def test_tool_call_empty_id(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("tool_calls[0].id" in e for e in errors)

    def test_tool_response_empty_tool_call_id_is_error(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "tool_call_id": "", "content": "result"},
            ]
        }
        errors = validate_row(row)
        assert any("tool_call_id" in e for e in errors)

    def test_tool_call_arguments_with_nan_is_rejected(self):
        # json.loads accepts the non-standard NaN/Infinity/-Infinity tokens
        # by default; a tool-call's arguments string must reject them too,
        # not just the top-level row parse.
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"score": NaN}'},
                        }
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("must be valid JSON" in e for e in errors)

    def test_tool_call_arguments_with_infinity_is_rejected(self):
        row = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"score": Infinity}'},
                        }
                    ],
                }
            ]
        }
        errors = validate_row(row)
        assert any("must be valid JSON" in e for e in errors)


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

    def test_serialization_check_runs_even_with_structural_errors(self):
        # Previously the JSON-serializability check was skipped once any
        # other structural error was found; both must surface now.
        row = {
            "messages": [{"role": "not-a-real-role", "content": "hi"}],
            "extra_metadata": {1, 2, 3},  # a set is not JSON-serializable
        }
        errors = validate_row(row)
        assert any("role must be one of" in e for e in errors)
        assert any("not JSON-serializable" in e for e in errors)

    def test_lone_surrogate_content_is_rejected(self):
        # ensure_ascii=True (json.dumps' default) never attempts the UTF-8
        # encode step, so a naive check would pass this row right up until
        # canonical_jsonl's ensure_ascii=False encode blew up on it.
        row = {"messages": [{"role": "assistant", "content": "before\ud800after"}]}
        errors = validate_row(row)
        assert any("not JSON-serializable" in e for e in errors)

    def test_ordinary_non_ascii_content_is_fine(self):
        # the fix must not start rejecting legitimate non-ASCII text.
        row = {"messages": [{"role": "assistant", "content": "héllo — 日本語"}]}
        assert validate_row(row) == []


class TestUnexpectedTopLevelFields:
    def test_unexpected_field_is_a_clear_issue(self):
        row = {**_VALID_ROW, "extra_metadata": {"foo": "bar"}}
        errors = validate_row(row)
        assert any("unexpected field" in e and "extra_metadata" in e for e in errors)
        # the rest of the row is otherwise valid — no unrelated errors
        assert len(errors) == 1


class TestCrashSafety:
    """Malformed JSON-shaped values (lists/dicts where a string/int is
    expected) must produce a clean issue, never raise — set-membership and
    equality checks against user-supplied values can otherwise TypeError on
    unhashable types."""

    def test_list_role_does_not_raise(self):
        errors = validate_row({"messages": [{"role": [], "content": "hi"}]})
        assert any("role must be one of" in e for e in errors)

    def test_dict_role_does_not_raise(self):
        errors = validate_row({"messages": [{"role": {}, "content": "hi"}]})
        assert any("role must be one of" in e for e in errors)

    def test_dict_weight_does_not_raise(self):
        errors = validate_row({"messages": [{"role": "assistant", "content": "hi", "weight": {}}]})
        assert any("weight must be 0 or 1" in e for e in errors)

    def test_list_weight_does_not_raise(self):
        errors = validate_row({"messages": [{"role": "assistant", "content": "hi", "weight": []}]})
        assert any("weight must be 0 or 1" in e for e in errors)

    def test_list_content_part_type_does_not_raise(self):
        row = {"messages": [{"role": "user", "content": [{"type": []}]}]}
        errors = validate_row(row)
        assert any("type must be one of" in e for e in errors)

    def test_dict_content_part_type_does_not_raise(self):
        row = {"messages": [{"role": "user", "content": [{"type": {}}]}]}
        errors = validate_row(row)
        assert any("type must be one of" in e for e in errors)
