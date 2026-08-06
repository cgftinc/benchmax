"""Behavior tests for `benchmax.sft.SftDataset` construction and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmax.sft import SFT_DATASET_FORMAT, SftDataset, SftDatasetError, SftIssue


def user(text: str = "q") -> dict[str, object]:
    return {"role": "user", "content": text}


def assistant(text: str | None = "a", **extra: object) -> dict[str, object]:
    return {"role": "assistant", "content": text, **extra}


def tool_call(call_id: str = "c1", name: str = "f", arguments: str = "{}") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_result(call_id: str = "c1", content: str = "ok") -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def tool_def(name: str = "f") -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def valid_row() -> dict[str, object]:
    return {"messages": [user(), assistant()]}


def issues_of(*rows: object) -> tuple[SftIssue, ...]:
    with pytest.raises(SftDatasetError) as excinfo:
        SftDataset.from_rows(list(rows))
    return excinfo.value.issues


def assert_single_issue(rows: list[object], location: str, fragment: str) -> SftIssue:
    issues = issues_of(*rows)
    assert len(issues) == 1, [issue.describe() for issue in issues]
    assert issues[0].location == location
    assert fragment in issues[0].message
    return issues[0]


class TestConstructionBasics:
    def test_format_identifier(self) -> None:
        assert SFT_DATASET_FORMAT == "benchmax-sft-v1"

    def test_direct_construction_rejected(self) -> None:
        with pytest.raises(TypeError, match="from_jsonl"):
            SftDataset((), ())

    def test_len_iter_and_rows(self) -> None:
        dataset = SftDataset.from_rows([valid_row(), valid_row()])
        assert len(dataset) == 2
        assert list(dataset) == list(dataset.rows)

    def test_empty_rows_rejected(self) -> None:
        issues = issues_of()
        assert [issue.as_dict() for issue in issues] == [
            {"line": None, "location": "$", "message": "dataset contains no rows", "row": None}
        ]

    def test_stored_rows_decoupled_from_caller_objects(self) -> None:
        source = valid_row()
        dataset = SftDataset.from_rows([source])
        source["messages"].append({"role": "bogus"})  # type: ignore[union-attr]
        assert len(dataset.rows[0]["messages"]) == 2  # type: ignore[arg-type]


class TestCanonicalSerialization:
    def test_sorted_keys_compact_separators_trailing_newline(self) -> None:
        dataset = SftDataset.from_rows(
            [{"metadata": {"b": 1, "a": 2}, "messages": [user(), assistant()]}]
        )
        payload = dataset.to_jsonl_bytes()
        assert payload.endswith(b"\n") and not payload.startswith(b"\xef\xbb\xbf")
        assert payload.decode("utf-8") == (
            '{"messages":[{"content":"q","role":"user"},{"content":"a","role":"assistant"}],'
            '"metadata":{"a":2,"b":1}}\n'
        )

    def test_equivalent_inputs_share_bytes(self) -> None:
        left = SftDataset.from_rows([{"messages": [user(), assistant()], "tools": []}])
        right = SftDataset.from_rows([{"tools": [], "messages": [user(), assistant()]}])
        assert left.to_jsonl_bytes() == right.to_jsonl_bytes()

    def test_non_ascii_not_escaped(self) -> None:
        dataset = SftDataset.from_rows([{"messages": [user("café ✓"), assistant()]}])
        assert "café ✓".encode() in dataset.to_jsonl_bytes()

    def test_row_order_preserved(self) -> None:
        rows = [{"messages": [user(f"q{i}"), assistant()]} for i in range(5)]
        dataset = SftDataset.from_rows(rows)
        contents = [r["messages"][0]["content"] for r in dataset.rows]  # type: ignore[index]
        assert contents == [f"q{i}" for i in range(5)]


class TestJsonlParsing:
    def write(self, tmp_path: Path, data: bytes) -> Path:
        path = tmp_path / "train.jsonl"
        path.write_bytes(data)
        return path

    def test_blank_lines_ignored_and_line_numbers_preserved(self, tmp_path: Path) -> None:
        data = b"\n \t\n" + json.dumps(valid_row()).encode() + b"\n\n{bad\n"
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, data))
        (issue,) = excinfo.value.issues
        assert issue.line == 5 and issue.row == 1
        assert "invalid JSON" in issue.message

    def test_file_without_trailing_newline(self, tmp_path: Path) -> None:
        dataset = SftDataset.from_jsonl(self.write(tmp_path, json.dumps(valid_row()).encode()))
        assert len(dataset) == 1

    def test_invalid_utf8_line(self, tmp_path: Path) -> None:
        data = json.dumps(valid_row()).encode() + b"\n\xff\xfe{}\n"
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, data))
        (issue,) = excinfo.value.issues
        assert issue.line == 2 and "not valid UTF-8" in issue.message

    def test_bom_rejected(self, tmp_path: Path) -> None:
        data = b"\xef\xbb\xbf" + json.dumps(valid_row()).encode() + b"\n"
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, data))
        assert "BOM" in excinfo.value.issues[0].message

    def test_non_object_lines_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, b'[1,2]\n"text"\n42\n'))
        assert len(excinfo.value.issues) == 3
        assert all("row must be a JSON object" in i.message for i in excinfo.value.issues)

    def test_whitespace_only_file_is_empty_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, b"\n  \n\t\n"))
        assert "dataset contains no rows" in excinfo.value.issues[0].message

    def test_issue_ordering_follows_lines(self, tmp_path: Path) -> None:
        bad = {"messages": [user(), assistant(weight=5)]}
        data = (json.dumps(bad) + "\n") * 3
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(self.write(tmp_path, data.encode()))
        lines = [issue.line for issue in excinfo.value.issues]
        assert lines == sorted(lines)


class TestStrictJsonRules:
    def test_duplicate_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.jsonl"
        path.write_bytes(b'{"messages": [], "messages": []}\n')
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_jsonl(path)
        assert "duplicate object key" in excinfo.value.issues[0].message

    def test_nan_and_infinity_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            assert_single_issue(
                [{"messages": [user(), assistant()], "metadata": {"x": value}}],
                "$.metadata.x",
                "finite",
            )

    def test_lone_surrogate_rejected(self) -> None:
        assert_single_issue(
            [{"messages": [user("bad \ud800"), assistant()]}],
            "$.messages[0].content",
            "lone surrogate",
        )

    def test_nesting_beyond_64_levels_rejected(self) -> None:
        deep: object = "leaf"
        for _ in range(70):
            deep = [deep]
        issues = issues_of({"messages": [user(), assistant()], "metadata": {"deep": deep}})
        assert any("nesting exceeds 64 levels" in issue.message for issue in issues)

    def test_nesting_at_64_levels_allowed(self) -> None:
        # Row=1, metadata=2, deep=3; 61 more lists reach exactly depth 64.
        deep: object = "leaf"
        for _ in range(62):
            deep = [deep]
        SftDataset.from_rows([{"messages": [user(), assistant()], "metadata": {"deep": deep}}])

    def test_non_json_types_rejected(self) -> None:
        assert_single_issue(
            [{"messages": [user(), assistant()], "metadata": {"x": {1, 2}}}],
            "$.metadata.x",
            "unsupported type",
        )
        assert_single_issue(
            [{"messages": [user(), assistant()], "metadata": {1: "x"}}],
            "$.metadata",
            "must be a string",
        )


class TestRowShape:
    def test_unknown_top_level_keys(self) -> None:
        issues = issues_of({"messages": [user(), assistant()], "prompt": "x", "extra": 1})
        assert issues[0].location == "$"
        assert issues[0].message == "unknown top-level key(s): extra, prompt"

    def test_messages_required(self) -> None:
        assert_single_issue([{}], "$", "messages is required")

    def test_messages_must_be_nonempty_list(self) -> None:
        assert_single_issue([{"messages": "x"}], "$.messages", "must be a list")
        assert_single_issue([{"messages": []}], "$.messages", "must not be empty")

    def test_messages_cap(self) -> None:
        many = [user() for _ in range(1024)] + [assistant()]
        issues = issues_of({"messages": many})
        assert any("maximum is 1024" in issue.message for issue in issues)
        SftDataset.from_rows([{"messages": [user() for _ in range(1023)] + [assistant()]}])

    def test_row_size_cap(self) -> None:
        big = {"messages": [user("x" * 1_100_000), assistant()]}
        assert_single_issue([big], "$", "row canonical size")


class TestMessageShapes:
    def test_invalid_role(self) -> None:
        issues = issues_of({"messages": [{"role": "narrator", "content": "x"}, assistant()]})
        assert issues[0].location == "$.messages[0].role"
        assert "role must be one of" in issues[0].message

    def test_role_crash_safety(self) -> None:
        for bad_role in ([1], {"a": 1}, None, 7):
            issues = issues_of({"messages": [{"role": bad_role, "content": "x"}, assistant()]})
            assert issues, bad_role

    def test_message_not_object(self) -> None:
        assert_single_issue([{"messages": ["hi", assistant()]}], "$.messages[0]", "JSON object")

    def test_user_content_rules(self) -> None:
        assert_single_issue(
            [{"messages": [{"role": "user"}, assistant()]}],
            "$.messages[0].content",
            "must be a string",
        )
        assert_single_issue(
            [{"messages": [{"role": "user", "content": ""}, assistant()]}],
            "$.messages[0].content",
            "must not be empty",
        )
        assert_single_issue(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                        assistant(),
                    ]
                }
            ],
            "$.messages[0].content",
            "must be a string",
        )

    def test_weight_rejected_on_non_assistant(self) -> None:
        issues = issues_of(
            {"messages": [{"role": "user", "content": "q", "weight": 1}, assistant()]}
        )
        assert "weight is only supported on assistant messages" in issues[0].message

    def test_assistant_weight_values(self) -> None:
        SftDataset.from_rows([{"messages": [user(), assistant(weight=1)]}])
        SftDataset.from_rows([{"messages": [user(), assistant(weight=0), assistant("b")]}])
        for bad in (True, False, 2, -1, 1.0, "1", None):
            issues = issues_of({"messages": [user(), assistant(weight=bad), assistant("b")]})
            assert any(i.message == "weight must be the integer 0 or 1" for i in issues), bad

    def test_assistant_needs_content_or_tool_call(self) -> None:
        for content in (None, ""):
            issues = issues_of({"messages": [user(), {"role": "assistant", "content": content}]})
            assert any("non-empty content or at least one tool call" in i.message for i in issues)
        issues = issues_of({"messages": [user(), {"role": "assistant"}]})
        assert any("non-empty content or at least one tool call" in i.message for i in issues)

    def test_assistant_null_content_with_tool_call_valid(self) -> None:
        SftDataset.from_rows(
            [
                {
                    "messages": [user(), assistant(None, tool_calls=[tool_call()]), tool_result()],
                    "tools": [tool_def()],
                }
            ]
        )

    def test_assistant_empty_tool_calls_rejected(self) -> None:
        issues = issues_of(
            {"messages": [user(), {"role": "assistant", "content": "a", "tool_calls": []}]}
        )
        assert any("non-empty list" in issue.message for issue in issues)

    def test_tool_result_shape(self) -> None:
        assert_single_issue(
            [
                {
                    "messages": [
                        user(),
                        assistant(None, tool_calls=[tool_call()]),
                        {"role": "tool", "tool_call_id": "c1"},
                        assistant("done"),
                    ],
                    "tools": [tool_def()],
                }
            ],
            "$.messages[2].content",
            "must be a string",
        )
        issues = issues_of(
            {
                "messages": [
                    user(),
                    assistant(),
                    {"role": "tool", "content": "x", "tool_call_id": ""},
                ]
            }
        )
        assert issues[0].location == "$.messages[2].tool_call_id"

    def test_unknown_message_keys(self) -> None:
        issues = issues_of({"messages": [user(), assistant(name="x")]})
        assert issues[0].message == "unknown key(s): name"


class TestToolDefinitions:
    def base(self, tools: list[object]) -> dict[str, object]:
        return {"messages": [user(), assistant()], "tools": tools}

    def test_valid_definition_without_calls(self) -> None:
        SftDataset.from_rows([self.base([tool_def()])])
        SftDataset.from_rows([self.base([])])

    def test_tools_cap_and_uniqueness(self) -> None:
        issues = issues_of(self.base([tool_def(f"f{i}") for i in range(129)]))
        assert any("maximum is 128" in issue.message for issue in issues)
        SftDataset.from_rows([self.base([tool_def(f"f{i}") for i in range(128)])])
        issues = issues_of(self.base([tool_def("f"), tool_def("f")]))
        assert any("duplicate tool definition" in issue.message for issue in issues)

    def test_definition_shape(self) -> None:
        assert_single_issue([self.base(["x"])], "$.tools[0]", "JSON object")
        issues = issues_of(
            self.base([{"type": "tool", "function": {"name": "f", "parameters": {}}}])
        )
        assert issues[0].location == "$.tools[0].type"
        issues = issues_of(self.base([{"type": "function"}]))
        assert "requires a function object" in issues[0].message
        issues = issues_of(
            self.base([{"type": "function", "function": {"name": "", "parameters": {}}}])
        )
        assert issues[0].location == "$.tools[0].function.name"
        issues = issues_of(self.base([{"type": "function", "function": {"name": "f"}}]))
        assert any("parameters is required" in issue.message for issue in issues)
        issues = issues_of(
            self.base([{"type": "function", "function": {"name": "f", "parameters": []}}])
        )
        assert issues[0].location == "$.tools[0].function.parameters"

    def test_parameters_type_must_be_object(self) -> None:
        bad = {"type": "function", "function": {"name": "f", "parameters": {"type": "array"}}}
        issues = issues_of(self.base([bad]))
        assert issues[0].location == "$.tools[0].function.parameters.type"
        ok = {"type": "function", "function": {"name": "f", "parameters": {"properties": {}}}}
        SftDataset.from_rows([self.base([ok])])

    def test_description_must_be_string(self) -> None:
        bad = {"type": "function", "function": {"name": "f", "parameters": {}, "description": 4}}
        issues = issues_of(self.base([bad]))
        assert issues[0].location == "$.tools[0].function.description"


class TestToolCallsAndLinking:
    def with_calls(self, *messages: object, tools: list[object] | None = None) -> dict[str, object]:
        return {
            "messages": [user(), *messages],
            "tools": tools if tools is not None else [tool_def()],
        }

    def test_full_flow_valid(self) -> None:
        SftDataset.from_rows(
            [
                self.with_calls(
                    assistant(None, tool_calls=[tool_call("a"), tool_call("b")]),
                    tool_result("a"),
                    tool_result("b"),
                    assistant("done"),
                )
            ]
        )

    def test_tools_required_when_calls_exist(self) -> None:
        row = {"messages": [user(), assistant(None, tool_calls=[tool_call()]), tool_result()]}
        assert_single_issue([row], "$.tools", "non-empty list when any message makes a tool call")
        row = {
            "messages": [user(), assistant(None, tool_calls=[tool_call()]), tool_result()],
            "tools": [],
        }
        assert_single_issue([row], "$.tools", "non-empty list when any message makes a tool call")

    def test_missing_result(self) -> None:
        issues = issues_of(
            self.with_calls(assistant(None, tool_calls=[tool_call()]), assistant("x"))
        )
        assert any("have no tool result" in issue.message for issue in issues)
        issues = issues_of(self.with_calls(assistant(None, tool_calls=[tool_call()])))
        assert any("have no tool result" in issue.message for issue in issues)

    def test_duplicate_result(self) -> None:
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call()]),
                tool_result(),
                tool_result(),
                assistant("x"),
            )
        )
        assert any("duplicate tool result" in issue.message for issue in issues)

    def test_out_of_order_results(self) -> None:
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call("a"), tool_call("b")]),
                tool_result("b"),
                tool_result("a"),
                assistant("x"),
            )
        )
        assert [issue.message for issue in issues] == [
            "tool result for id 'b' is out of declaration order"
        ]

    def test_late_result_after_window_closed(self) -> None:
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call()]), assistant("x"), tool_result()
            )
        )
        messages = [issue.message for issue in issues]
        assert any("have no tool result" in message for message in messages)
        assert any("after its tool-call window closed" in message for message in messages)

    def test_unknown_result_id(self) -> None:
        issues = issues_of({"messages": [user(), assistant(), tool_result("ghost")]})
        assert "unknown tool_call_id" in issues[0].message

    def test_duplicate_call_ids_across_messages(self) -> None:
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call("a")]),
                tool_result("a"),
                assistant(None, tool_calls=[tool_call("a")]),
                tool_result("a"),
                assistant("x"),
            )
        )
        assert any("duplicate tool call id" in issue.message for issue in issues)

    def test_call_shape(self) -> None:
        issues = issues_of(self.with_calls(assistant(None, tool_calls=["x"]), assistant("y")))
        assert any(issue.location == "$.messages[1].tool_calls[0]" for issue in issues)
        bad = {"id": "c1", "type": "tool_call", "function": {"name": "f", "arguments": "{}"}}
        issues = issues_of(
            self.with_calls(assistant(None, tool_calls=[bad]), tool_result(), assistant("y"))
        )
        assert any(issue.location == "$.messages[1].tool_calls[0].type" for issue in issues)
        bad = {"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        issues = issues_of(self.with_calls(assistant(None, tool_calls=[bad]), assistant("y")))
        assert any(issue.location == "$.messages[1].tool_calls[0].id" for issue in issues)

    def test_undefined_function_name(self) -> None:
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call(name="ghost")]), tool_result(), assistant("y")
            )
        )
        assert any("undefined function" in issue.message for issue in issues)

    def test_arguments_rules(self) -> None:
        for arguments, fragment in (
            (7, "JSON-encoded string"),
            ("[1]", "JSON object"),
            ("{bad", "must decode as JSON"),
            ('{"a": 1, "a": 2}', "duplicate object key"),
            ('{"a": NaN}', "non-finite"),
            ('{"a": 1e999}', "finite"),
        ):
            issues = issues_of(
                self.with_calls(
                    assistant(None, tool_calls=[tool_call(arguments=arguments)]),  # type: ignore[arg-type]
                    tool_result(),
                    assistant("y"),
                )
            )
            assert any("arguments" in issue.location for issue in issues), arguments
            assert any(fragment in issue.message for issue in issues), arguments

    def test_deeply_nested_arguments_rejected(self) -> None:
        arguments = '{"deep": ' + "[" * 70 + "]" * 70 + "}"
        issues = issues_of(
            self.with_calls(
                assistant(None, tool_calls=[tool_call(arguments=arguments)]),
                tool_result(),
                assistant("y"),
            )
        )
        assert any("nesting exceeds 64 levels" in issue.message for issue in issues)


class TestMetadata:
    def test_must_be_object(self) -> None:
        assert_single_issue(
            [{"messages": [user(), assistant()], "metadata": []}], "$.metadata", "JSON object"
        )

    def test_reserved_prefix_rejected_at_top_level_only(self) -> None:
        issues = issues_of({"messages": [user(), assistant()], "metadata": {"_castform_run": 1}})
        assert "reserved" in issues[0].message
        SftDataset.from_rows(
            [{"messages": [user(), assistant()], "metadata": {"nested": {"_castform_run": 1}}}]
        )

    def test_tools_key_reserved_for_runtime(self) -> None:
        issues = issues_of({"messages": [user(), assistant()], "metadata": {"tools": [{"x": 1}]}})
        assert 'metadata key "tools" is reserved' in issues[0].message
        # Nested occurrences are user data, not the runtime slot.
        SftDataset.from_rows(
            [{"messages": [user(), assistant()], "metadata": {"inner": {"tools": 1}}}]
        )

    def test_size_cap(self) -> None:
        big = {"messages": [user(), assistant()], "metadata": {"blob": "x" * 66_000}}
        assert_single_issue([big], "$.metadata", "metadata canonical size")


class TestTrainableTurn:
    def test_no_assistant_message(self) -> None:
        issues = issues_of({"messages": [user()]})
        assert any("at least one assistant message with weight 1" in i.message for i in issues)

    def test_all_masked(self) -> None:
        issues = issues_of({"messages": [user(), assistant(weight=0)]})
        assert any("at least one assistant message with weight 1" in i.message for i in issues)

    def test_masked_plus_trained_valid(self) -> None:
        SftDataset.from_rows(
            [{"messages": [user(), assistant(weight=0), assistant("b", weight=1)]}]
        )


class TestErrorObject:
    def test_error_is_value_error_with_ordered_issue_tuple(self) -> None:
        with pytest.raises(SftDatasetError) as excinfo:
            SftDataset.from_rows([valid_row(), {"messages": []}, {"messages": "x"}])
        error = excinfo.value
        assert isinstance(error, ValueError)
        assert isinstance(error.issues, tuple)
        assert [issue.row for issue in error.issues] == [1, 2]
        assert "invalid benchmax-sft-v1 dataset: 2 issue(s)" in str(error)

    def test_issue_as_dict_shape(self) -> None:
        issue = issues_of({"messages": []})[0]
        assert issue.as_dict() == {
            "line": None,
            "location": "$.messages",
            "message": "messages must not be empty",
            "row": 0,
        }
