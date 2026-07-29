"""Tests for the SFT canonicalization boundary: load_sft_dataset + canonical_jsonl."""

from __future__ import annotations

import json

import pytest
from benchmax.sft.dataset import (
    SftDataset,
    SftIssue,
    SftRow,
    SftSerializationError,
    canonical_jsonl,
    load_sft_dataset,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestValidRows:
    def test_loads_rows_with_provenance(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "yo"}]}\n',
        )
        dataset = load_sft_dataset(path)
        assert len(dataset.rows) == 1
        assert dataset.rows[0].source_path == str(path)
        assert dataset.rows[0].physical_line == 1
        assert dataset.load_issues == []

    def test_normalization_applied_on_load(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"prompt": "hi", "completion": "yo"}\n',
        )
        dataset = load_sft_dataset(path)
        assert dataset.rows[0].data == {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        }


class TestPhysicalLineNumbers:
    def test_blank_lines_counted_but_skipped(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            "\n"
            "\n"
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        assert [r.physical_line for r in dataset.rows] == [1, 4]
        assert dataset.load_issues == []

    def test_malformed_json_line_becomes_issue_not_exception(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            "\n"
            "{not valid json\n"
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)  # must not raise
        assert [r.physical_line for r in dataset.rows] == [1, 4]
        assert len(dataset.load_issues) == 1
        issue = dataset.load_issues[0]
        assert issue.physical_line == 3
        assert issue.severity == "error"
        assert "invalid JSON" in issue.message

    def test_deeply_nested_json_line_becomes_issue_not_exception(self, tmp_path):
        # deeply nested JSON overflows json's parser stack (RecursionError, not a
        # ValueError); the loader must surface it on the per-line issue path
        deep = "[" * 20000 + "]" * 20000
        path = _write(
            tmp_path,
            "train.jsonl",
            deep + "\n"
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)  # must not raise RecursionError
        assert [r.physical_line for r in dataset.rows] == [2]
        assert len(dataset.load_issues) == 1
        assert dataset.load_issues[0].physical_line == 1
        assert dataset.load_issues[0].severity == "error"
        assert "invalid JSON" in dataset.load_issues[0].message

    def test_non_object_json_line_becomes_issue(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            "[1, 2, 3]\n"
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        assert len(dataset.rows) == 1
        assert dataset.rows[0].physical_line == 2
        assert dataset.load_issues[0].physical_line == 1
        assert "JSON object" in dataset.load_issues[0].message


class TestNonFiniteConstants:
    def test_nan_becomes_load_issue_not_exception(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            '{"messages": [{"role": "assistant", "content": NaN}]}\n',
        )
        dataset = load_sft_dataset(path)  # must not raise
        assert [r.physical_line for r in dataset.rows] == [1]
        assert len(dataset.load_issues) == 1
        assert dataset.load_issues[0].physical_line == 2
        assert dataset.load_issues[0].severity == "error"

    def test_infinity_becomes_load_issue_not_exception(self, tmp_path):
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "a", "weight": Infinity}]}\n'
        )
        dataset = load_sft_dataset(path)  # must not raise
        assert dataset.rows == []
        assert len(dataset.load_issues) == 1
        assert dataset.load_issues[0].severity == "error"

    def test_negative_infinity_becomes_load_issue_not_exception(self, tmp_path):
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "a", "weight": -Infinity}]}\n'
        )
        dataset = load_sft_dataset(path)  # must not raise
        assert dataset.rows == []
        assert len(dataset.load_issues) == 1
        assert dataset.load_issues[0].severity == "error"


class TestUnicodeLineSeparators:
    def test_u2028_inside_content_does_not_split_physical_lines(self, tmp_path):
        separator_text = "before after"
        row1 = {
            "messages": [
                {"role": "user", "content": separator_text},
                {"role": "assistant", "content": "ok"},
            ]
        }
        row2 = {"messages": [{"role": "assistant", "content": "b"}]}
        text = json.dumps(row1, ensure_ascii=False) + "\n" + json.dumps(row2) + "\n"
        path = _write(tmp_path, "u2028.jsonl", text)

        dataset = load_sft_dataset(path)
        assert dataset.load_issues == []
        assert [r.physical_line for r in dataset.rows] == [1, 2]
        assert dataset.rows[0].data["messages"][0]["content"] == separator_text

    def test_u2028_round_trips_through_canonical_jsonl(self, tmp_path):
        separator_text = "before after"
        row1 = {
            "messages": [
                {"role": "user", "content": separator_text},
                {"role": "assistant", "content": "ok"},
            ]
        }
        row2 = {"messages": [{"role": "assistant", "content": "b"}]}
        text = json.dumps(row1, ensure_ascii=False) + "\n" + json.dumps(row2) + "\n"
        path = _write(tmp_path, "u2028.jsonl", text)

        dataset = load_sft_dataset(path)
        canon_path = _write(tmp_path, "u2028_canon.jsonl", canonical_jsonl(dataset).decode("utf-8"))
        reloaded = load_sft_dataset(canon_path)

        assert reloaded.load_issues == []
        assert [r.physical_line for r in reloaded.rows] == [1, 2]
        assert reloaded.rows[0].data["messages"][0]["content"] == separator_text


class TestMissingFile:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sft_dataset(tmp_path / "does_not_exist.jsonl")


class TestCanonicalJsonl:
    def test_empty_dataset_renders_empty_bytes(self, tmp_path):
        path = _write(tmp_path, "train.jsonl", "\n")
        dataset = load_sft_dataset(path)
        assert canonical_jsonl(dataset) == b""

    def test_round_trips_to_valid_jsonl(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        rendered = canonical_jsonl(dataset).decode("utf-8")
        lines = rendered.splitlines()
        assert len(lines) == 2
        assert [json.loads(line) for line in lines] == [row.data for row in dataset.rows]

    def test_ordinary_non_ascii_round_trips(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "héllo \\u2014 日本語"}]}\n',
        )
        dataset = load_sft_dataset(path)
        rendered = canonical_jsonl(dataset).decode("utf-8")
        assert json.loads(rendered)["messages"][0]["content"] == "héllo — 日本語"

    def test_lone_surrogate_is_refused_instead_of_producing_broken_bytes(self, tmp_path):
        # The row parses as JSON, so it survives loading; the serialization
        # gate is what stops it, rather than an UnicodeEncodeError escaping
        # from the middle of a partially-rendered file.
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "assistant", "content": "\\ud800"}]}\n'
        )
        dataset = load_sft_dataset(path)
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        assert any("not JSON-serializable" in i.message for i in excinfo.value.issues)

    def test_legacy_split_file_canonicalization_preserves_everything(self, tmp_path):
        legacy_row = {
            "prompt_messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        {"type": "text", "text": "what is this"},
                    ],
                },
            ],
            "completion_messages": [
                {
                    "role": "assistant",
                    "content": "it is a cat",
                    "weight": 1,
                    "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "cat"}}],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        }
        path = _write(tmp_path, "legacy.jsonl", json.dumps(legacy_row) + "\n")

        dataset = load_sft_dataset(path)
        canon_rows = [json.loads(line) for line in canonical_jsonl(dataset).decode().splitlines()]
        assert len(canon_rows) == 1
        row = canon_rows[0]

        assert set(row.keys()) == {"messages", "tools"}
        assert row["messages"][0] == {"role": "system", "content": "sys"}

        user_msg = row["messages"][1]
        assert user_msg["content"][0]["type"] == "image_url"
        assert user_msg["content"][0]["image_url"]["url"] == "data:image/png;base64,AAAA"
        assert user_msg["content"][1] == {"type": "text", "text": "what is this"}

        assistant_msg = row["messages"][2]
        assert assistant_msg["weight"] == 1
        tool_call = assistant_msg["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "lookup"
        assert json.loads(tool_call["function"]["arguments"]) == {"q": "cat"}

        assert row["tools"] == legacy_row["tools"]


_TRAINED_ROW = {
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
}


def _handmade_dataset(rows_data, load_issues=None):
    """An SftDataset built directly, bypassing load_sft_dataset — the raw-SDK
    path the serialization gate has to hold on."""
    rows = [SftRow("handmade.jsonl", i + 1, data) for i, data in enumerate(rows_data)]
    return SftDataset(
        path="handmade.jsonl", rows=rows, load_issues=list(load_issues or [])
    )


class TestCanonicalJsonlRefusesInvalidData:
    """canonical_jsonl enforces the canonicalize -> validate -> upload boundary
    rather than documenting it: nothing that failed to load, and no row the
    schema rejects, can be serialized toward storage."""

    def test_partially_loaded_file_is_refused(self, tmp_path):
        # one good row, one malformed line: serializing the good row alone
        # would silently upload a truncated dataset.
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            "{not valid json\n"
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        assert len(dataset.rows) == 2  # the good rows did load

        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        error = excinfo.value
        assert error.path == str(path)
        assert [(i.physical_line, "invalid JSON" in i.message) for i in error.issues] == [
            (2, True)
        ]

    def test_non_object_line_is_refused(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            "[1, 2, 3]\n" '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        with pytest.raises(SftSerializationError):
            canonical_jsonl(dataset)

    def test_schema_invalid_row_is_refused(self, tmp_path):
        # loads cleanly (valid JSON object) but carries no trained assistant turn
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "user", "content": "hi"}]}\n'
        )
        dataset = load_sft_dataset(path)
        assert dataset.load_issues == []

        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        assert any("no trained assistant turn" in i.message for i in excinfo.value.issues)

    def test_refusal_carries_provenance_for_every_bad_row(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "user", "content": "hi"}]}\n'
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "yo"}]}\n'
            '{"messages": []}\n',
        )
        dataset = load_sft_dataset(path)
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        bad_lines = {i.physical_line for i in excinfo.value.issues}
        assert bad_lines == {1, 3}  # line 2 is valid and contributes nothing
        assert all(i.source_path == str(path) for i in excinfo.value.issues)

    def test_refusal_message_names_the_path_and_issue_count(self, tmp_path):
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "user", "content": "hi"}]}\n'
        )
        dataset = load_sft_dataset(path)
        with pytest.raises(SftSerializationError, match="refusing to serialize"):
            canonical_jsonl(dataset)

    def test_handmade_dataset_with_invalid_row_is_refused(self):
        # the raw-SDK path: SftDataset/SftRow are publicly constructible, so
        # the gate cannot rely on load_sft_dataset having been used.
        dataset = _handmade_dataset([_TRAINED_ROW, {"messages": [], "extra": 1}])
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        assert all(i.source_path == "handmade.jsonl" for i in excinfo.value.issues)

    def test_handmade_dataset_with_unserializable_row_is_refused(self):
        dataset = _handmade_dataset([{**_TRAINED_ROW, "tools": [{"bad": {1, 2, 3}}]}])
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        assert any("not JSON-serializable" in i.message for i in excinfo.value.issues)

    def test_error_load_issue_alone_is_enough_to_refuse(self):
        # every row is valid, but the file did not load whole
        dataset = _handmade_dataset(
            [_TRAINED_ROW],
            load_issues=[SftIssue("handmade.jsonl", 7, "error", "invalid JSON: boom")],
        )
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        assert [i.physical_line for i in excinfo.value.issues] == [7]

    def test_notice_load_issue_does_not_block(self):
        # notices are advisory (size, token length, masking) and must not turn
        # into a refusal.
        dataset = _handmade_dataset(
            [_TRAINED_ROW],
            load_issues=[SftIssue("handmade.jsonl", 1, "notice", "just so you know")],
        )
        assert json.loads(canonical_jsonl(dataset).decode("utf-8")) == _TRAINED_ROW

    def test_many_issues_are_summarized_not_dumped(self):
        dataset = _handmade_dataset([{"messages": []} for _ in range(12)])
        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)
        message = str(excinfo.value)
        assert len(excinfo.value.issues) == 12
        assert "(+7 more)" in message

    def test_valid_dataset_still_serializes(self, tmp_path):
        path = _write(tmp_path, "train.jsonl", json.dumps(_TRAINED_ROW) + "\n")
        dataset = load_sft_dataset(path)
        assert json.loads(canonical_jsonl(dataset).decode("utf-8")) == _TRAINED_ROW


class TestDeeplyNestedRowsAreReportedNotCrashes:
    """Deeply nested JSON overflows the parser/encoder stack with RecursionError
    rather than a ValueError. Neither the loader nor the serialization gate may
    let that escape as a traceback."""

    def test_deeply_nested_line_becomes_a_load_issue(self, tmp_path):
        deep = "[" * 20000 + "]" * 20000
        path = _write(tmp_path, "train.jsonl", deep + "\n" + json.dumps(_TRAINED_ROW) + "\n")
        dataset = load_sft_dataset(path)  # must not raise RecursionError
        assert [r.physical_line for r in dataset.rows] == [2]
        assert [i.physical_line for i in dataset.load_issues] == [1]
        assert "invalid JSON" in dataset.load_issues[0].message

    def test_deeply_nested_line_makes_canonical_jsonl_refuse(self, tmp_path):
        deep = "[" * 20000 + "]" * 20000
        path = _write(tmp_path, "train.jsonl", deep + "\n" + json.dumps(_TRAINED_ROW) + "\n")
        dataset = load_sft_dataset(path)
        with pytest.raises(SftSerializationError):
            canonical_jsonl(dataset)

    def test_handmade_deeply_nested_row_is_refused_not_crashed(self):
        # bypasses the loader's parse guard, so the encoder is what overflows
        deep_value = []
        cursor = deep_value
        for _ in range(20000):
            nested = []
            cursor.append(nested)
            cursor = nested
        dataset = _handmade_dataset([{**_TRAINED_ROW, "tools": deep_value}])

        with pytest.raises(SftSerializationError) as excinfo:
            canonical_jsonl(dataset)  # must not raise RecursionError
        assert any("not JSON-serializable" in i.message for i in excinfo.value.issues)
