"""Tests for the SFT canonicalization boundary: load_sft_dataset + canonical_jsonl."""

from __future__ import annotations

import json

import pytest

from benchmax.sft.dataset import load_sft_dataset
from benchmax.sft.dataset import canonical_jsonl as _canonical_jsonl


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
        path = _write(tmp_path, "train.jsonl", '{"messages": [{"role": "a", "weight": Infinity}]}\n')
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
        canon_path = _write(tmp_path, "u2028_canon.jsonl", _canonical_jsonl(dataset).decode("utf-8"))
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
        assert _canonical_jsonl(dataset) == b""

    def test_round_trips_to_valid_jsonl(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "a"}]}\n'
            '{"messages": [{"role": "assistant", "content": "b"}]}\n',
        )
        dataset = load_sft_dataset(path)
        rendered = _canonical_jsonl(dataset).decode("utf-8")
        lines = rendered.splitlines()
        assert len(lines) == 2
        assert [json.loads(line) for line in lines] == [row.data for row in dataset.rows]

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
        canon_rows = [json.loads(line) for line in _canonical_jsonl(dataset).decode().splitlines()]
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
