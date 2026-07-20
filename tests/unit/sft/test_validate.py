"""Tests for validate_sft_dataset / SftValidationReport."""

from __future__ import annotations

import json

from benchmax.sft.dataset import SftDataset, SftRow, load_sft_dataset
from benchmax.sft.validate import DEFAULT_MAX_ROW_BYTES, validate_sft_dataset

_TRAINED_ROW = {
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
}


def _dataset(path: str, rows_data: list[dict]) -> SftDataset:
    rows = [SftRow(path, i + 1, data) for i, data in enumerate(rows_data)]
    return SftDataset(path=path, rows=rows, load_issues=[])


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestValidDataset:
    def test_ok_true_with_no_issues(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        report = validate_sft_dataset(train)
        assert report.ok is True
        assert bool(report) is True
        assert not any(i.severity == "error" for i in report.issues)
        assert report.train_row_count == 1
        assert report.eval_row_count == 0

    def test_multimodal_row_is_ok(self):
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
        train = _dataset("train.jsonl", [row])
        report = validate_sft_dataset(train)
        assert report.ok is True


class TestEmptyTrain:
    def test_empty_train_is_not_ok(self):
        train = _dataset("train.jsonl", [])
        report = validate_sft_dataset(train)
        assert report.train_row_count == 0
        assert report.ok is False
        assert bool(report) is False
        assert any(
            i.severity == "error" and "train dataset is empty" in i.message for i in report.issues
        )

    def test_empty_train_via_load_sft_dataset(self, tmp_path):
        path = _write(tmp_path, "train.jsonl", "\n\n")
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)
        assert report.ok is False


class TestEmptyEval:
    def test_absent_eval_is_notice(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        report = validate_sft_dataset(train)
        assert report.ok is True
        notices = [i for i in report.issues if i.severity == "notice"]
        assert any("no eval dataset provided" in i.message for i in notices)

    def test_empty_eval_dataset_is_notice_not_error(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        eval_ds = _dataset("eval.jsonl", [])
        report = validate_sft_dataset(train, eval_ds)
        assert report.ok is True
        assert report.eval_row_count == 0
        notice = next(i for i in report.issues if "eval dataset is empty" in i.message)
        assert notice.severity == "notice"

    def test_nonempty_eval_counted(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        eval_ds = _dataset("eval.jsonl", [_TRAINED_ROW, _TRAINED_ROW])
        report = validate_sft_dataset(train, eval_ds)
        assert report.eval_row_count == 2
        assert not any("no eval dataset" in i.message or "eval dataset is empty" in i.message for i in report.issues)


class TestWeightNotice:
    def test_weight_present_is_notice_not_error(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo", "weight": 1},
            ]
        }
        train = _dataset("train.jsonl", [row])
        report = validate_sft_dataset(train)
        assert report.ok is True
        weight_issues = [i for i in report.issues if "weight" in i.message and i.severity == "notice"]
        assert len(weight_issues) == 1

    def test_masking_summary_counts_trained_and_masked(self):
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a", "weight": 0},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "b", "weight": 1},
            ]
        }
        no_weight_row = _TRAINED_ROW
        train = _dataset("train.jsonl", [row, no_weight_row])
        report = validate_sft_dataset(train)
        assert report.masking_summary.rows_with_weight == 1
        assert report.masking_summary.masked_assistant_messages == 1
        # trained: weight=1 message in `row`, plus the implicit-weight assistant message in no_weight_row
        assert report.masking_summary.trained_assistant_messages == 2

    def test_no_weight_rows_have_zero_masking_summary(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        report = validate_sft_dataset(train)
        assert report.masking_summary.rows_with_weight == 0
        assert report.masking_summary.masked_assistant_messages == 0
        assert report.masking_summary.trained_assistant_messages == 1


class TestRowSizeNotice:
    def test_row_at_default_max_row_bytes_gets_notice(self):
        big_content = "x" * (DEFAULT_MAX_ROW_BYTES + 1000)
        row = {
            "messages": [
                {"role": "user", "content": big_content},
                {"role": "assistant", "content": "ok"},
            ]
        }
        assert len(json.dumps(row).encode("utf-8")) >= DEFAULT_MAX_ROW_BYTES
        train = _dataset("train.jsonl", [row])
        report = validate_sft_dataset(train)
        size_notices = [i for i in report.issues if "max_row_bytes" in i.message]
        assert len(size_notices) == 1
        assert size_notices[0].severity == "notice"

    def test_small_row_gets_no_size_notice(self):
        train = _dataset("train.jsonl", [_TRAINED_ROW])
        report = validate_sft_dataset(train)
        assert not any("max_row_bytes" in i.message for i in report.issues)


class TestTokenLengthStats:
    def test_stats_reflect_char_over_four_heuristic(self):
        row_a = {
            "messages": [
                {"role": "user", "content": "a" * 40},
                {"role": "assistant", "content": "b" * 40},
            ]
        }
        row_b = {
            "messages": [
                {"role": "user", "content": "c" * 4},
                {"role": "assistant", "content": "d" * 4},
            ]
        }
        train = _dataset("train.jsonl", [row_a, row_b])
        report = validate_sft_dataset(train)
        stats = report.token_length_stats
        assert stats.max_tokens == 80 // 4
        assert stats.min_tokens == 8 // 4
        assert stats.rows_over_max_seq_len == 0

    def test_row_over_max_seq_len_flagged_as_notice(self):
        row = {
            "messages": [
                {"role": "user", "content": "a" * 400},
                {"role": "assistant", "content": "ok"},
            ]
        }
        train = _dataset("train.jsonl", [row])
        report = validate_sft_dataset(train, max_seq_len=10)
        assert report.token_length_stats.rows_over_max_seq_len == 1
        notice = next(i for i in report.issues if "max_seq_len" in i.message)
        assert notice.severity == "notice"
        assert report.ok is True  # a heuristic overflow never blocks the gate


class TestErrorClassesWithPhysicalLines:
    def test_physical_lines_survive_blanks_and_malformed_json(self, tmp_path):
        text = (
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "yo"}]}\n'
            "\n"
            "{not valid json\n"
            '{"messages": []}\n'
        )
        path = _write(tmp_path, "train.jsonl", text)
        train = load_sft_dataset(path)

        assert [r.physical_line for r in train.rows] == [1, 4]

        report = validate_sft_dataset(train)
        by_line = {i.physical_line: i for i in report.issues if i.severity == "error"}

        assert 3 in by_line
        assert "invalid JSON" in by_line[3].message

        assert 4 in by_line
        assert "non-empty list" in by_line[4].message

        assert 1 not in by_line
        assert report.ok is False


class TestSchemaCrashSafetyThroughFullPipeline:
    """Malformed-but-JSON-shaped rows (list role, dict weight, etc.) must
    survive load_sft_dataset -> validate_sft_dataset as ordinary error
    issues, never as an uncaught exception."""

    def test_list_role_produces_report_not_exception(self, tmp_path):
        path = _write(tmp_path, "train.jsonl", '{"messages": [{"role": [], "content": "hi"}]}\n')
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)  # must not raise
        assert any(i.severity == "error" for i in report.issues)
        assert report.ok is False

    def test_dict_weight_produces_report_not_exception(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "assistant", "content": "hi", "weight": {}}]}\n',
        )
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)  # must not raise
        assert any(i.severity == "error" for i in report.issues)

    def test_list_content_part_type_produces_report_not_exception(self, tmp_path):
        path = _write(
            tmp_path,
            "train.jsonl",
            '{"messages": [{"role": "user", "content": [{"type": []}]}, '
            '{"role": "assistant", "content": "hi"}]}\n',
        )
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)  # must not raise
        assert any(i.severity == "error" for i in report.issues)


class TestSerializationGuard:
    def test_unserializable_row_reports_error_without_crashing_row_size_check(self):
        # A row built directly (bypassing the loader's NaN/type guards) with
        # a non-serializable value must not crash validate_sft_dataset's
        # row-size measurement — it should surface as an issue instead.
        row_with_set = {**_TRAINED_ROW, "tools": [{"bad": {1, 2, 3}}]}
        train = _dataset("train.jsonl", [row_with_set])
        report = validate_sft_dataset(train)  # must not raise
        assert any("not JSON-serializable" in i.message for i in report.issues)
        assert not any("max_row_bytes" in i.message for i in report.issues)

    def test_structural_error_and_serialization_error_both_surface(self):
        row = {
            "messages": [{"role": "not-a-real-role", "content": "hi"}],
            "extra_metadata": {1, 2, 3},  # a set is not JSON-serializable
        }
        train = _dataset("train.jsonl", [row])
        report = validate_sft_dataset(train)  # must not raise
        errors = [i.message for i in report.issues if i.severity == "error"]
        assert any("role must be one of" in m for m in errors)
        assert any("not JSON-serializable" in m for m in errors)

    def test_nan_rejected_at_load_never_reaches_validate(self, tmp_path):
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "assistant", "content": NaN}]}\n'
        )
        train = load_sft_dataset(path)  # NaN rejected at load — not a valid row
        assert train.rows == []
        assert any(i.severity == "error" for i in train.load_issues)
        report = validate_sft_dataset(train)  # must not raise
        assert report.ok is False


class TestEmptyTextContentPartFullPipeline:
    def test_only_empty_text_part_does_not_count_as_trained(self, tmp_path):
        text = (
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": [{"type": "text", "text": ""}]}]}\n'
        )
        path = _write(tmp_path, "train.jsonl", text)
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)
        assert report.ok is False
        assert any(
            i.severity == "error" and "no trained assistant turn" in i.message
            for i in report.issues
        )

    def test_non_empty_text_part_counts_as_trained(self, tmp_path):
        text = (
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}]}\n'
        )
        path = _write(tmp_path, "train.jsonl", text)
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)
        assert report.ok is True


class TestToolCallArgumentsNonFiniteFullPipeline:
    def test_nan_in_tool_call_arguments_produces_error_not_exception(self, tmp_path):
        text = (
            '{"messages": [{"role": "assistant", "content": null, "tool_calls": '
            '[{"id": "1", "type": "function", "function": {"name": "f", '
            '"arguments": "{\\"score\\": NaN}"}}]}]}\n'
        )
        path = _write(tmp_path, "train.jsonl", text)
        train = load_sft_dataset(path)
        report = validate_sft_dataset(train)  # must not raise
        assert report.ok is False
        assert any("must be valid JSON" in i.message for i in report.issues)


class TestLoneSurrogateFullPipeline:
    def test_lone_surrogate_content_produces_error_not_exception(self, tmp_path):
        path = _write(
            tmp_path, "train.jsonl", '{"messages": [{"role": "assistant", "content": "\\ud800"}]}\n'
        )
        train = load_sft_dataset(path)  # valid JSON syntax — loads fine
        assert len(train.rows) == 1
        report = validate_sft_dataset(train)  # must not raise
        assert report.ok is False
        assert any("not JSON-serializable" in i.message for i in report.issues)
