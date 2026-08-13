"""Alignment and metric tests, all on original synthetic fixtures.

The metric numbers here are hand-calculated in the test bodies rather than
copied from a run, so a change in behavior fails loudly instead of silently
re-baselining.
"""

from __future__ import annotations

import pytest
from pii_masking.benchmark_scoring import (
    INVALID_AMBIGUOUS_ALIGNMENT,
    INVALID_NO_ALIGNMENT,
    LITERAL,
    PLACEHOLDER,
    VALID,
    ScoringError,
    TaskCounters,
    align,
    build_report,
    check_intervals,
    f_beta,
    merge_intervals,
    overlap_length,
    segment_output,
    task_average,
    task_metrics,
    total_length,
)


# ── segmentation ──────────────────────────────────────────────────────────────
class TestSegmentation:
    def test_literals_and_placeholders_alternate(self):
        segments = segment_output("call [NAME_1] now")

        assert [s.kind for s in segments] == [LITERAL, PLACEHOLDER, LITERAL]
        assert segments[0].text == "call "
        assert segments[2].text == " now"

    def test_adjacent_placeholders_collapse_into_one_run(self):
        segments = segment_output("[NAME_1][PHONE_1] here")

        assert [s.kind for s in segments] == [PLACEHOLDER, LITERAL]

    def test_placeholders_separated_by_real_text_stay_separate(self):
        segments = segment_output("[NAME_1] and [PHONE_1]")

        assert [s.kind for s in segments] == [PLACEHOLDER, LITERAL, PLACEHOLDER]

    def test_placeholders_separated_only_by_whitespace_merge(self):
        # "[TITLE] [GIVENNAME] [SURNAME]" over a four-token name has several
        # equally valid splits that all mask the same characters, so the run is
        # treated as one label-agnostic span.
        assert [s.kind for s in segment_output("[NAME_1] [PHONE_1]")] == [PLACEHOLDER]
        assert [s.kind for s in segment_output("[A_1]\t[B_1]\n[C_1]")] == [PLACEHOLDER]

    def test_index_zero_is_a_valid_placeholder(self):
        assert [s.kind for s in segment_output("[NAME_0]")] == [PLACEHOLDER]

    def test_placeholder_without_an_index_is_valid(self):
        assert [s.kind for s in segment_output("[NAME]")] == [PLACEHOLDER]

    @pytest.mark.parametrize(
        "text", ["[lowercase]", "[Name_1]", "[_NAME]", "[1NAME]", "[NAME-1]", "[]", "[NAME_1"]
    )
    def test_malformed_placeholders_are_literal_text(self, text):
        assert all(s.kind == LITERAL for s in segment_output(text))


# ── alignment ─────────────────────────────────────────────────────────────────
class TestAlignment:
    def test_single_masked_span_in_the_middle(self):
        result = align("call Ada now", "call [NAME_1] now")

        assert result.status == VALID
        assert result.intervals == ((5, 8),)

    def test_leading_placeholder_anchors_on_the_trailing_literal(self):
        result = align("Ada called", "[NAME_1] called")

        assert result.intervals == ((0, 3),)

    def test_trailing_placeholder_runs_to_the_end(self):
        result = align("call Ada", "call [NAME_1]")

        assert result.intervals == ((5, 8),)

    def test_whole_text_masked(self):
        result = align("Ada", "[NAME_1]")

        assert result.intervals == ((0, 3),)

    def test_two_separate_spans(self):
        result = align("Ada and Bob", "[NAME_1] and [NAME_2]")

        assert result.intervals == ((0, 3), (8, 11))

    def test_no_placeholders_requires_an_exact_copy(self):
        assert align("nothing here", "nothing here").status == VALID
        assert align("nothing here", "nothing here").intervals == ()

    def test_no_placeholders_and_a_rewrite_is_invalid(self):
        assert align("nothing here", "Nothing here.").status == INVALID_NO_ALIGNMENT

    def test_a_placeholder_must_cover_at_least_one_character(self):
        # "ab" with a placeholder between 'a' and 'b' would cover zero chars.
        assert align("ab", "a[NAME_1]b").status == INVALID_NO_ALIGNMENT

    def test_repeated_anchor_text_is_ambiguous(self):
        # The middle "b" can sit at index 2 or 4, and both complete a full
        # monotonic cover, so the mapping is genuinely not determined.
        assert align("aXbXbXc", "a[N_1]b[N_2]c").status == INVALID_AMBIGUOUS_ALIGNMENT

    def test_ambiguity_is_not_resolved_leftmost(self):
        result = align("aXbXbXc", "a[N_1]b[N_2]c")

        assert result.status == INVALID_AMBIGUOUS_ALIGNMENT
        assert result.intervals == ()

    def test_edge_anchoring_can_make_a_repeated_literal_unambiguous(self):
        # "a" repeats, but the trailing literal is anchored at the source end,
        # which leaves exactly one solution.
        result = align("aXaXa", "a[N_1]a")

        assert result.status == VALID
        assert result.intervals == ((1, 4),)

    def test_a_preamble_makes_the_output_invalid(self):
        assert align("call Ada", "Sure! call [NAME_1]").status == INVALID_NO_ALIGNMENT

    def test_a_code_fence_makes_the_output_invalid(self):
        assert align("call Ada", "```\ncall [NAME_1]\n```").status == INVALID_NO_ALIGNMENT

    def test_truncation_cutting_a_trailing_literal_is_invalid(self):
        assert align("call Ada now", "call [NAME_1] no").status == INVALID_NO_ALIGNMENT

    def test_truncation_ending_in_a_placeholder_is_over_masking_not_invalid(self):
        # Indistinguishable from a model that masked too much; alignment cannot
        # tell them apart, so precision and FPR carry the penalty instead.
        result = align("call Ada now", "call [NAME_1]")

        assert result.status == VALID
        assert result.intervals == ((5, 12),)

    def test_literal_drift_is_invalid(self):
        assert align("call Ada now", "call [NAME_1] NOW").status == INVALID_NO_ALIGNMENT

    def test_adjacent_placeholders_yield_one_merged_interval(self):
        result = align("AdaBob here", "[NAME_1][NAME_2] here")

        assert result.intervals == ((0, 6),)

    def test_empty_output_against_nonempty_source_is_invalid(self):
        assert align("something", "").status == INVALID_NO_ALIGNMENT

    def test_invalid_alignments_carry_no_intervals(self):
        for output in ("Sure! call [NAME_1]", "a[N_1]x[N_2]c"):
            assert align("axbxc", output).intervals == ()


# ── intervals ─────────────────────────────────────────────────────────────────
class TestIntervals:
    def test_touching_intervals_merge(self):
        assert merge_intervals([(0, 3), (3, 5)]) == [(0, 5)]

    def test_overlapping_intervals_merge(self):
        assert merge_intervals([(0, 4), (2, 6)]) == [(0, 6)]

    def test_disjoint_intervals_are_preserved(self):
        assert merge_intervals([(5, 7), (0, 3)]) == [(0, 3), (5, 7)]

    def test_nested_intervals_merge(self):
        assert merge_intervals([(0, 10), (3, 5)]) == [(0, 10)]

    def test_total_length_is_half_open(self):
        assert total_length([(0, 3)]) == 3

    def test_overlap_of_disjoint_lists_is_zero(self):
        assert overlap_length([(0, 3)], [(5, 8)]) == 0

    def test_overlap_of_touching_lists_is_zero(self):
        # Half-open: [0,3) and [3,6) share no character.
        assert overlap_length([(0, 3)], [(3, 6)]) == 0

    def test_partial_overlap(self):
        assert overlap_length([(0, 5)], [(3, 8)]) == 2

    def test_overlap_across_multiple_intervals(self):
        assert overlap_length([(0, 5), (10, 15)], [(3, 12)]) == 4

    def test_out_of_range_intervals_are_rejected(self):
        with pytest.raises(ScoringError, match="outside"):
            check_intervals([(0, 11)], 10)

    def test_negative_intervals_are_rejected(self):
        with pytest.raises(ScoringError, match="outside"):
            check_intervals([(-1, 3)], 10)

    def test_empty_intervals_are_rejected(self):
        with pytest.raises(ScoringError, match="empty or inverted"):
            check_intervals([(3, 3)], 10)

    def test_inverted_intervals_are_rejected(self):
        with pytest.raises(ScoringError, match="empty or inverted"):
            check_intervals([(5, 2)], 10)


# ── metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_perfect_mask(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 4)])

        metrics = task_metrics(counters)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["f2"] == 1.0
        assert metrics["fpr"] == 0.0
        assert metrics["support_chars"] == 4

    def test_no_prediction_scores_zero_without_dividing_by_zero(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[])

        metrics = task_metrics(counters)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f2"] == 0.0

    def test_no_gold_and_no_prediction(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[], predicted=[])

        metrics = task_metrics(counters)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["fpr"] == 0.0

    def test_pure_false_positive(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[], predicted=[(0, 2)])

        metrics = task_metrics(counters)

        assert metrics["precision"] == 0.0
        # 2 false-positive chars out of 10 non-PII chars.
        assert metrics["fpr"] == pytest.approx(0.2)

    def test_partial_overlap_hand_calculated(self):
        counters = TaskCounters()
        # gold [0,10), predicted [5,15) in a 20-char text: overlap 5.
        counters.add(source_length=20, gold=[(0, 10)], predicted=[(5, 15)])

        metrics = task_metrics(counters)

        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["f1"] == pytest.approx(0.5)
        assert metrics["f2"] == pytest.approx(0.5)
        # false positives 5, non-PII chars 20 - 10 = 10.
        assert metrics["fpr"] == pytest.approx(0.5)

    def test_f2_weights_recall_above_precision(self):
        # precision 1.0, recall 0.5 -> F1 0.667, F2 0.556
        assert f_beta(1.0, 0.5, 1.0) == pytest.approx(2 / 3, rel=1e-6)
        assert f_beta(1.0, 0.5, 2.0) == pytest.approx(5 / 9, rel=1e-6)
        # precision 0.5, recall 1.0 -> F2 rewards the recall-heavy case more
        assert f_beta(0.5, 1.0, 2.0) == pytest.approx(5 / 6, rel=1e-6)

    def test_f_beta_is_zero_when_both_inputs_are_zero(self):
        assert f_beta(0.0, 0.0, 1.0) == 0.0
        assert f_beta(0.0, 0.0, 2.0) == 0.0

    def test_gold_intervals_are_merged_before_counting(self):
        counters = TaskCounters()
        # Overlapping gold must count 6 characters, not 4 + 4.
        counters.add(source_length=10, gold=[(0, 4), (2, 6)], predicted=[(0, 6)])

        assert task_metrics(counters)["support_chars"] == 6
        assert task_metrics(counters)["precision"] == 1.0

    def test_consecutive_gold_intervals_are_merged(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 3), (3, 6)], predicted=[(0, 6)])

        assert task_metrics(counters)["support_chars"] == 6

    def test_aggregation_is_micro_not_macro(self):
        counters = TaskCounters()
        # A tiny perfect document and a large failing one. Macro would report
        # 0.5 recall; micro weights by characters and reports 1/101.
        counters.add(source_length=10, gold=[(0, 1)], predicted=[(0, 1)])
        counters.add(source_length=200, gold=[(0, 100)], predicted=[])

        metrics = task_metrics(counters)

        assert metrics["recall"] == pytest.approx(1 / 101)

    def test_invalid_documents_are_counted_and_predict_nothing(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[], status=INVALID_NO_ALIGNMENT)

        metrics = task_metrics(counters)

        assert metrics["invalid_documents"] == 1
        assert metrics["invalid_rate"] == 1.0
        assert metrics["invalid_by_class"] == {INVALID_NO_ALIGNMENT: 1}
        assert metrics["recall"] == 0.0

    def test_invalid_rate_is_per_document(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 4)])
        counters.add(
            source_length=10, gold=[(0, 4)], predicted=[], status=INVALID_AMBIGUOUS_ALIGNMENT
        )
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 4)])

        assert task_metrics(counters)["invalid_rate"] == pytest.approx(1 / 3)


class TestTaskAverage:
    def test_average_is_unweighted(self):
        small = TaskCounters()
        small.add(source_length=10, gold=[(0, 5)], predicted=[(0, 5)])
        large = TaskCounters()
        for _ in range(100):
            large.add(source_length=10, gold=[(0, 5)], predicted=[])

        per_task = {"a": task_metrics(small), "b": task_metrics(large)}

        # Unweighted: (1.0 + 0.0) / 2, despite 'b' having 100x the documents.
        assert task_average(per_task)["recall"] == pytest.approx(0.5)

    def test_average_of_no_tasks_is_zero(self):
        assert task_average({})["f2"] == 0.0


class TestReport:
    def _counters(self, recall_numerator: int) -> TaskCounters:
        counters = TaskCounters()
        counters.add(
            source_length=20,
            gold=[(0, 10)],
            predicted=[(0, recall_numerator)] if recall_numerator else [],
        )
        return counters

    def test_report_contains_both_models_and_a_delta(self):
        report = build_report(
            {
                "base": {"task-a": self._counters(2), "task-b": self._counters(2)},
                "sft": {"task-a": self._counters(8), "task-b": self._counters(8)},
            }
        )

        assert set(report["models"]) == {"base", "sft"}
        assert report["primary_metric"] == "f2"
        assert report["delta"]["recall"] == pytest.approx(0.6)

    def test_report_has_no_source_text(self):
        import json

        report = build_report({"base": {"task-a": self._counters(5)}})

        assert "source_text" not in json.dumps(report)

    def test_report_is_reproducible(self):
        first = build_report({"base": {"task-a": self._counters(5)}})
        second = build_report({"base": {"task-a": self._counters(5)}})

        assert first == second

    def test_delta_is_absent_without_both_models(self):
        report = build_report({"base": {"task-a": self._counters(5)}})

        assert "delta" not in report


# ── end-to-end alignment into metrics ─────────────────────────────────────────
class TestAlignThenScore:
    def score_one(self, source, output, gold):
        result = align(source, output)
        counters = TaskCounters()
        counters.add(
            source_length=len(source),
            gold=gold,
            predicted=list(result.intervals),
            status=result.status,
        )
        return task_metrics(counters)

    def test_a_correct_mask_scores_perfectly(self):
        metrics = self.score_one("call Ada now", "call [NAME_1] now", [(5, 8)])

        assert metrics["f2"] == 1.0
        assert metrics["invalid_rate"] == 0.0

    def test_a_missed_entity_lowers_recall(self):
        metrics = self.score_one("call Ada now", "call Ada now", [(5, 8)])

        assert metrics["recall"] == 0.0
        assert metrics["invalid_rate"] == 0.0

    def test_a_correct_answer_with_a_preamble_is_invalid_not_credited(self):
        metrics = self.score_one("call Ada now", "Sure: call [NAME_1] now", [(5, 8)])

        assert metrics["invalid_rate"] == 1.0
        assert metrics["recall"] == 0.0

    def test_over_masking_costs_precision_and_fpr(self):
        metrics = self.score_one("call Ada now", "[NAME_1]", [(5, 8)])

        assert metrics["recall"] == 1.0
        assert metrics["precision"] == pytest.approx(3 / 12)
        assert metrics["fpr"] == pytest.approx(9 / 9)


class TestLeakRate:
    """Aggregate recall can look healthy while many documents each leak."""

    def test_a_fully_masked_document_does_not_leak(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 4)])

        assert task_metrics(counters)["leak_rate"] == 0.0

    def test_a_partially_masked_document_leaks(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 2)])

        assert task_metrics(counters)["leak_rate"] == 1.0

    def test_documents_without_pii_are_excluded_from_the_denominator(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[], predicted=[])
        counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 4)])

        metrics = task_metrics(counters)

        assert metrics["documents_with_pii"] == 1
        assert metrics["leak_rate"] == 0.0

    def test_leak_rate_is_not_implied_by_character_recall(self):
        counters = TaskCounters()
        # One huge fully-masked document plus three small leaky ones: character
        # recall stays high while most documents leak.
        counters.add(source_length=1000, gold=[(0, 900)], predicted=[(0, 900)])
        for _ in range(3):
            counters.add(source_length=10, gold=[(0, 4)], predicted=[(0, 3)])

        metrics = task_metrics(counters)

        assert metrics["recall"] > 0.98
        assert metrics["leak_rate"] == pytest.approx(0.75)

    def test_an_invalid_output_leaks_every_entity(self):
        counters = TaskCounters()
        counters.add(source_length=10, gold=[(0, 4)], predicted=[], status=INVALID_NO_ALIGNMENT)

        assert task_metrics(counters)["leak_rate"] == 1.0


class TestMultiTokenNameAlignment:
    """The case that made 57% of real documents unscoreable before the fix."""

    def test_a_three_placeholder_name_aligns_uniquely(self):
        result = align(
            "Miss Zarya Sunja Smoter, hello", "[TITLE_1] [GIVENNAME_1] [SURNAME_1], hello"
        )

        assert result.status == VALID
        assert result.intervals == ((0, 23),)

    def test_the_merged_run_includes_internal_whitespace(self):
        result = align("Ada Lovelace here", "[GIVENNAME_1] [SURNAME_1] here")

        # The space inside the redacted name is part of the masked region.
        assert result.intervals == ((0, 12),)

    def test_placeholders_separated_by_words_still_align_separately(self):
        result = align("Ada and Bob here", "[NAME_1] and [NAME_2] here")

        assert result.intervals == ((0, 3), (8, 11))

    def test_gold_and_predictions_use_the_same_rule(self):
        source = "Miss Zarya Sunja Smoter"
        gold = align(source, "[TITLE_1] [GIVENNAME_1] [SURNAME_1]")
        # A model that splits the name differently reaches the same characters.
        pred = align(source, "[TITLE_1] [NAME_1]")

        assert gold.status == VALID and pred.status == VALID
        assert gold.intervals == pred.intervals


class TestLenientGrammar:
    """Strict asks 'did it follow the contract'; lenient asks 'did it find the PII'."""

    SOURCE = "Please contact Ada Lovelace at ada@example.com or 555-0142."
    SPANS = ((15, 27), (31, 46), (50, 58))

    def test_strict_rejects_a_foreign_label_vocabulary(self):
        from pii_masking.benchmark_scoring import STRICT_PLACEHOLDER_PATTERN

        out = "Please contact [PERSON_NAME] at [EMAIL_ADDRESS] or [PHONE_NUMBER]."

        assert align(self.SOURCE, out, STRICT_PLACEHOLDER_PATTERN).status == INVALID_NO_ALIGNMENT

    def test_strict_rejects_angle_brackets(self):
        from pii_masking.benchmark_scoring import STRICT_PLACEHOLDER_PATTERN

        out = "Please contact <NAME> at <EMAIL> or <PHONE>."

        assert align(self.SOURCE, out, STRICT_PLACEHOLDER_PATTERN).status == INVALID_NO_ALIGNMENT

    @pytest.mark.parametrize(
        "output",
        [
            "Please contact [PERSON_NAME] at [EMAIL_ADDRESS] or [PHONE_NUMBER].",
            "Please contact <NAME> at <EMAIL> or <PHONE>.",
            "Please contact [GIVENNAME_1] [SURNAME_1] at [EMAIL_1] or [TELEPHONENUM_1].",
        ],
    )
    def test_lenient_finds_the_same_spans_regardless_of_vocabulary(self, output):
        from pii_masking.benchmark_scoring import LENIENT_PLACEHOLDER_PATTERN

        result = align(self.SOURCE, output, LENIENT_PLACEHOLDER_PATTERN)

        assert result.status == VALID
        assert result.intervals == self.SPANS

    def test_gold_parses_identically_under_both_grammars(self):
        from pii_masking.benchmark_scoring import (
            LENIENT_PLACEHOLDER_PATTERN,
            STRICT_PLACEHOLDER_PATTERN,
        )

        gold = "Please contact [GIVENNAME_1] [SURNAME_1] at [EMAIL_1] or [TELEPHONENUM_1]."

        assert (
            align(self.SOURCE, gold, STRICT_PLACEHOLDER_PATTERN).intervals
            == align(self.SOURCE, gold, LENIENT_PLACEHOLDER_PATTERN).intervals
        )

    @pytest.mark.parametrize("text", ["<ada@example.com>", "<html>", "<p>", "a < b and c > d"])
    def test_lenient_does_not_swallow_ordinary_angle_bracket_text(self, text):
        from pii_masking.benchmark_scoring import LENIENT_PLACEHOLDER_PATTERN

        assert LENIENT_PLACEHOLDER_PATTERN.search(text) is None

    def test_lenient_still_requires_a_consistent_alignment(self):
        from pii_masking.benchmark_scoring import LENIENT_PLACEHOLDER_PATTERN

        # Leniency is about the label vocabulary, not about accepting rewrites.
        out = "Sure! Here you go: Please contact <NAME>."

        assert align(self.SOURCE, out, LENIENT_PLACEHOLDER_PATTERN).status == INVALID_NO_ALIGNMENT
