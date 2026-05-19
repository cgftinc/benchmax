"""Tests for question deduplication."""

from __future__ import annotations

from unittest.mock import MagicMock

from benchmax.rag.qa_generation.generated_qa import GeneratedQA
from benchmax.rag.qa_generation.transformers.dedup import (
    DedupConfig,
    QuestionDeduplicator,
)


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.setdefault = lambda key, default: default
    return ctx


def _make_item(question: str, answer: str = "Answer.") -> GeneratedQA:
    return GeneratedQA(
        qa={
            "question": question,
            "answer": answer,
            "qa_type": "multi_hop",
            "reference_chunks": [],
            "min_hop_count": 2,
            "is_co_located": None,
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": {},
        },
    )


class TestQuestionDeduplicator:
    def test_removes_near_duplicates(self):
        items = [
            _make_item("How do I set up feature flags in PostHog?"),
            _make_item("How to set up feature flags in PostHog?"),
            _make_item("What events does PostHog autocapture?"),
        ]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.45))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        assert len(passed) == 2
        assert len(rejected) == 1
        assert rejected[0].filter_verdict.reason == "near_duplicate"

    def test_keeps_distinct_questions(self):
        items = [
            _make_item("How do I set up feature flags?"),
            _make_item("What is session replay in PostHog?"),
            _make_item("How to configure a reverse proxy?"),
        ]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.70))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        assert len(passed) == 3
        assert len(rejected) == 0

    def test_preserves_first_occurrence(self):
        items = [
            _make_item("How to configure feature flags?"),
            _make_item("How to configure feature flags?"),  # exact dup
        ]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.50))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        assert len(passed) == 1
        assert passed[0].qa["question"] == "How to configure feature flags?"

    def test_disabled(self):
        items = [
            _make_item("Same question here"),
            _make_item("Same question here"),
        ]
        dedup = QuestionDeduplicator(DedupConfig(enabled=False))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        assert len(passed) == 2

    def test_single_item(self):
        items = [_make_item("Only one question")]
        dedup = QuestionDeduplicator()
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        assert len(passed) == 1

    def test_empty_list(self):
        dedup = QuestionDeduplicator()
        passed, rejected = dedup.deduplicate([], [], _make_context())

        assert len(passed) == 0

    def test_appends_to_existing_rejected(self):
        items = [
            _make_item("How to configure feature flags?"),
            _make_item("How to configure feature flags?"),
        ]
        existing_rejected = [_make_item("Previously rejected")]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.50))
        passed, rejected = dedup.deduplicate(items, existing_rejected, _make_context())

        assert len(passed) == 1
        assert len(rejected) == 2  # 1 existing + 1 new dup

    def test_high_threshold_keeps_similar(self):
        items = [
            _make_item("How do I set up feature flags in PostHog?"),
            _make_item("How to set up feature flags in PostHog?"),
        ]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.95))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        # With 0.95 threshold, these similar but not identical should pass
        assert len(passed) == 2

    def test_transitive_duplicates(self):
        items = [
            _make_item("How to set up feature flags in PostHog app?"),
            _make_item("How to set up feature flags in PostHog?"),
            _make_item("How to set up feature flags in PostHog application?"),
        ]
        dedup = QuestionDeduplicator(DedupConfig(similarity_threshold=0.60))
        passed, rejected = dedup.deduplicate(items, [], _make_context())

        # First one kept, second and third are duplicates of first
        assert len(passed) == 1
        assert passed[0].qa["question"] == items[0].qa["question"]
