"""Tests for deterministic guards filter."""

from __future__ import annotations

from unittest.mock import MagicMock

from benchmax.rag.qa_generation.pipeline_config import DeterministicGuardsConfig
from benchmax.rag.qa_generation.filters.deterministic_guards import (
    DeterministicGuardsFilter,
    _jaccard,
    _max_chunk_pair_overlap,
    _word_set,
)
from benchmax.rag.qa_generation.generated_qa import GeneratedQA


def _make_context() -> MagicMock:
    ctx: dict = {"deterministic_guard_stats": {"passed": 0, "rejected": 0}}
    return ctx


def _make_item(
    *,
    question: str = "How do I configure PostHog?",
    answer: str = "Use the init function with your API key and host." * 4,
    qa_type: str = "lookup",
    chunks: list[dict] | None = None,
) -> GeneratedQA:
    if chunks is None:
        chunks = [{"id": "c1", "metadata": {"file": "a.mdx"}, "content": "Some content."}]
    return GeneratedQA(
        qa={
            "question": question,
            "answer": answer,
            "qa_type": qa_type,
            "reference_chunks": chunks,
        },
        generation_metadata={"qa_type_target": qa_type, "refinement_count": 0},
    )


# --- Jaccard / overlap helpers ---


class TestWordSetAndJaccard:
    def test_word_set_basic(self):
        assert _word_set("Hello World 123") == {"hello", "world", "123"}

    def test_word_set_empty(self):
        assert _word_set("") == set()

    def test_jaccard_identical(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        # intersection=1, union=3
        assert abs(_jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9

    def test_jaccard_empty(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard(set(), set()) == 0.0


class TestMaxChunkPairOverlap:
    def test_single_chunk(self):
        exceeded, sim = _max_chunk_pair_overlap(
            [{"content": "hello world"}], threshold=0.5
        )
        assert not exceeded
        assert sim == 0.0

    def test_identical_chunks(self):
        c = {"content": "PostHog analytics setup guide for production"}
        exceeded, sim = _max_chunk_pair_overlap([c, c], threshold=0.5)
        assert exceeded
        assert sim == 1.0

    def test_disjoint_chunks(self):
        c1 = {"content": "PostHog analytics setup guide production"}
        c2 = {"content": "Redis caching layer implementation strategy"}
        exceeded, sim = _max_chunk_pair_overlap([c1, c2], threshold=0.5)
        assert not exceeded

    def test_high_overlap_detected(self):
        # Shared snippet content
        c1 = {"content": "Set up a reverse proxy for PostHog to avoid ad blockers."}
        c2 = {"content": "Set up a reverse proxy for PostHog to avoid tracking blockers."}
        exceeded, sim = _max_chunk_pair_overlap([c1, c2], threshold=0.5)
        assert exceeded
        assert sim > 0.5


# --- Deterministic guards filter ---


class TestDeterministicGuards:
    def test_lookup_passes(self):
        cfg = DeterministicGuardsConfig()
        f = DeterministicGuardsFilter(cfg)
        item = _make_item()
        ctx = _make_context()
        f.evaluate([item], ctx)
        assert item.filter_verdict is not None
        assert item.filter_verdict.status == "passed"

    def test_multi_hop_short_answer_rejected(self):
        """Multi-hop answers under min_answer_chars should be rejected."""
        cfg = DeterministicGuardsConfig()
        f = DeterministicGuardsFilter(cfg)
        item = _make_item(
            qa_type="multi_hop",
            answer="Too short.",  # under 24 chars
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "Content A"},
                {"id": "c2", "metadata": {"file": "b.mdx"}, "content": "Content B"},
            ],
        )
        ctx = _make_context()
        f.evaluate([item], ctx)
        assert item.filter_verdict.status == "rejected"
        assert item.filter_verdict.reasoning == "answer_too_short"

    def test_multi_hop_high_chunk_overlap_rejected(self):
        """Multi-hop with near-identical reference chunks should be rejected."""
        cfg = DeterministicGuardsConfig(chunk_overlap_threshold=0.6)
        f = DeterministicGuardsFilter(cfg)
        item = _make_item(
            qa_type="multi_hop",
            answer="A detailed answer that synthesizes information from both chunks." * 3,
            chunks=[
                {
                    "id": "c1",
                    "metadata": {"file": "a.mdx"},
                    "content": "PostHog analytics setup guide for production deployment.",
                },
                {
                    "id": "c2",
                    "metadata": {"file": "b.mdx"},
                    "content": "PostHog analytics setup guide for production environments.",
                },
            ],
        )
        ctx = _make_context()
        f.evaluate([item], ctx)
        assert item.filter_verdict.status == "rejected"
        assert item.filter_verdict.reasoning == "multi_hop_high_chunk_overlap"

    def test_multi_hop_distinct_chunks_passes(self):
        """Multi-hop with genuinely different reference chunks should pass."""
        cfg = DeterministicGuardsConfig(chunk_overlap_threshold=0.6)
        f = DeterministicGuardsFilter(cfg)
        item = _make_item(
            qa_type="multi_hop",
            answer="A detailed answer that synthesizes information from both chunks." * 3,
            chunks=[
                {
                    "id": "c1",
                    "metadata": {"file": "a.mdx"},
                    "content": "PostHog analytics setup guide for production deployment.",
                },
                {
                    "id": "c2",
                    "metadata": {"file": "b.mdx"},
                    "content": "Redis caching layer implementation for low-latency queries.",
                },
            ],
        )
        ctx = _make_context()
        f.evaluate([item], ctx)
        assert item.filter_verdict.status == "passed"

    def test_meta_question_rejected(self):
        cfg = DeterministicGuardsConfig()
        f = DeterministicGuardsFilter(cfg)
        item = _make_item(
            question="Can't create question from these chunks",
            answer="No valid answer" * 10,
        )
        ctx = _make_context()
        f.evaluate([item], ctx)
        assert item.filter_verdict.status == "rejected"
        assert item.filter_verdict.reasoning == "meta_question_about_generation"

    def test_qa_type_fallback_to_generation_metadata(self):
        """When qa['qa_type'] is empty, generation_metadata['qa_type_target'] is used.

        A multi_hop item constructed this way must still satisfy the multi_hop rules:
        >= 2 reference chunks.
        """
        cfg = DeterministicGuardsConfig()
        f = DeterministicGuardsFilter(cfg)

        # Build an item where qa_type is absent but generation_metadata says multi_hop.
        # Only 1 reference chunk should trigger multi_hop_insufficient_chunks.
        item_short = GeneratedQA(
            qa={
                "question": "How do A and B interact with each other?",
                "answer": "A interacts with B through the shared X interface, which exposes a bidirectional event bus that both components consume and publish to.",
                "qa_type": "",  # empty — must fall back to generation_metadata
                "reference_chunks": [
                    {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "Content A."},
                ],
            },
            generation_metadata={"qa_type_target": "multi_hop", "refinement_count": 0},
        )
        ctx = _make_context()
        f.evaluate([item_short], ctx)
        assert item_short.filter_verdict.status == "rejected"
        assert item_short.filter_verdict.reasoning == "multi_hop_insufficient_chunks"

        # A multi_hop item with 2+ chunks must pass.
        long_answer = "A interacts with B through the shared X interface. " * 4
        item_pass = GeneratedQA(
            qa={
                "question": "How do component A and component B cooperate?",
                "answer": long_answer,
                "qa_type": "",  # empty — falls back
                "reference_chunks": [
                    {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "Component A details."},
                    {"id": "c2", "metadata": {"file": "b.mdx"}, "content": "Component B details."},
                ],
            },
            generation_metadata={"qa_type_target": "multi_hop", "refinement_count": 0},
        )
        ctx2 = _make_context()
        f.evaluate([item_pass], ctx2)
        assert item_pass.filter_verdict.status == "passed"
