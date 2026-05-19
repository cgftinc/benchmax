"""Tests for post-filter QA type relabeling."""

from __future__ import annotations

from benchmax.rag.qa_generation.pipeline import _relabel_qa_types
from benchmax.rag.qa_generation.generated_qa import GeneratedQA


def _make_item(
    *,
    qa_type: str,
    chunks: list[dict],
) -> GeneratedQA:
    return GeneratedQA(
        qa={
            "question": "Some question",
            "answer": "Some answer",
            "qa_type": qa_type,
            "reference_chunks": chunks,
        },
        generation_metadata={"qa_type_target": qa_type, "refinement_count": 0},
    )


class TestRelabelQATypes:
    def test_lookup_with_two_refs_becomes_multi_hop(self):
        item = _make_item(
            qa_type="lookup",
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
                {"id": "c2", "metadata": {"file": "a.mdx"}, "content": "B"},
            ],
        )
        stats = _relabel_qa_types([item])
        assert item.qa["qa_type"] == "multi_hop"
        assert stats["lookup_to_multi_hop"] == 1

    def test_multi_hop_with_single_chunk_becomes_lookup(self):
        item = _make_item(
            qa_type="multi_hop",
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
            ],
        )
        stats = _relabel_qa_types([item])
        assert item.qa["qa_type"] == "lookup"
        assert stats["multi_hop_to_lookup"] == 1

    def test_multi_hop_same_file_with_two_refs_stays_multi_hop(self):
        item = _make_item(
            qa_type="multi_hop",
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
                {"id": "c2", "metadata": {"file": "a.mdx"}, "content": "B"},
            ],
        )
        stats = _relabel_qa_types([item])
        assert item.qa["qa_type"] == "multi_hop"
        assert stats["relabeled"] == 0

    def test_valid_multi_hop_unchanged(self):
        item = _make_item(
            qa_type="multi_hop",
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
                {"id": "c2", "metadata": {"file": "b.mdx"}, "content": "B"},
            ],
        )
        stats = _relabel_qa_types([item])
        assert item.qa["qa_type"] == "multi_hop"
        assert stats["relabeled"] == 0

    def test_valid_lookup_unchanged(self):
        item = _make_item(
            qa_type="lookup",
            chunks=[
                {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
            ],
        )
        stats = _relabel_qa_types([item])
        assert item.qa["qa_type"] == "lookup"
        assert stats["relabeled"] == 0

    def test_multiple_items(self):
        items = [
            _make_item(
                qa_type="lookup",
                chunks=[
                    {"id": "c1", "metadata": {"file": "a.mdx"}, "content": "A"},
                    {"id": "c2", "metadata": {"file": "a.mdx"}, "content": "B"},
                ],
            ),
            _make_item(
                qa_type="multi_hop",
                chunks=[
                    {"id": "c3", "metadata": {"file": "c.mdx"}, "content": "C"},
                ],
            ),
            _make_item(
                qa_type="multi_hop",
                chunks=[
                    {"id": "c4", "metadata": {"file": "d.mdx"}, "content": "D"},
                    {"id": "c5", "metadata": {"file": "e.mdx"}, "content": "E"},
                ],
            ),
        ]
        stats = _relabel_qa_types(items)
        assert items[0].qa["qa_type"] == "multi_hop"
        assert items[1].qa["qa_type"] == "lookup"
        assert items[2].qa["qa_type"] == "multi_hop"
        assert stats["relabeled"] == 2
        assert stats["lookup_to_multi_hop"] == 1
        assert stats["multi_hop_to_lookup"] == 1
