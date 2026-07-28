"""Tests for removed_reference_chunks tracking in GroundingLLMFilter."""

from __future__ import annotations

import pytest
from castform.rag.qa_generation.filters.grounding_llm import GroundingLLMFilter
from castform.rag.qa_generation.generated_qa import GeneratedQA


def _make_item(chunks: list[dict]) -> GeneratedQA:
    return GeneratedQA(
        qa={
            "question": "What do A and B do?",
            "answer": "A does X.",
            "qa_type": "multi_hop",
            "reference_chunks": chunks,
            "min_hop_count": len(chunks),
            "is_co_located": None,
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": {},
        },
        generation_metadata={"refinement_count": 0},
    )


def _make_filter() -> GroundingLLMFilter:
    from unittest.mock import MagicMock

    from castform.rag.qa_generation.pipeline_config import GroundingLLMFilterConfig

    cfg = GroundingLLMFilterConfig(
        enabled=True,
        judge_model="test-model",
        judge_api_key="test-key",
        judge_base_url="http://test",
        batch_enabled=False,
    )
    return GroundingLLMFilter(chunk_source=MagicMock(), cfg=cfg)


def test_external_judge_requires_an_explicit_model_key(monkeypatch):
    """Generic platform credentials are never reused for an external judge."""
    from unittest.mock import MagicMock

    from castform.rag.qa_generation.pipeline_config import GroundingLLMFilterConfig

    monkeypatch.setenv("CASTFORM_AUTH_TOKEN", "forbidden-auth-token")
    monkeypatch.setenv("CASTFORM_API_KEY", "sk_seam")
    cfg = GroundingLLMFilterConfig(
        enabled=True,
        judge_api_key="",
        judge_base_url="http://test",
        batch_enabled=False,
    )
    with pytest.raises(ValueError, match="requires an explicit API key"):
        GroundingLLMFilter(chunk_source=MagicMock(), cfg=cfg)


class TestGroundingRemovedChunksTracking:
    def test_non_supporting_chunk_tracked_in_removed(self):
        """Chunks not in supporting_chunk_ids are moved to removed_reference_chunks."""
        chunks = [
            {"id": "c1", "metadata": {}, "content": "A does X."},
            {"id": "c2", "metadata": {}, "content": "Irrelevant content."},
        ]
        item = _make_item(chunks)
        filt = _make_filter()

        judge_result = {
            "answerable": True,
            "confidence": 0.9,
            "reasoning": "c1 supports the answer",
            "supporting_chunk_ids": ["c1"],
        }

        filt._verdict_from_judge_result(
            item,
            judge_result=judge_result,
            ref_chunks=chunks,
            evidence_blocks=["A does X.", "Irrelevant content."],
            max_refinements=2,
        )

        assert [c["id"] for c in item.qa["reference_chunks"]] == ["c1"]
        removed = item.qa.get("removed_reference_chunks", [])
        assert len(removed) == 1
        assert removed[0]["chunk"]["id"] == "c2"
        assert removed[0]["reason"] == "not_supporting"
        assert removed[0]["filter"] == "grounding_llm"

    def test_no_supporting_ids_leaves_reference_chunks_unchanged(self):
        """When judge gives no supporting_ids, reference_chunks stay as-is."""
        chunks = [
            {"id": "c1", "metadata": {}, "content": "A does X."},
            {"id": "c2", "metadata": {}, "content": "B does Y."},
        ]
        item = _make_item(chunks)
        filt = _make_filter()

        judge_result = {
            "answerable": True,
            "confidence": 0.8,
            "reasoning": "Both chunks support answer",
            "supporting_chunk_ids": [],
        }

        filt._verdict_from_judge_result(
            item,
            judge_result=judge_result,
            ref_chunks=chunks,
            evidence_blocks=["A does X.", "B does Y."],
            max_refinements=2,
        )

        # No IDs given — all chunks kept, none removed
        assert len(item.qa["reference_chunks"]) == 2
        assert item.qa.get("removed_reference_chunks") is None

    def test_all_supported_no_removed(self):
        """When all chunks are in supporting_ids, removed_reference_chunks is not set."""
        chunks = [
            {"id": "c1", "metadata": {}, "content": "A does X."},
            {"id": "c2", "metadata": {}, "content": "B does Y."},
        ]
        item = _make_item(chunks)
        filt = _make_filter()

        judge_result = {
            "answerable": True,
            "confidence": 0.95,
            "reasoning": "All chunks support",
            "supporting_chunk_ids": ["c1", "c2"],
        }

        filt._verdict_from_judge_result(
            item,
            judge_result=judge_result,
            ref_chunks=chunks,
            evidence_blocks=["A does X.", "B does Y."],
            max_refinements=2,
        )

        assert {c["id"] for c in item.qa["reference_chunks"]} == {"c1", "c2"}
        assert item.qa.get("removed_reference_chunks") is None
