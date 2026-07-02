"""Tests for removed_reference_chunks tracking in GroundingLLMFilter."""

from __future__ import annotations

from benchmax.rag.qa_generation.filters.grounding_llm import GroundingLLMFilter
from benchmax.rag.qa_generation.generated_qa import GeneratedQA


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

    from benchmax.rag.qa_generation.pipeline_config import GroundingLLMFilterConfig

    cfg = GroundingLLMFilterConfig(
        enabled=True,
        judge_model="test-model",
        judge_api_key="test-key",
        judge_base_url="http://test",
        batch_enabled=False,
    )
    return GroundingLLMFilter(chunk_source=MagicMock(), cfg=cfg)


def test_keyless_filter_resolves_judge_key_via_seam(monkeypatch):
    """An empty judge_api_key resolves the platform bearer via the credential
    seam (PLATFORM_API_KEY here) instead of building the judge client with an
    empty key that would 401."""
    from unittest.mock import MagicMock

    from benchmax.rag.qa_generation.pipeline_config import GroundingLLMFilterConfig

    monkeypatch.delenv("ACT_AS_TOKEN_PATH", raising=False)
    monkeypatch.setenv("PLATFORM_API_KEY", "sk_seam")
    cfg = GroundingLLMFilterConfig(
        enabled=True,
        judge_api_key="",  # keyless → seam
        judge_base_url="http://test",
        batch_enabled=False,
    )
    filt = GroundingLLMFilter(chunk_source=MagicMock(), cfg=cfg)
    assert filt.judge_client.api_key == "sk_seam"


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
        from unittest.mock import MagicMock

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
