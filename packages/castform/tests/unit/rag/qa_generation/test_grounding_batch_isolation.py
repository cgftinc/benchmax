"""Batch-mode error isolation for GroundingLLMFilter.

Regression for a bug where the batch error path referenced ``data.query`` (no such field
on ``_GroundingItemData``): when ``_verdict_from_judge_result`` raised for one item, the
handler itself raised ``AttributeError`` and aborted the whole batch. The error path must
instead produce an error verdict for the failing item (from its question) and leave
siblings untouched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from castform.rag.qa_generation.filters.grounding_llm import GroundingLLMFilter
from castform.rag.qa_generation.generated_qa import GeneratedQA
from castform.rag.qa_generation.pipeline_config import (
    GroundingLLMFilterConfig,
    PipelineContext,
)


def _make_item(question: str) -> GeneratedQA:
    return GeneratedQA(
        qa={
            "question": question,
            "answer": "A does X.",
            "qa_type": "multi_hop",
            "reference_chunks": [{"id": "c1", "metadata": {}, "content": "A does X."}],
        },
        generation_metadata={"refinement_count": 0},
    )


def _make_batch_filter() -> GroundingLLMFilter:
    cfg = GroundingLLMFilterConfig(
        enabled=True,
        judge_model="test-model",
        judge_api_key="test-key",
        judge_base_url="http://test",
        batch_enabled=True,
    )
    return GroundingLLMFilter(chunk_source=MagicMock(), cfg=cfg)


async def test_batch_error_path_isolates_failing_item(monkeypatch):
    filt = _make_batch_filter()
    items = [_make_item("Q0"), _make_item("Q1")]
    stats: dict = {}
    ctx = PipelineContext(config=None, source=None)

    # Skip the network judge phase entirely (no prompts -> no batch_process_async call).
    monkeypatch.setattr(filt, "_build_judge_prompt", lambda *a, **k: None)

    # Make the verdict builder raise for the first item only; the second runs for real.
    real_verdict = filt._verdict_from_judge_result

    def flaky(item, **kwargs):
        if item is items[0]:
            raise ValueError("boom")
        return real_verdict(item, **kwargs)

    monkeypatch.setattr(filt, "_verdict_from_judge_result", flaky)

    result = await filt._evaluate_batch(items, context=ctx, stats=stats, max_refinements=2)

    # Batch completed (no AttributeError from the error path); isolation held.
    assert result is items
    # Failing item got the error verdict, built from its question.
    assert items[0].filter_verdict is not None
    assert items[0].filter_verdict.metadata["reason_code"] == "filter_error"
    # Sibling was evaluated normally, not clobbered by the failure.
    assert items[1].filter_verdict is not None
    assert items[1].filter_verdict.metadata.get("reason_code") != "filter_error"
    assert stats.get("errors") == 1
