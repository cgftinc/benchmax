"""Tests for deterministic, judge-free reward primitives."""

import math

import pytest
from benchmax.rewards import (
    citation_score,
    clip01,
    count_search_calls,
    extract_answer_block,
    extract_completion_text,
    overlap_reward,
    search_within_budget,
    tool_call_efficiency,
)


def test_completion_text_is_the_final_assistant_answer_only():
    transcript = [
        {"role": "assistant", "content": "earlier <tool_call>"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": " final answer "},
    ]
    assert extract_completion_text(transcript) == "final answer"
    assert extract_completion_text(transcript + [{"role": "user", "content": "more"}]) == ""
    assert extract_completion_text(" answer ") == "answer"


def test_answer_block_and_clip():
    assert extract_answer_block("prefix <answer>42</answer>") == "42"
    assert extract_answer_block(" 42 ") == "42"
    assert clip01(-1) == 0
    assert clip01(2) == 1
    assert clip01(float("nan")) == 0


def test_overlap_uses_final_answer_not_transcript_history():
    transcript = [
        {"role": "assistant", "content": "secret reference"},
        {"role": "assistant", "content": "unrelated"},
    ]
    assert overlap_reward(transcript, "secret reference", minimum_overlap=0.9) == 0
    assert overlap_reward("secret reference", "secret reference") == 1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        overlap_reward("a", "a", minimum_overlap=2)


def test_search_call_count_and_budget():
    transcript = [
        {"role": "assistant", "content": "<tool_call><tool_call>"},
        {"role": "tool", "content": "<tool_call>"},
    ]
    assert count_search_calls(transcript) == 2
    assert search_within_budget(2, 2)
    with pytest.raises(ValueError, match="non-negative"):
        search_within_budget(-1, 2)


def test_citation_precision_and_recall():
    score = citation_score(
        "Answer [Source: a] [Source: unknown]",
        [
            {"metadata": {"source_id": "a"}},
            {"metadata": {"source_id": "b"}},
        ],
    )
    assert score == {"precision": 0.5, "recall": 0.5}


def test_tool_call_efficiency_decay_and_ranges():
    completion = "<tool_call>" * 4
    assert tool_call_efficiency(
        completion, correctness=0.5, reference_chunk_count=1, decay_rate=0.2
    ) == pytest.approx(0.5 * math.exp(-0.2))
    assert (
        tool_call_efficiency(
            completion,
            ranges=[(0, 2, 1.0), (3, 5, 0.4)],
        )
        == 0.4
    )
    with pytest.raises(ValueError, match="non-negative"):
        tool_call_efficiency(completion, max_calls=-1)
