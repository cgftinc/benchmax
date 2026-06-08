"""Tests for shared reward helpers."""

from __future__ import annotations

from benchmax.envs.reward_helpers import (
    extract_answer_block,
    extract_completion_text,
    overlap_reward,
)


class TestExtractCompletionText:
    def test_string(self):
        assert extract_completion_text("hello") == "hello"

    def test_message_list(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer one"},
            {"role": "assistant", "content": "answer two"},
        ]
        result = extract_completion_text(msgs)
        assert "answer one" in result
        assert "answer two" in result

    def test_empty_list(self):
        assert extract_completion_text([]) == ""

    def test_non_list_non_string(self):
        assert extract_completion_text(42) == ""  # type: ignore[arg-type]


class TestExtractAnswerBlock:
    def test_with_tags(self):
        assert extract_answer_block("before <answer>the answer</answer> after") == "the answer"

    def test_without_tags(self):
        # No <answer> block → empty string, not the full text. Consumers
        # gate rewards on a non-empty answer.
        assert extract_answer_block("no tags here") == ""

    def test_empty_tags(self):
        assert extract_answer_block("before <answer></answer> after") == ""

    def test_empty(self):
        assert extract_answer_block("") == ""


class TestOverlapReward:
    def test_matching_text(self):
        score = overlap_reward("the reference text is here", "unused", reference_chunks=[
            {"content": "the reference text is here"}
        ])
        assert score > 0.25

    def test_no_match(self):
        score = overlap_reward("completely different", "unused", reference_chunks=[
            {"content": "nothing in common xyz abc"}
        ])
        assert score == 0.0

    def test_below_threshold(self):
        score = overlap_reward("a tiny bit matches", "unused", reference_chunks=[
            {"content": "a" * 1000}
        ])
        assert score == 0.0

    def test_falls_back_to_ground_truth(self):
        score = overlap_reward("the ground truth text", "the ground truth text")
        assert score > 0.25

    def test_tool_call_penalty(self):
        msgs = [
            {"role": "assistant", "content": "<tool_call>" * 4 + "text"},
        ]
        score = overlap_reward(msgs, "", reference_chunks=[{"content": "text"}])
        assert score == 0.0

    def test_empty_reference(self):
        assert overlap_reward("text", "", reference_chunks=[]) == 0.0
