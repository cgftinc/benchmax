"""Tests for the generic trace processing pipeline."""

import pytest
from castform.traces.adapter import (
    NormalizedTrace,
    ToolCall,
    TraceMessage,
)
from castform.traces.processing import (
    VALID_IDENTIFIER_RE,
    TrainingExample,
    apply_filters,
    apply_heuristic_filters,
    apply_score_filter,
    build_training_examples,
    check_outcome_balance,
    deduplicate_completions,
    detect_system_prompt,
    detect_tools,
    filter_by_tool_calls,
    filter_tool_result_relay,
    split_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trace(
    messages: list[TraceMessage],
    trace_id: str = "t1",
    scores: dict | None = None,
    errors: list[str] | None = None,
) -> NormalizedTrace:
    return NormalizedTrace(
        id=trace_id,
        messages=messages,
        scores=scores or {},
        errors=errors,
    )


def _msg(role: str, content: str, **kwargs) -> TraceMessage:
    return TraceMessage(role=role, content=content, **kwargs)


# ---------------------------------------------------------------------------
# detect_system_prompt
# ---------------------------------------------------------------------------


class TestDetectSystemPrompt:
    def test_finds_most_common(self):
        traces = [
            _trace([_msg("system", "Be helpful"), _msg("user", "Q1")], trace_id="t1"),
            _trace([_msg("system", "Be helpful"), _msg("user", "Q2")], trace_id="t2"),
            _trace([_msg("system", "Be concise"), _msg("user", "Q3")], trace_id="t3"),
        ]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.prompt == "Be helpful"
        assert result.count == 2
        assert result.total_traces == 3
        assert result.variants == ["Be concise"]

    def test_returns_none_when_no_system(self):
        traces = [_trace([_msg("user", "Hello")])]
        assert detect_system_prompt(traces) is None

    def test_single_trace_no_variants(self):
        traces = [_trace([_msg("system", "Sys"), _msg("user", "Q")])]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.variants == []

    def test_scans_all_traces_not_just_first(self):
        traces = [
            _trace([_msg("user", "Q1")], trace_id="t1"),  # no system
            _trace([_msg("system", "Found it"), _msg("user", "Q2")], trace_id="t2"),
        ]
        result = detect_system_prompt(traces)
        assert result is not None
        assert result.prompt == "Found it"


# ---------------------------------------------------------------------------
# detect_tools
# ---------------------------------------------------------------------------


class TestDetectTools:
    def test_detects_valid_tools(self):
        traces = [
            _trace(
                [
                    _msg("user", "Q"),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[
                            ToolCall(name="search", arguments='{"q": "test"}'),
                            ToolCall(name="search", arguments='{"q": "other"}'),
                        ],
                    ),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[ToolCall(name="lookup", arguments='{"id": "1"}')],
                    ),
                ]
            )
        ]
        result = detect_tools(traces)
        assert len(result.tools) == 2
        tool_names = {t.name for t in result.tools}
        assert tool_names == {"search", "lookup"}

        search_tool = next(t for t in result.tools if t.name == "search")
        assert search_tool.call_count == 2
        assert "q" in search_tool.param_keys

    def test_drops_invalid_tool_names(self):
        traces = [
            _trace(
                [
                    _msg("user", "Q"),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[
                            ToolCall(name="valid_tool", arguments="{}"),
                            ToolCall(name="invalid name!", arguments="{}"),
                            ToolCall(name="import os\nos.system('bad')", arguments="{}"),
                        ],
                    ),
                ]
            )
        ]
        result = detect_tools(traces)
        assert len(result.tools) == 1
        assert result.tools[0].name == "valid_tool"

    def test_filters_invalid_argument_keys(self):
        traces = [
            _trace(
                [
                    _msg("user", "Q"),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[
                            ToolCall(
                                name="search",
                                arguments='{"valid_key": 1, "bad key!": 2, "also-bad": 3}',
                            ),
                        ],
                    ),
                ]
            )
        ]
        result = detect_tools(traces)
        tool = result.tools[0]
        assert tool.param_keys == {"valid_key"}
        assert tool.sample_args == [{"valid_key": 1}]

    def test_empty_traces(self):
        result = detect_tools([])
        assert result.tools == []

    def test_limits_sample_args_to_3(self):
        traces = [
            _trace(
                [
                    _msg("user", "Q"),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[ToolCall(name="search", arguments=f'{{"q": "test{i}"}}')],
                    ),
                ],
                trace_id=f"t{i}",
            )
            for i in range(10)
        ]
        result = detect_tools(traces)
        search = result.tools[0]
        assert len(search.sample_args) <= 3


# ---------------------------------------------------------------------------
# build_training_examples
# ---------------------------------------------------------------------------


class TestBuildTrainingExamples:
    def test_simple_conversation(self):
        traces = [
            _trace(
                [
                    _msg("system", "Be helpful"),
                    _msg("user", "What is 2+2?"),
                    _msg("assistant", "4"),
                ]
            )
        ]
        # Default: include_system_prompt=False (system prompt comes from BaseEnv)
        examples = build_training_examples(traces)
        assert len(examples) == 1
        ex = examples[0]
        assert ex.trace_id == "t1"
        assert ex.turn_index == 0
        assert len(ex.prompt_messages) == 1  # user only (system excluded by default)
        assert ex.prompt_messages[0].role == "user"
        assert len(ex.completion_messages) == 1  # assistant

    def test_includes_system_prompt_when_requested(self):
        traces = [
            _trace(
                [
                    _msg("system", "Be helpful"),
                    _msg("user", "Q"),
                    _msg("assistant", "A"),
                ]
            )
        ]
        examples = build_training_examples(traces, include_system_prompt=True)
        assert len(examples) == 1
        assert len(examples[0].prompt_messages) == 2  # system + user
        assert examples[0].prompt_messages[0].role == "system"

    def test_multi_turn_cumulative_context(self):
        traces = [
            _trace(
                [
                    _msg("user", "Q1"),
                    _msg("assistant", "A1"),
                    _msg("user", "Q2"),
                    _msg("assistant", "A2"),
                ]
            )
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 2
        # First turn: prompt=[user Q1]
        assert len(examples[0].prompt_messages) == 1
        # Second turn: prompt=[user Q1, assistant A1, user Q2]
        assert len(examples[1].prompt_messages) == 3

    def test_tool_calls_bundled_with_assistant(self):
        traces = [
            _trace(
                [
                    _msg("user", "Search for X"),
                    _msg(
                        "assistant",
                        "",
                        tool_calls=[ToolCall(name="search", arguments='{"q": "X"}')],
                    ),
                    _msg("tool", "Found X", name="search", tool_call_id="tc1"),
                    _msg("assistant", "Here is what I found about X."),
                ]
            )
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 2
        # First completion: assistant + tool response
        assert len(examples[0].completion_messages) == 2
        assert examples[0].completion_messages[0].role == "assistant"
        assert examples[0].completion_messages[1].role == "tool"

    def test_skips_traces_with_errors(self):
        traces = [
            _trace(
                [_msg("user", "Q"), _msg("assistant", "A")],
                trace_id="good",
            ),
            _trace(
                [_msg("user", "Q")],
                trace_id="bad",
                errors=["extraction failed"],
            ),
        ]
        examples = build_training_examples(traces)
        assert len(examples) == 1
        assert examples[0].trace_id == "good"

    def test_skips_empty_traces(self):
        traces = [_trace([], trace_id="empty")]
        examples = build_training_examples(traces)
        assert len(examples) == 0

    def test_no_example_when_no_prompt_before_assistant(self):
        # Edge case: assistant message is the first message
        traces = [_trace([_msg("assistant", "Unprompted response")])]
        examples = build_training_examples(traces)
        assert len(examples) == 0

    def test_inherits_scores_and_metadata(self):
        traces = [
            NormalizedTrace(
                id="t1",
                messages=[_msg("user", "Q"), _msg("assistant", "A")],
                scores={"quality": 0.9},
                metadata={"source": "test"},
            )
        ]
        examples = build_training_examples(traces)
        assert examples[0].scores == {"quality": 0.9}
        assert examples[0].metadata == {"source": "test"}


class TestTrainingExampleSerialization:
    def test_to_jsonl_dict(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="[USER] Q",
            ground_truth="[ASSISTANT] A",
            trace_id="t1",
            turn_index=0,
            scores={"quality": 0.9},
        )
        d = ex.to_jsonl_dict()
        # prompt_messages is structured message dicts
        assert isinstance(d["prompt_messages"], list)
        assert d["prompt_messages"][0]["role"] == "user"
        # ground_truth is the completion message dict
        assert isinstance(d["ground_truth"], dict)
        assert d["ground_truth"]["role"] == "assistant"
        # metadata goes in rollout_args
        assert d["rollout_args"]["trace_id"] == "t1"
        assert d["rollout_args"]["turn_index"] == 0
        assert d["rollout_args"]["scores"] == {"quality": 0.9}
        assert d["rollout_args"]["raw_prompt"] == "[USER] Q"

    def test_to_dict_roundtrip(self):
        ex = TrainingExample(
            prompt_messages=[
                _msg("system", "Be helpful"),
                _msg("user", "What is 2+2?"),
            ],
            completion_messages=[
                TraceMessage(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(name="calc", arguments='{"expr": "2+2"}', id="tc1")],
                ),
            ],
            prompt="[USER] What is 2+2?",
            ground_truth="[ASSISTANT] → calc(2+2)",
            trace_id="t1",
            turn_index=3,
            scores={"quality": 0.9, "task_success": 1.0},
            metadata={"env": "test"},
        )
        d = ex.to_dict()
        restored = TrainingExample.from_dict(d)

        assert restored.trace_id == ex.trace_id
        assert restored.turn_index == ex.turn_index
        assert restored.prompt == ex.prompt
        assert restored.ground_truth == ex.ground_truth
        assert restored.scores == ex.scores
        assert restored.metadata == ex.metadata
        assert len(restored.prompt_messages) == 2
        assert restored.prompt_messages[0].role == "system"
        assert restored.completion_messages[0].tool_calls[0].name == "calc"

    def test_to_dict_json_serializable(self):
        import json

        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
        )
        # Must not raise
        text = json.dumps(ex.to_dict())
        restored = TrainingExample.from_dict(json.loads(text))
        assert restored.trace_id == "t1"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestHeuristicFilters:
    def _make_example(self, ground_truth: str) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", ground_truth)],
            prompt="[USER] Q",
            ground_truth=ground_truth,
            trace_id="t1",
            turn_index=0,
        )

    def test_drops_short_completions(self):
        examples = [
            self._make_example("OK"),
            self._make_example("A" * 100),
        ]
        result = apply_heuristic_filters(examples, min_completion_chars=50)
        assert len(result.kept) == 1
        assert len(result.dropped) == 1
        assert result.dropped[0][1].reason == "too_short"

    def test_summary(self):
        examples = [
            self._make_example("X"),
            self._make_example("Y"),
            self._make_example("A" * 100),
        ]
        result = apply_heuristic_filters(examples, min_completion_chars=50)
        assert result.summary == {"heuristic": 2}


class TestFilterByToolCalls:
    def _make_tool_example(self, tool_name: str, content: str = "") -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[
                TraceMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[ToolCall(name=tool_name, arguments="{}")],
                )
            ],
            prompt="[USER] Q",
            ground_truth=f"tool_call: {tool_name}",
            trace_id="t1",
            turn_index=0,
        )

    def _make_text_example(self, content: str) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", content)],
            prompt="[USER] Q",
            ground_truth=content,
            trace_id="t1",
            turn_index=0,
        )

    def test_drops_excluded_tool_calls(self):
        examples = [
            self._make_tool_example("get_user_details"),
            self._make_tool_example("process_refund"),
        ]
        result = filter_by_tool_calls(examples, ["get_user_details"])
        assert len(result.kept) == 1
        assert len(result.dropped) == 1
        assert result.dropped[0][1].detail == "get_user_details"

    def test_keeps_non_excluded_tools(self):
        examples = [self._make_tool_example("process_refund")]
        result = filter_by_tool_calls(examples, ["get_user_details"])
        assert len(result.kept) == 1
        assert len(result.dropped) == 0

    def test_keeps_examples_without_tool_calls(self):
        examples = [self._make_text_example("Here is a detailed response " * 5)]
        result = filter_by_tool_calls(examples, ["get_user_details"])
        assert len(result.kept) == 1

    def test_keeps_excluded_tool_with_substantive_text(self):
        examples = [self._make_tool_example("get_user_details", "A" * 100)]
        result = filter_by_tool_calls(examples, ["get_user_details"])
        assert len(result.kept) == 1

    def test_empty_exclude_list_keeps_all(self):
        examples = [
            self._make_tool_example("get_user_details"),
            self._make_tool_example("process_refund"),
        ]
        result = filter_by_tool_calls(examples, [])
        assert len(result.kept) == 2

    def test_multiple_tools_all_excluded(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[
                TraceMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(name="get_user", arguments="{}"),
                        ToolCall(name="get_order", arguments="{}"),
                    ],
                )
            ],
            prompt="[USER] Q",
            ground_truth="tools",
            trace_id="t1",
            turn_index=0,
        )
        result = filter_by_tool_calls([ex], ["get_user", "get_order"])
        assert len(result.dropped) == 1

    def test_multiple_tools_partially_excluded(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[
                TraceMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(name="get_user", arguments="{}"),
                        ToolCall(name="process_refund", arguments="{}"),
                    ],
                )
            ],
            prompt="[USER] Q",
            ground_truth="tools",
            trace_id="t1",
            turn_index=0,
        )
        result = filter_by_tool_calls([ex], ["get_user"])
        assert len(result.kept) == 1


class TestScoreFilter:
    def _make_scored_example(self, score: float) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
            scores={"quality": score},
        )

    def test_min_score(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", min_score=0.5)
        assert len(result) == 2

    def test_max_score(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", max_score=0.5)
        assert len(result) == 2

    def test_range(self):
        examples = [self._make_scored_example(s) for s in [0.2, 0.5, 0.8]]
        result = apply_score_filter(examples, "quality", min_score=0.3, max_score=0.6)
        assert len(result) == 1
        assert result[0].scores["quality"] == 0.5

    def test_boundary_min_score_inclusive(self):
        examples = [self._make_scored_example(0.5)]
        result = apply_score_filter(examples, "quality", min_score=0.5)
        assert len(result) == 1  # exactly min_score is kept

    def test_boundary_max_score_inclusive(self):
        examples = [self._make_scored_example(0.5)]
        result = apply_score_filter(examples, "quality", max_score=0.5)
        assert len(result) == 1  # exactly max_score is kept

    def test_missing_score_excluded(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
            scores={},  # no quality score
        )
        result = apply_score_filter([ex], "quality", min_score=0.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def _make_examples(self, n: int) -> list[TrainingExample]:
        return [
            TrainingExample(
                prompt_messages=[_msg("user", f"Q{i}")],
                completion_messages=[_msg("assistant", f"A{i}")],
                prompt=f"Q{i}",
                ground_truth=f"A{i}",
                trace_id=f"t{i}",
                turn_index=0,
            )
            for i in range(n)
        ]

    def test_basic_split(self):
        examples = self._make_examples(20)
        train, eval_ = split_dataset(examples, train_count=16, eval_count=4)
        assert len(train) == 16
        assert len(eval_) == 4

    def test_shuffle_is_deterministic(self):
        examples = self._make_examples(20)
        train1, eval1 = split_dataset(examples, train_count=16, eval_count=4, seed=42)
        train2, eval2 = split_dataset(examples, train_count=16, eval_count=4, seed=42)
        assert train1 == train2
        assert eval1 == eval2

    def test_shuffle_mixes_order(self):
        examples = self._make_examples(20)
        train, _ = split_dataset(examples, train_count=16, eval_count=4)
        # With shuffle, first train example should NOT always be t0
        trace_ids = [row["rollout_args"]["trace_id"] for row in train]
        assert trace_ids != [f"t{i}" for i in range(16)]  # not in original order

    def test_min_train_samples_enforced(self):
        examples = self._make_examples(20)
        with pytest.raises(ValueError, match="at least 16"):
            split_dataset(examples, train_count=10, eval_count=5)

    def test_not_enough_examples(self):
        examples = self._make_examples(10)
        with pytest.raises(ValueError, match="Not enough"):
            split_dataset(examples, train_count=16, eval_count=4)

    def test_jsonl_dict_schema(self):
        examples = self._make_examples(20)
        train, _ = split_dataset(examples, train_count=16, eval_count=4)
        row = train[0]
        assert "prompt_messages" in row
        assert isinstance(row["prompt_messages"], list)  # structured message dicts
        assert "ground_truth" in row
        assert isinstance(row["ground_truth"], dict)  # completion message dict
        assert "rollout_args" in row
        assert "trace_id" in row["rollout_args"]
        assert "turn_index" in row["rollout_args"]


# ---------------------------------------------------------------------------
# Tool result relay filter
# ---------------------------------------------------------------------------


class TestFilterToolResultRelay:
    def _make_relay_example(
        self,
        tool_result: str,
        completion: str,
        has_tool_calls: bool = False,
    ) -> TrainingExample:
        completion_msg = TraceMessage(
            role="assistant",
            content=completion,
            tool_calls=[ToolCall(name="next_step", arguments="{}")] if has_tool_calls else None,
        )
        return TrainingExample(
            prompt_messages=[
                _msg("user", "Check my order"),
                TraceMessage(role="tool", content=tool_result, name="get_order"),
            ],
            completion_messages=[completion_msg],
            prompt="[USER] Check my order",
            ground_truth=completion,
            trace_id="t1",
            turn_index=1,
        )

    def test_detects_relay(self):
        ex = self._make_relay_example(
            "Order ORD-123 shipped March 15 expected delivery April 10",
            "Order ORD-123 shipped March 15 expected delivery April 10 confirmed",
        )
        result = filter_tool_result_relay([ex])
        assert len(result.dropped) == 1
        assert result.dropped[0][1].filter == "tool_relay"

    def test_keeps_non_relay(self):
        ex = self._make_relay_example(
            "Order ORD-123 shipped on March 15",
            "I see your order is on its way. Would you like me to set up a delivery notification?",
        )
        result = filter_tool_result_relay([ex])
        assert len(result.kept) == 1

    def test_keeps_when_no_tool_result(self):
        ex = TrainingExample(
            prompt_messages=[_msg("user", "Hello")],
            completion_messages=[_msg("assistant", "Hi there, how can I help?")],
            prompt="[USER] Hello",
            ground_truth="Hi there",
            trace_id="t1",
            turn_index=0,
        )
        result = filter_tool_result_relay([ex])
        assert len(result.kept) == 1

    def test_keeps_completion_with_tool_calls(self):
        ex = self._make_relay_example(
            "Order ORD-123 shipped",
            "Order ORD-123 shipped",
            has_tool_calls=True,
        )
        result = filter_tool_result_relay([ex])
        assert len(result.kept) == 1

    def test_handles_json_tool_result(self):
        ex = self._make_relay_example(
            '{"order_id": "ORD-123", "status": "shipped", "date": "March 15"}',
            "ORD-123 shipped March 15",
        )
        result = filter_tool_result_relay([ex])
        assert len(result.dropped) == 1

    def test_threshold_boundary(self):
        ex = self._make_relay_example(
            "alpha bravo charlie delta echo foxtrot",
            "alpha bravo charlie completely different content here more words",
        )
        # With threshold 0.9, low overlap should be kept
        result = filter_tool_result_relay([ex], overlap_threshold=0.9)
        assert len(result.kept) == 1

    def test_max_overlap_across_multiple_tool_results(self):
        ex = TrainingExample(
            prompt_messages=[
                _msg("user", "Check both"),
                TraceMessage(role="tool", content="Result alpha bravo charlie", name="tool_a"),
                TraceMessage(role="tool", content="Completely different data here", name="tool_b"),
            ],
            completion_messages=[_msg("assistant", "alpha bravo charlie delta")],
            prompt="prompt",
            ground_truth="gt",
            trace_id="t1",
            turn_index=1,
        )
        result = filter_tool_result_relay([ex])
        assert len(result.dropped) == 1  # matches tool_a


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplicateCompletions:
    def _make_example(self, content: str, trace_id: str = "t1", turn: int = 0) -> TrainingExample:
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", content)],
            prompt="Q",
            ground_truth=content,
            trace_id=trace_id,
            turn_index=turn,
        )

    def test_exact_duplicates_clustered(self):
        examples = [
            self._make_example("Your order has been shipped", f"t{i}", 0) for i in range(10)
        ]
        result = deduplicate_completions(examples, max_per_cluster=3)
        assert len(result.kept) == 3
        assert len(result.dropped) == 7

    def test_near_duplicates_caught(self):
        examples = [
            self._make_example(
                "Your order number 12345 has been shipped successfully today", "t1", 0
            ),
            self._make_example(
                "Your order number 67890 has been shipped successfully today", "t2", 0
            ),
            self._make_example(
                "Your order number 11111 has been shipped successfully today", "t3", 0
            ),
            self._make_example(
                "Your order number 22222 has been shipped successfully today", "t4", 0
            ),
        ]
        result = deduplicate_completions(examples, max_per_cluster=2)
        assert len(result.kept) == 2
        assert len(result.dropped) == 2

    def test_unique_completions_kept(self):
        examples = [
            self._make_example("I will process your refund immediately", "t1", 0),
            self._make_example("Let me check your account settings for you", "t2", 0),
            self._make_example("The exchange has been initiated successfully", "t3", 0),
        ]
        result = deduplicate_completions(examples)
        assert len(result.kept) == 3

    def test_deterministic_regardless_of_input_order(self):
        examples = [self._make_example("Same response here", f"t{i}", 0) for i in range(5)]
        import random

        shuffled = list(examples)
        random.Random(99).shuffle(shuffled)
        r1 = deduplicate_completions(examples, max_per_cluster=2)
        r2 = deduplicate_completions(shuffled, max_per_cluster=2)
        kept_ids_1 = {(e.trace_id, e.turn_index) for e in r1.kept}
        kept_ids_2 = {(e.trace_id, e.turn_index) for e in r2.kept}
        assert kept_ids_1 == kept_ids_2

    def test_drop_reason_has_cluster_id(self):
        examples = [self._make_example("Same thing", f"t{i}", 0) for i in range(5)]
        result = deduplicate_completions(examples, max_per_cluster=1)
        for _, reason in result.dropped:
            assert reason.filter == "dedup"
            assert reason.reason == "duplicate"
            assert reason.detail is not None
            assert reason.detail.startswith("cluster_")

    def test_large_dataset_completes_quickly(self):
        """Regression: dedup must not hang on large datasets with distinct completions."""
        import hashlib
        import time

        examples = [
            self._make_example(
                hashlib.sha256(f"unique-{i}".encode()).hexdigest() * 20,
                f"t{i}",
                0,
            )
            for i in range(2000)
        ]
        start = time.monotonic()
        result = deduplicate_completions(examples)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"dedup took {elapsed:.1f}s on 2000 distinct examples"
        assert len(result.kept) + len(result.dropped) == 2000


# ---------------------------------------------------------------------------
# Outcome balance
# ---------------------------------------------------------------------------


class TestCheckOutcomeBalance:
    def _make_scored_example(self, score: float | None) -> TrainingExample:
        scores = {"task_success": score} if score is not None else {}
        return TrainingExample(
            prompt_messages=[_msg("user", "Q")],
            completion_messages=[_msg("assistant", "A")],
            prompt="Q",
            ground_truth="A",
            trace_id="t1",
            turn_index=0,
            scores=scores,
        )

    def test_balanced_dataset(self):
        examples = [self._make_scored_example(1.0)] * 7 + [self._make_scored_example(0.0)] * 3
        balance = check_outcome_balance(examples)
        assert balance.is_balanced
        assert balance.message is None
        assert balance.success_count == 7
        assert balance.failure_count == 3

    def test_imbalanced_dataset(self):
        examples = [self._make_scored_example(1.0)] * 19 + [self._make_scored_example(0.0)] * 1
        balance = check_outcome_balance(examples)
        assert not balance.is_balanced
        assert balance.message is not None
        assert "5.0%" in balance.message

    def test_missing_scores(self):
        examples = [self._make_scored_example(None)] * 5
        balance = check_outcome_balance(examples)
        assert balance.unknown_count == 5
        assert balance.failure_fraction == 0.0
        assert balance.is_balanced is True
        assert balance.message is not None
        assert "not applicable" in balance.message

    def test_custom_threshold(self):
        examples = [self._make_scored_example(0.8)] * 8 + [self._make_scored_example(0.3)] * 2
        balance = check_outcome_balance(examples, success_threshold=0.5)
        assert balance.success_count == 8
        assert balance.failure_count == 2


# ---------------------------------------------------------------------------
# Filter composition
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def _make_examples(self, n: int, content_len: int = 100) -> list[TrainingExample]:
        return [
            TrainingExample(
                prompt_messages=[_msg("user", f"Q{i}")],
                completion_messages=[_msg("assistant", f"A{i}" + "x" * content_len)],
                prompt=f"Q{i}",
                ground_truth=f"A{i}",
                trace_id=f"t{i}",
                turn_index=0,
            )
            for i in range(n)
        ]

    def test_accumulates_drops(self):
        examples = [
            TrainingExample(
                prompt_messages=[_msg("user", "Q")],
                completion_messages=[_msg("assistant", "OK")],  # too short
                prompt="Q",
                ground_truth="OK",
                trace_id="t0",
                turn_index=0,
            ),
        ] + self._make_examples(5)

        result = apply_filters(examples, [("heuristic", {"min_completion_chars": 50})])
        assert len(result.dropped) == 1
        assert result.summary == {"heuristic": 1}

    def test_multiple_stages(self):
        examples = [
            TrainingExample(
                prompt_messages=[_msg("user", "Q")],
                completion_messages=[_msg("assistant", "OK")],  # 2 chars, dropped by heuristic
                prompt="Q",
                ground_truth="OK",
                trace_id="t0",
                turn_index=0,
            ),
            TrainingExample(
                prompt_messages=[_msg("user", "Q")],
                completion_messages=[
                    # 15 chars, survives heuristic(10)
                    TraceMessage(
                        role="assistant",
                        content="Looking up user",
                        tool_calls=[ToolCall(name="get_user", arguments="{}")],
                    )
                ],
                prompt="Q",
                ground_truth="tool",
                trace_id="t1",
                turn_index=0,
            ),
        ] + self._make_examples(3, content_len=100)

        result = apply_filters(
            examples,
            [
                ("heuristic", {"min_completion_chars": 10}),
                ("tool_calls", {"exclude_tools": ["get_user"]}),
            ],
        )
        assert result.summary == {"heuristic": 1, "tool_calls": 1}
        assert len(result.kept) == 3

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            apply_filters([], [("nonexistent", {})])

    def test_rejects_bad_ordering(self):
        with pytest.raises(ValueError, match="Per-example filter.*after dataset-level"):
            apply_filters(
                [],
                [
                    ("dedup", {}),
                    ("heuristic", {"min_completion_chars": 50}),
                ],
            )


class TestValidIdentifierRegex:
    def test_valid_names(self):
        for name in ["search", "lookup_tool", "_private", "Tool1"]:
            assert VALID_IDENTIFIER_RE.match(name), f"{name} should be valid"

    def test_invalid_names(self):
        for name in ["123start", "has space", "has-dash", "import\nos", ""]:
            assert not VALID_IDENTIFIER_RE.match(name), f"{name!r} should be invalid"
