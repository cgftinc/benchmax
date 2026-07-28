"""Tests for Braintrust message extraction strategies."""


from castform.traces.braintrust.message_extraction import (
    extract_messages,
    extract_scores,
)


class TestStrategy1_RootInputHasFullConversation:  # noqa: N801 — strategy number, then description
    """When root.input.messages already contains assistant messages."""

    def test_full_conversation_in_root(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        assert len(msgs) == 3
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"
        assert msgs[2].role == "assistant"
        assert msgs[2].content == "4"

    def test_with_tool_calls_in_root(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Search for revenue"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "revenue"}',
                                },
                                "id": "tc1",
                            }
                        ],
                    },
                    {"role": "tool", "content": "Revenue was $1M", "tool_call_id": "tc1"},
                    {"role": "assistant", "content": "Revenue was $1M."},
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        assert len(msgs) == 4
        assert msgs[1].tool_calls is not None
        assert msgs[1].tool_calls[0].name == "search"
        assert msgs[2].role == "tool"


class TestStrategy2_ReconstructFromChildren:  # noqa: N801 — strategy number, then description
    """When root.input has only user message, conversation is in children."""

    def test_reconstruct_from_llm_child(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                ]
            },
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {
                        "messages": [
                            {"role": "system", "content": "Be helpful"},
                            {"role": "user", "content": "Hello"},
                        ]
                    },
                    "output": {"role": "assistant", "content": "Hi there!"},
                },
            ],
        }
        msgs = extract_messages(trace)
        assert any(m.role == "assistant" for m in msgs)
        assert any(m.content == "Hi there!" for m in msgs)

    def test_picks_llm_child_with_most_messages(self):
        trace = {
            "input": {"messages": [{"role": "user", "content": "Q"}]},
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {"messages": [{"role": "user", "content": "Q"}]},
                    "output": {"role": "assistant", "content": "short"},
                },
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:02Z",
                    "input": {
                        "messages": [
                            {"role": "system", "content": "Be helpful"},
                            {"role": "user", "content": "Q"},
                        ]
                    },
                    "output": {"role": "assistant", "content": "detailed answer"},
                },
            ],
        }
        msgs = extract_messages(trace)
        # Should pick the child with 2 input messages (system + user)
        assert any(m.content == "Be helpful" for m in msgs)

    def test_tool_children_become_tool_responses(self):
        trace = {
            "input": {"messages": [{"role": "user", "content": "Search docs"}]},
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {
                        "messages": [{"role": "user", "content": "Search docs"}]
                    },
                    "output": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "search", "arguments": "{}"}, "id": "tc1"}
                        ],
                    },
                },
                {
                    "span_attributes": {"type": "tool"},
                    "created": "2024-01-01T00:00:02Z",
                    "name": "search",
                    "input": {},
                    "output": "Found 3 documents",
                    "span_id": "tc1",
                },
            ],
        }
        msgs = extract_messages(trace)
        tool_msgs = [m for m in msgs if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "Found 3 documents"

    def test_children_sorted_by_timestamp(self):
        trace = {
            "input": {"messages": [{"role": "user", "content": "Q"}]},
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:03Z",  # later
                    "input": {"messages": [{"role": "user", "content": "Q"}]},
                    "output": {"role": "assistant", "content": "second"},
                },
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",  # earlier
                    "input": {
                        "messages": [
                            {"role": "system", "content": "S"},
                            {"role": "user", "content": "Q"},
                        ]
                    },
                    "output": {"role": "assistant", "content": "first"},
                },
            ],
        }
        msgs = extract_messages(trace)
        # The earlier child (with more messages) should be picked
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        assert assistant_msgs[0].content == "first"


class TestStrategy3_Fallback:  # noqa: N801 — strategy number, then description
    """When neither Strategy 1 nor 2 produces assistant messages."""

    def test_fallback_root_plus_outputs(self):
        trace = {
            "input": {
                "messages": [{"role": "user", "content": "Hello"}]
            },
            "children": [
                {
                    "span_attributes": {"type": "custom"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {},
                    "output": "Some output text",
                },
            ],
        }
        msgs = extract_messages(trace)
        assert len(msgs) >= 1
        # Should have user message + assistant message from output
        assert msgs[0].role == "user"


class TestToolCallNormalization:
    """Test the various tool_calls formats Braintrust uses."""

    def test_nested_function_format(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {"name": "search", "arguments": '{"q": "x"}'},
                                "id": "tc1",
                            }
                        ],
                    },
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        tc = msgs[1].tool_calls[0]
        assert tc.name == "search"
        assert tc.id == "tc1"

    def test_direct_format(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"name": "lookup", "arguments": "{}", "id": "tc2"}
                        ],
                    },
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        tc = msgs[1].tool_calls[0]
        assert tc.name == "lookup"

    def test_old_function_format(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "calling",
                        "function": {"name": "old_tool", "arguments": "{}"},
                    },
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        tc = msgs[1].tool_calls[0]
        assert tc.name == "old_tool"

    def test_dict_arguments_get_serialized(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"name": "search", "arguments": {"query": "test"}, "id": "tc1"}
                        ],
                    },
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        tc = msgs[1].tool_calls[0]
        assert tc.arguments == '{"query": "test"}'


class TestExtractScores:
    def test_dict_scores(self):
        scores = extract_scores({"scores": {"accuracy": 0.9, "speed": 0.5}})
        assert scores == {"accuracy": 0.9, "speed": 0.5}

    def test_list_scores(self):
        scores = extract_scores({
            "scores": [
                {"name": "task_success", "score": 1.0},
                {"name": "efficiency", "score": 0.7},
            ]
        })
        assert scores == {"task_success": 1.0, "efficiency": 0.7}

    def test_empty_scores(self):
        assert extract_scores({}) == {}
        assert extract_scores({"scores": None}) == {}

    def test_invalid_score_values_skipped(self):
        scores = extract_scores({"scores": {"good": 0.9, "bad": "not a number"}})
        assert scores == {"good": 0.9}


class TestDuplicateToolResponseDedup:
    """Regression: subsequent tool spans whose output is already in the best
    LLM child's input messages should not be appended again."""

    def test_trailing_tool_span_already_consumed_by_llm_input(self):
        """When the best LLM child's input already contains a tool response,
        a later tool span with the same content should be skipped."""
        trace = {
            "input": {"messages": [{"role": "user", "content": "Modify my order"}]},
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {
                        "messages": [
                            {"role": "user", "content": "Modify my order"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "modify_order",
                                            "arguments": '{"id": "W123"}',
                                        },
                                        "id": "tc1",
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "content": '{"status": "modified"}',
                                "tool_call_id": "tc1",
                                "name": "modify_order",
                            },
                        ]
                    },
                    "output": {
                        "role": "assistant",
                        "content": "Your order has been modified.",
                    },
                },
                # This tool span has the same content as what's already
                # in the LLM child's input — should be skipped
                {
                    "span_attributes": {"type": "tool"},
                    "created": "2024-01-01T00:00:01Z",
                    "name": "modify_order",
                    "input": {},
                    "output": '{"status": "modified"}',
                    "span_id": "tc1",
                },
            ],
        }
        msgs = extract_messages(trace)
        tool_msgs = [m for m in msgs if m.role == "tool"]
        # Should only have ONE tool message — the one from the LLM's input
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == '{"status": "modified"}'

        # Last message should be the assistant's final answer, not a tool response
        assert msgs[-1].role == "assistant"
        assert msgs[-1].content == "Your order has been modified."

    def test_genuinely_new_tool_response_still_appended(self):
        """A subsequent tool span with NEW content (not in the best LLM
        child's input) should still be appended."""
        trace = {
            "input": {"messages": [{"role": "user", "content": "Do two things"}]},
            "children": [
                {
                    "span_attributes": {"type": "llm"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {
                        "messages": [
                            {"role": "user", "content": "Do two things"},
                        ]
                    },
                    "output": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {"name": "step1", "arguments": "{}"},
                                "id": "tc1",
                            }
                        ],
                    },
                },
                {
                    "span_attributes": {"type": "tool"},
                    "created": "2024-01-01T00:00:02Z",
                    "name": "step1",
                    "input": {},
                    "output": "step1 result — new content",
                    "span_id": "tc1",
                },
            ],
        }
        msgs = extract_messages(trace)
        tool_msgs = [m for m in msgs if m.role == "tool"]
        # This tool content is NOT in the LLM's input, so it should be appended
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "step1 result — new content"


class TestEdgeCases:
    def test_empty_trace(self):
        msgs = extract_messages({})
        assert msgs == []

    def test_none_content_normalized(self):
        trace = {
            "input": {
                "messages": [
                    {"role": "user", "content": None},
                    {"role": "assistant", "content": "response"},
                ]
            },
            "children": [],
        }
        msgs = extract_messages(trace)
        assert msgs[0].content == ""

    def test_string_output(self):
        trace = {
            "input": {"messages": [{"role": "user", "content": "Q"}]},
            "children": [
                {
                    "span_attributes": {"type": "custom"},
                    "created": "2024-01-01T00:00:01Z",
                    "input": {},
                    "output": "Plain string output",
                },
            ],
        }
        msgs = extract_messages(trace)
        assert any(m.content == "Plain string output" for m in msgs)
