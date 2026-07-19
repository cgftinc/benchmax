import pytest

from benchmax.rubrics._utils import (
    _extract_completion_text,
    _extract_json,
    _static_rubric_key,
    _zero_rubric_result,
)


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"score": 1}') == {"score": 1}

    def test_markdown_fence(self):
        assert _extract_json('```json\n{"score": 1}\n```') == {"score": 1}

    def test_embedded_in_text(self):
        assert _extract_json('Sure! {"score": 0, "reasoning": "x"} done.') == {
            "score": 0,
            "reasoning": "x",
        }

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _extract_json("not json at all")


class TestExtractCompletionText:
    def test_str_passthrough(self):
        assert _extract_completion_text("  hi  ") == "hi"

    def test_message_list_assistant_last(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": " a "}]
        assert _extract_completion_text(msgs) == "a"

    def test_message_list_non_assistant_last_is_empty(self):
        msgs = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "q"}]
        assert _extract_completion_text(msgs) == ""

    def test_empty_list_is_empty(self):
        assert _extract_completion_text([]) == ""


class TestStaticRubricKey:
    def test_snake_case(self):
        assert _static_rubric_key("Edge Case Handling") == "rubric_edge_case_handling"

    def test_strips_punctuation_and_collapses(self):
        assert _static_rubric_key("  Foo!! Bar??  ") == "rubric_foo_bar"


@pytest.mark.asyncio
async def test_zero_rubric_result():
    assert await _zero_rubric_result() == {
        "score": 0,
        "reasoning": "Empty response",
        "llm_output": "",
    }
