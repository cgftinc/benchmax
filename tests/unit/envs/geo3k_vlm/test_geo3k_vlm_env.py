"""Unit tests for Geo3KVLMEnv and Infinity-Doc OCR reward."""

from __future__ import annotations

import asyncio

from benchmax.envs.geo3k_vlm import Geo3KVLMEnv
from benchmax.envs.geo3k_vlm.reward_fn import infinity_doc_reward, segments


def test_preprocess_builds_openai_image_parts() -> None:
    example = Geo3KVLMEnv.dataset_preprocess(
        {
            "prompt": "Find x.",
            "images": ["https://example.com/geo.png"],
            "answer": "12",
        }
    )

    assert len(example["prompt_messages"]) == 1
    user = example["prompt_messages"][0]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.com/geo.png"}},
        {"type": "text", "text": "Find x."},
    ]
    assert example["task"]["answer"] == "12"


def test_preprocess_does_not_prepend_system_prompt() -> None:
    dataset_prompt = "Document Parsing: convert the image to markdown."
    example = Geo3KVLMEnv.dataset_preprocess(
        {
            "prompt": dataset_prompt,
            "images": ["https://example.com/doc.png"],
            "answer": "# Title",
        }
    )

    assert Geo3KVLMEnv.system_prompt == ""
    assert example["prompt_messages"][0]["content"][-1] == {
        "type": "text",
        "text": dataset_prompt,
    }


def test_preprocess_uses_arrow_safe_content_shape() -> None:
    example = Geo3KVLMEnv.dataset_preprocess(
        {
            "prompt": "Find x.",
            "images": ["https://example.com/geo.png"],
            "answer": "12",
        }
    )
    assert {type(message["content"]) for message in example["prompt_messages"]} == {list}


def test_reward_empty_ref_and_pred_is_three() -> None:
    score = infinity_doc_reward(
        [{"role": "assistant", "content": ""}],
        {"answer": ""},
    )
    assert score == 3.0


def test_reward_empty_prediction_is_zero() -> None:
    score = infinity_doc_reward(
        [{"role": "assistant", "content": ""}],
        {"answer": "Hello world"},
    )
    assert score == 0.0


def test_reward_exact_match_is_high() -> None:
    text = "Hello world\n\nSecond paragraph"
    score = infinity_doc_reward(
        [{"role": "assistant", "content": text}],
        {"answer": text},
    )
    assert score >= 2.5


def test_segments_html_table_rows() -> None:
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    assert segments(html) == ["A | B", "C | D"]


def test_segments_markdown_table_rows() -> None:
    md = "| Name | Age |\n| --- | --- |\n| Ada | 36 |"
    assert segments(md) == ["Name | Age", "Ada | 36"]


def test_env_compute_reward_and_lifecycle() -> None:
    env = Geo3KVLMEnv()

    async def _run() -> None:
        await env.init_rollout("rid")
        reward = await env.compute_reward(
            "rid",
            [{"role": "assistant", "content": "Hello"}],
            {"answer": "Hello"},
        )
        assert "answer_correct" in reward
        assert reward["answer_correct"] >= 1.0
        assert await env.list_tools() == []
        await env.release_rollout("rid")
        await env.shutdown()

    asyncio.run(_run())
