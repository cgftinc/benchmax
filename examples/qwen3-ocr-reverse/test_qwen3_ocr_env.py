"""Unit tests for Qwen3OCREnv and Infinity-Doc OCR reward."""

from __future__ import annotations

import asyncio
from pathlib import Path

from benchmax.envs import BaseRollout
from qwen3_ocr_env import Qwen3OCREnv
from qwen3_ocr_env import OCR_PROMPT_TEMPLATE
from qwen3_ocr_reward import infinity_doc_reward, segments


def test_preprocess_builds_openai_image_parts() -> None:
    example = Qwen3OCREnv.dataset_preprocess(
        {
            "prompt": "Find x.",
            "images": ["https://example.com/document.png"],
            "answer": "12",
        }
    )

    assert len(example.payload["prompt_messages"]) == 1
    user = example.payload["prompt_messages"][0]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.com/document.png"}},
        {"type": "text", "text": "Find x."},
    ]
    assert example.payload["answer"] == "12"


def test_preprocess_does_not_prepend_system_prompt() -> None:
    dataset_prompt = "Document Parsing: convert the image to markdown."
    example = Qwen3OCREnv.dataset_preprocess(
        {
            "prompt": dataset_prompt,
            "images": ["https://example.com/doc.png"],
            "answer": "# Title",
        }
    )

    assert Qwen3OCREnv.system_prompt == ""
    assert example.payload["prompt_messages"][0]["content"][-1] == {
        "type": "text",
        "text": dataset_prompt,
    }


def test_preprocess_uses_arrow_safe_content_shape() -> None:
    example = Qwen3OCREnv.dataset_preprocess(
        {
            "prompt": "Find x.",
            "images": ["https://example.com/document.png"],
            "answer": "12",
        }
    )
    assert {
        type(message["content"])
        for message in example.payload["prompt_messages"]
    } == {list}


def test_reward_empty_ref_and_pred_is_three() -> None:
    score = infinity_doc_reward(
        [{"role": "assistant", "content": ""}],
        {"answer": ""},
    )
    assert score == 1.0


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
    assert score >= 2.5 / 3.0


def test_segments_html_table_rows() -> None:
    html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    assert segments(html) == ["A | B", "C | D"]


def test_segments_markdown_table_rows() -> None:
    md = "| Name | Age |\n| --- | --- |\n| Ada | 36 |"
    assert segments(md) == ["Name | Age", "Ada | 36"]


def test_env_compute_reward_and_lifecycle() -> None:
    env = Qwen3OCREnv()

    async def _run() -> None:
        reward = await env.compute_reward(
            BaseRollout(
                rollout_id="rid",
                termination_reason="finished",
                messages=[{"role": "assistant", "content": "Hello"}],
                example_args={"answer": "Hello"},
            )
        )
        assert "answer_correct" in reward
        assert reward["answer_correct"] >= 1.0 / 3.0
        assert await env.list_tools() == []
        await env.aclose()

    asyncio.run(_run())


def test_convert_infinity_doc_rows_writes_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INFINITY_DOC_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("INFINITY_DOC_DATASET", "infly/Infinity-Doc-55K")
    monkeypatch.setenv("INFINITY_DOC_SPLIT", "train")

    class FakeImage:
        def convert(self, mode: str) -> "FakeImage":
            assert mode == "RGB"
            return self

        def save(self, path: Path, format: str = "PNG") -> None:
            assert format == "PNG"
            Path(path).write_bytes(b"png")

    rows = Qwen3OCREnv._convert_infinity_doc_rows(
        [{"id": "doc-1", "image": FakeImage(), "gt": "# Title", "attributes": {"lang": "en"}}],
        "train_images",
        [7],
    )

    assert len(rows) == 1
    assert rows[0]["prompt"] == OCR_PROMPT_TEMPLATE
    assert rows[0]["answer"] == "# Title"
    assert rows[0]["metadata"]["source_index_after_shuffle"] == 7
    image_path = Path(rows[0]["images"][0])
    assert image_path.exists()
    assert image_path.parent == tmp_path / "train_images"


def test_preprocess_falls_back_to_ocr_prompt_template() -> None:
    example = Qwen3OCREnv.dataset_preprocess(
        {
            "images": ["https://example.com/doc.png"],
            "answer": "# Title",
        }
    )
    assert example.payload["prompt_messages"][0]["content"][-1] == {
        "type": "text",
        "text": OCR_PROMPT_TEMPLATE,
    }
