from __future__ import annotations

import pytest
from benchmax.envs.base import BaseRollout
from main import Geo3KDataset, Geo3KEnv


@pytest.mark.asyncio
async def test_geo3k_example_is_openai_multimodal_and_scores_boxed_answer() -> None:
    dataset = Geo3KDataset(
        [
            {
                "problem": "<image> Find x.",
                "answer": "42",
                "images": ["data:image/None;base64,aW1hZ2U="],
            }
        ]
    )
    example = dataset[0]

    assert example.payload == {
        "prompt_messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                    },
                    {"type": "text", "text": "Find x."},
                ],
            }
        ],
        "answer": "42",
    }
    reward = await Geo3KEnv().compute_reward(
        BaseRollout(
            rollout_id="rollout-1",
            termination_reason="stop",
            messages=[
                *example.payload["prompt_messages"],
                {
                    "role": "assistant",
                    "content": "Reasoning. Answer: \\boxed{42}",
                },
            ],
            example_args={"answer": "42"},
        )
    )

    assert reward == {"correctness": 1.0}


@pytest.mark.asyncio
async def test_capped_subsets_sample_the_shuffled_split(monkeypatch, tmp_path) -> None:
    """Caps must sample the whole split deterministically, not its head."""

    import main as env_module
    from datasets import Dataset as HFDataset

    rows = [
        {
            "problem": f"<image> Problem {index}.",
            "answer": str(index),
            "images": ["data:image/png;base64,aW1hZ2U="],
        }
        for index in range(50)
    ]
    monkeypatch.setattr(
        env_module,
        "_load_rows",
        lambda name, *, split, cache_dir: HFDataset.from_list(rows),
    )

    env = Geo3KEnv(max_train_examples=8)
    first = await env.create_dataset("train", tmp_path)
    again = await env.create_dataset("train", tmp_path)
    other_seed = Geo3KEnv(max_train_examples=8, sample_seed=7)
    reshuffled = await other_seed.create_dataset("train", tmp_path)

    def answers(dataset):
        return [example.payload["answer"] for example in dataset]

    assert len(first) == 8
    # Deterministic for a seed, not the head of the split, seed-sensitive.
    assert answers(first) == answers(again)
    assert answers(first) != [str(i) for i in range(8)]
    assert answers(first) != answers(reshuffled)


@pytest.mark.asyncio
async def test_zoom_tool_returns_an_upscaled_crop_as_content_parts(tmp_path) -> None:
    import base64
    import io

    from benchmax.envs.shared_types import Example
    from main import Geo3KEnv
    from PIL import Image

    image = Image.new("RGB", (200, 100), "white")
    image.putpixel((190, 90), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    env = Geo3KEnv()
    payload = {
        "prompt_messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "Find x."},
                ],
            }
        ],
        "answer": "42",
    }
    example = Example(id="zoom-example", payload=payload)

    async with env.rollout_context("r-1", example):
        parts = await env.run_tool("r-1", "zoom", x0=0.5, y0=0.5, x1=1.0, y1=1.0)
        degenerate = await env.run_tool("r-1", "zoom", x0=0.5, y0=0.5, x1=0.5, y1=0.5)
    assert "r-1" not in env._images  # released with the rollout

    assert isinstance(degenerate, str) and degenerate.startswith("Error:")
    assert [part["type"] for part in parts] == ["image_url", "text"]
    encoded = parts[0]["image_url"]["url"].partition("base64,")[2]
    crop = Image.open(io.BytesIO(base64.b64decode(encoded)))
    # Quarter crop upscaled back to the original long edge.
    assert crop.size == (200, 100)
    assert "x0=0.50" in parts[1]["text"]
