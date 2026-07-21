from __future__ import annotations

import pytest

from benchmax.envs.base import BaseRollout
from dominant_color_env import DominantColorEnv


def _first_example(dataset):
    return dataset[0]


@pytest.mark.asyncio
async def test_examples_carry_rules_first_image_and_ordered_answer(tmp_path) -> None:
    env = DominantColorEnv(num_train_examples=4, num_eval_examples=2)
    dataset = await env.create_dataset("train", tmp_path)
    assert len(dataset) == 4

    payload = _first_example(dataset).payload
    system, user = payload["prompt_messages"]
    assert system["role"] == "system"
    assert "exactly 3 images" in system["content"]
    assert "see_next_image" in system["content"]
    assert "dominant color" in system["content"]
    # The full palette is enumerated up front.
    from dominant_color_dataset import PALETTE

    assert len(PALETTE) >= 15
    for name in PALETTE:
        assert name in system["content"]
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][1]["text"] == "This is image 1 of 3."
    assert payload["answer"] == ", ".join(payload["colors"])
    assert len(payload["colors"]) == 3


@pytest.mark.asyncio
async def test_dataset_is_seed_deterministic_and_seed_sensitive(tmp_path) -> None:
    def ids(dataset):
        return [example.id for example in dataset]

    env = DominantColorEnv(num_train_examples=6)
    first = await env.create_dataset("train", tmp_path)
    again = await env.create_dataset("train", tmp_path)
    other_seed = DominantColorEnv(num_train_examples=6, sample_seed=7)
    reseeded = await other_seed.create_dataset("train", tmp_path)

    assert ids(first) == ids(again)
    assert ids(first) != ids(reseeded)
    # Train and eval draw from distinct streams.
    assert ids(first) != ids(await env.create_dataset("eval", tmp_path))


@pytest.mark.asyncio
async def test_see_next_image_walks_the_sequence_then_runs_dry(tmp_path) -> None:
    env = DominantColorEnv(num_train_examples=1)
    dataset = await env.create_dataset("train", tmp_path)
    example = _first_example(dataset)

    async with env.rollout_context("r-1", example):
        second = await env.run_tool("r-1", "see_next_image")
        third = await env.run_tool("r-1", "see_next_image")
        dry = await env.run_tool("r-1", "see_next_image")
    assert "r-1" not in env._sessions  # released with the rollout

    for index, parts in ((2, second), (3, third)):
        assert [part["type"] for part in parts] == ["image_url", "text"]
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert f"This is image {index} of 3." in parts[1]["text"]
    assert "last image" in third[1]["text"]
    assert "last image" not in second[1]["text"]
    assert isinstance(dry, str) and dry.startswith("No more images")

    # Gaussian noise keeps every PNG byte-unique even for repeated colors...
    uris = {
        example.payload["prompt_messages"][1]["content"][0]["image_url"]["url"],
        second[0]["image_url"]["url"],
        third[0]["image_url"]["url"],
    }
    assert len(uris) == 3
    # ...while the same rollout re-renders identically (seeded noise).
    async with env.rollout_context("r-2", example):
        replay = await env.run_tool("r-2", "see_next_image")
    assert replay[0]["image_url"]["url"] == second[0]["image_url"]["url"]


@pytest.mark.asyncio
async def test_reward_is_all_or_nothing_on_exact_order() -> None:
    env = DominantColorEnv()

    async def score(content: str | None) -> float:
        messages = []
        if content is not None:
            messages.append({"role": "assistant", "content": content})
        reward = await env.compute_reward(
            BaseRollout(
                rollout_id="r-1",
                termination_reason="stop",
                messages=messages,
                example_args={"colors": ["red", "green", "blue"]},
            )
        )
        return reward["correctness"]

    assert await score("I saw \\boxed{Red, green, BLUE}") == 1.0
    assert await score("\\boxed{red, blue, green}") == 0.0  # wrong order
    assert await score("\\boxed{red, green}") == 0.0  # answered early
    assert await score("red, green, blue") == 0.0  # no boxed answer
    assert await score(None) == 0.0


@pytest.mark.asyncio
async def test_reward_accepts_grey_spelling_and_word_boundaries() -> None:
    env = DominantColorEnv()
    reward = await env.compute_reward(
        BaseRollout(
            rollout_id="r-1",
            termination_reason="stop",
            messages=[{"role": "assistant", "content": "\\boxed{Grey, navy, olive}"}],
            example_args={"colors": ["gray", "navy", "olive"]},
        )
    )
    assert reward == {"correctness": 1.0}


def test_tile_images_are_dominated_by_the_answer_color() -> None:
    import base64
    import io

    from PIL import Image

    from dominant_color_dataset import PALETTE, render_tile_image_uri

    uri = render_tile_image_uri(
        "teal",
        size=128,
        tile_grid=8,
        dominant_fraction=0.8,
        sigma=12.0,
        seed="tiles:0",
    )
    image = Image.open(io.BytesIO(base64.b64decode(uri.partition("base64,")[2])))

    def classify(tile_x: int, tile_y: int) -> str:
        tile = image.crop(
            (tile_x * 16, tile_y * 16, (tile_x + 1) * 16, (tile_y + 1) * 16)
        )
        pixels = list(tile.getdata())
        mean = [sum(channel) / len(pixels) for channel in zip(*pixels)]
        return min(
            PALETTE,
            key=lambda name: sum((m - c) ** 2 for m, c in zip(mean, PALETTE[name])),
        )

    names = [classify(x, y) for y in range(8) for x in range(8)]
    dominant_share = names.count("teal") / len(names)
    # Exactly 80% by construction; the noisy classification must recover it.
    assert 0.7 <= dominant_share <= 0.9
    assert len(set(names)) > 1  # distractor tiles really are other colors
