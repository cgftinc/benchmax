from __future__ import annotations

import pytest

from benchmax.envs.base import BaseRollout
from geo3k_dataset import Geo3KDataset
from geo3k_env import Geo3KEnv


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
                        "image_url": {
                            "url": "data:image/png;base64,aW1hZ2U="
                        },
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
