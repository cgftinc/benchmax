from __future__ import annotations

from types import SimpleNamespace

import benchmax.envs.base.env as base_env_module
import pytest
from benchmax.envs import RolloutRequest, StaticBearerAuth
from benchmax.rewards import JudgeError
from main import TelestichEnv, _bucket, _count_tool_calls


async def test_two_rollout_group_uses_native_benchmax_contract(monkeypatch) -> None:
    poems = {
        "poem-1": """<answer>
The rain lies shining on the road
A final chord falls from the piano
The dark begins to sing a song
</answer>""",
        "poem-2": """<answer>
Night folds softly into wind
An empty station keeps its echo
Blue windows wake into morning
</answer>""",
    }

    async def fake_completion(*, request, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=poems[request.rollout_id],
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
        )

    env = TelestichEnv(judge_base_url="https://judge.example/v1")

    async def assign_quality(prompt, rolls, acceptable, great) -> None:
        assert prompt == "Write a quiet telestich whose last letters spell dog."
        assert acceptable is None
        assert great is None
        for score, rollout in zip((0.6, 0.5), rolls, strict=True):
            rollout.quality = score
            rollout.q = score
            rollout.bucket = _bucket(score)

    monkeypatch.setattr(base_env_module, "_create_chat_completion", fake_completion)
    monkeypatch.setattr(env, "_score_quality", assign_quality)

    example = env._example_from_row(
        {
            "prompt": "Write a quiet telestich whose last letters spell dog.",
            "ground_truth": "dog",
            "acceptable_refs": [],
            "great_refs": [],
        }
    )
    requests = [
        RolloutRequest(
            rollout_id=rollout_id,
            example=example,
            model="test-model",
            base_url="https://model.example/v1",
            model_auth=StaticBearerAuth("test-token"),
        )
        for rollout_id in poems
    ]

    outcomes = await env.run_group(requests)

    assert set(outcomes) == set(poems)
    assert all(outcome.termination_reason == "finished" for outcome in outcomes.values())
    assert outcomes["poem-1"].rewards["quality"] == pytest.approx(0.6)
    assert outcomes["poem-2"].rewards["quality"] == pytest.approx(0.5)
    assert all(
        set(outcome.rewards) == {"quality", "rhyme", "diversity", "conciseness"}
        for outcome in outcomes.values()
    )
    assert all(sum(outcome.rewards.values()) > 0 for outcome in outcomes.values())


async def test_group_judge_failure_zeroes_every_sibling(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poems = {
        "poem-1": """<answer>
The rain lies shining on the road
A final chord falls from the piano
The dark begins to sing a song
</answer>""",
        "poem-2": """<answer>
Night folds softly into wind
An empty station keeps its echo
Blue windows wake into morning
</answer>""",
    }

    async def fake_completion(*, request, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=poems[request.rollout_id],
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
        )

    async def fail_judge(*args, **kwargs):
        del args, kwargs
        raise JudgeError("ranking service unavailable")

    env = TelestichEnv(judge_base_url="https://judge.example/v1")
    monkeypatch.setattr(base_env_module, "_create_chat_completion", fake_completion)
    monkeypatch.setattr(env, "_score_quality", fail_judge)
    example = env._example_from_row(
        {
            "prompt": "Write a telestich whose last letters spell dog.",
            "ground_truth": "dog",
            "acceptable_refs": [],
            "great_refs": [],
        }
    )
    requests = [
        RolloutRequest(
            rollout_id=rollout_id,
            example=example,
            model="test-model",
            base_url="https://model.example/v1",
            model_auth=StaticBearerAuth("test-token"),
        )
        for rollout_id in poems
    ]

    outcomes = await env.run_group(requests)

    assert all(outcome.termination_reason == "judge_error" for outcome in outcomes.values())
    assert all(
        outcome.rewards
        == {
            "quality": 0.0,
            "rhyme": 0.0,
            "diversity": 0.0,
            "conciseness": 0.0,
        }
        for outcome in outcomes.values()
    )
    assert "ranking service unavailable" in caplog.text


def test_tool_call_count_reads_structured_base_rollout_messages() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {}},
                {"id": "call-2", "type": "function", "function": {}},
            ],
        },
        {"role": "tool", "content": "feedback", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "<answer>done</answer>"},
    ]

    assert _count_tool_calls(messages) == 2
