"""Integration smoke for the faithful server-side compute_group_reward check.

Drives validate_env(local=False) -> run_group -> /v1/rollout/batch/stream, so the
env's compute_group_reward runs in a real sandbox worker over a co-located group:
  GOOD env   -> valid group reward -> group_reward.ok is True
  BROKEN env -> raises -> rollout-service reports group_reward_error -> ok is False

Requires a staging deploy with the server side live (rollout-service +
platform-service; PR castform-ai/castform#136). Credentials: set PLATFORM_API_KEY
(and CASTFORM_BASE_DOMAIN / CASTFORM_PLATFORM_URL+CASTFORM_LLM_URL if not the
default castform.dev), or load via .env.test. Skipped in CI; run locally:

    PLATFORM_API_KEY=... uv run pytest -m integration tests/integration/test_group_reward_smoke.py
"""

from __future__ import annotations

import os
import sys

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example
from benchmax.platform import validate_env

pytestmark = pytest.mark.integration

os.environ.setdefault("CASTFORM_PLATFORM_URL", "https://api.castform.dev")
os.environ.setdefault("CASTFORM_LLM_URL", "https://llm.castform.dev/v1")


class _GroupEnvBase(BaseEnv):
    """Minimal env (no tools/corpus/judge) — a trivial per-sample reward so the
    only interesting signal is compute_group_reward."""

    system_prompt = "You are a helpful assistant. Answer in one short sentence."

    @classmethod
    def dataset_preprocess(cls, example, **kwargs) -> Example:
        return make_example(
            prompt_messages=[{"role": "user", "content": example.get("prompt", "")}],
            task={"ground_truth": example.get("ground_truth", "")},
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self):
        return []

    async def run_tool(self, rollout_id, tool_name, **kwargs):
        raise NotImplementedError

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        return {"nonempty": 1.0}


class GoodGroupEnv(_GroupEnvBase):
    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        return [{"group_bonus": 0.25} for _ in rollout_ids]


class BrokenGroupEnv(_GroupEnvBase):
    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        raise ValueError("intentional group-reward failure (integration smoke)")


def _validate(env_cls: type):
    api_key = os.environ.get("PLATFORM_API_KEY")
    if not api_key:
        pytest.skip("PLATFORM_API_KEY not set")
    extra = {}
    model = os.environ.get("GROUP_REWARD_TEST_MODEL")
    if model:
        extra["llm_model"] = model
    return validate_env(
        env_class=env_cls,
        env_args={},
        train_dataset=[{"prompt": "Say hello in one word.", "ground_truth": "hello"}],
        api_key=api_key,
        local=False,  # remote-only: exercise the real server-side group path
        group_reward_samples=2,
        pip_dependencies=[],
        # Pickle this module's env classes by value (they live in a named module,
        # not __main__), so the bundle is self-contained for the sandbox.
        local_modules=[sys.modules[__name__]],
        verbose=False,
        **extra,
    )


def test_group_reward_good_passes():
    report = _validate(GoodGroupEnv)
    assert report.remote is not None
    gr = report.remote.group_reward
    assert gr is not None, "group reward check did not run (proxy not deployed?)"
    assert gr.ok is True, f"expected pass, got {gr}"


def test_group_reward_broken_surfaces_error():
    report = _validate(BrokenGroupEnv)
    assert report.remote is not None
    gr = report.remote.group_reward
    assert gr is not None, "group reward check did not run (proxy not deployed?)"
    assert gr.ok is False, f"expected fail, got {gr}"
    assert "group-reward failure" in (gr.error or "")
