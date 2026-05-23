"""Tests for LinkerEnv — search environment for LLM-driven chunk linking."""

from __future__ import annotations

from typing import Any

from benchmax.envs.postgres_search.linker_env import _SYSTEM_PROMPT, LinkerEnv


class StubSearch:
    """Minimal SearchClient for testing."""

    def __init__(self, modes: list[str] | None = None):
        self._modes = modes or ["vector"]

    def search(
        self, query: str, mode: str = "auto", top_k: int = 10
    ) -> list[dict[str, Any]]:
        return [{"content": "result", "source": "doc_0", "metadata": {}, "score": 1.0}]

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    @property
    def available_modes(self) -> list[str]:
        return self._modes

    def get_params(self) -> dict[str, str]:
        return {"backend": "stub"}


class TestSystemPrompt:
    def test_system_prompt_is_a_class_attribute(self):
        # Must live on the class (not just the instance) so the
        # dataset_preprocess classmethod can read it via cls.system_prompt.
        assert LinkerEnv.system_prompt == _SYSTEM_PROMPT
        assert LinkerEnv.system_prompt

    def test_instance_resolves_system_prompt(self):
        env = LinkerEnv(search=StubSearch())
        assert env.system_prompt == _SYSTEM_PROMPT


class TestDatasetPreprocess:
    def test_bakes_system_prompt_into_example_via_classmethod(self):
        # Regression: dataset_preprocess is a classmethod reading
        # cls.system_prompt. Before the fix the prompt was only set on the
        # instance, so cls.system_prompt resolved to BaseEnv's "" and the
        # system message was silently dropped from training Examples.
        result = LinkerEnv.dataset_preprocess(
            {"target_n": 2, "reasoning_mode": "", "prompt": "Primary chunk text."}
        )
        system_msgs = [m for m in result["prompt_messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "evidence chain" in system_msgs[0]["content"]

    def test_templates_user_prompt(self):
        result = LinkerEnv.dataset_preprocess(
            {"target_n": 3, "reasoning_mode": "", "prompt": "Primary chunk text."}
        )
        user_msgs = [m for m in result["prompt_messages"] if m["role"] == "user"]
        assert user_msgs
        assert "Find 3 secondary chunk(s)" in user_msgs[0]["content"]
        assert "Primary chunk text." in user_msgs[0]["content"]
        assert result["task"]["target_n"] == 3
