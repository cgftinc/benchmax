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
        assert LinkerEnv.system_prompt == _SYSTEM_PROMPT
        assert LinkerEnv.system_prompt

    def test_instance_resolves_system_prompt(self):
        env = LinkerEnv(search=StubSearch())
        assert env.system_prompt == _SYSTEM_PROMPT


class TestDatasetParsing:
    def test_system_prompt_is_in_prompt_messages(self):
        result = LinkerEnv(search=StubSearch())._example_from_row(
            {"target_n": 2, "reasoning_mode": "", "prompt": "Primary chunk text."}
        )
        system_msgs = [
            m for m in result.payload["prompt_messages"] if m["role"] == "system"
        ]
        assert system_msgs == [{"role": "system", "content": _SYSTEM_PROMPT}]

    def test_templates_user_prompt(self):
        result = LinkerEnv(search=StubSearch())._example_from_row(
            {"target_n": 3, "reasoning_mode": "", "prompt": "Primary chunk text."}
        )
        user_msgs = [
            m for m in result.payload["prompt_messages"] if m["role"] == "user"
        ]
        assert user_msgs
        assert "Find 3 secondary chunk(s)" in user_msgs[0]["content"]
        assert "Primary chunk text." in user_msgs[0]["content"]
        assert result.payload["target_n"] == 3
