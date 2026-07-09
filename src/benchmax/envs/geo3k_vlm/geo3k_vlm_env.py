"""Multimodal document / VLM env with Infinity-Doc OCR reward."""

from __future__ import annotations

from typing import Any, Optional

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.geo3k_vlm.reward_fn import infinity_doc_reward
from benchmax.envs.types import Example, Messages, ToolDefinition


SYSTEM_PROMPT = ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


class Geo3KVLMEnv(BaseEnv):
    """Tool-free multimodal env for document OCR / geometry VLM rollouts."""

    system_prompt: str = SYSTEM_PROMPT
    recommended_max_turns: Optional[int] = 1
    recommended_max_tool_calls: Optional[int] = 0

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example:
        prompt = str(example.get("prompt") or example.get("problem") or "").strip()
        images = [
            str(image)
            for image in _as_list(example.get("images") or example.get("image_urls"))
            if image
        ]
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image}} for image in images
        ]
        text = prompt or cls.system_prompt
        if text or not content:
            content.append({"type": "text", "text": text})
        return make_example(
            prompt_messages=[{"role": "user", "content": content}],
            task={
                "answer": example.get("answer") or example.get("label") or "",
                "metadata": example.get("metadata") or {},
            },
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        raise ValueError(f"{self.__class__.__name__} has no tools")

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, float]:
        return {"answer_correct": infinity_doc_reward(messages, task, **kwargs)}
