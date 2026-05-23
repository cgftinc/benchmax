"""Launch a GuessEnv training run via the platform HTTP endpoint.

Mirrors the ``__main__`` flow in ../telestitch/telestich.py but uses a
minimal wordle-style env (guess my number). Bundles the env, uploads it
+ a tiny dataset to platform storage, and POSTs /v1/train/runs/launch.

    cd core/benchmax/examples/guess
    uv sync
    CASTFORM_API_KEY=sk_... uv run python launch_guess.py

Set CASTFORM_BASE_DOMAIN (or CASTFORM_PLATFORM_URL) to target staging/dev.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from benchmax import config
from benchmax.envs.base_env import BaseEnv, ToolDefinition
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example, Messages
from benchmax.platform.client import TrainerClient
from benchmax.platform.training_run import upload_training_run

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("CASTFORM_API_KEY", "")
BASE_URL = config.platform_url()

TRAIN_TARGETS = list(range(1, 15))
EVAL_TARGETS = [5, 19, 42]


SYSTEM_PROMPT = (
    "I'm thinking of a positive integer. Try to guess it. "
    "Respond with just the number, nothing else."
)


def _extract_guess(messages: Any) -> int | None:
    if not isinstance(messages, list) or not messages:
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = (msg.get("content") or "").strip()
            for tok in text.replace(",", " ").split():
                tok = tok.strip(".!?:;\"'")
                if tok.lstrip("-").isdigit():
                    try:
                        return int(tok)
                    except ValueError:
                        return None
            return None
    return None


class GuessEnv(BaseEnv):
    system_prompt: str = SYSTEM_PROMPT

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs: Any) -> Example:
        return make_example(
            prompt_messages=[
                {"role": "user", "content": "Are you ready to play a guessing game?"},
                {
                    "role": "assistant",
                    "content": "Yes! Pick any positive integer and I'll try to guess it.",
                },
                {"role": "user", "content": "I have a number in mind. What is it?"},
            ],
            task={"target": int(example["target"])},
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        return ""

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        target = (task or {}).get("target")
        guess = _extract_guess(messages)
        if target is None:
            return {"correctness": 0.0, "format": 0.0}
        format_ok = 1.0 if guess is not None else 0.0
        correctness = 1.0 if guess == target else 0.0
        logger.info(
            "[reward] target=%s guess=%s format=%s correctness=%s",
            target, guess, format_ok, correctness,
        )
        return {"correctness": correctness, "format": format_ok}

    async def init_rollout(self, rollout_id: str, **rollout_args: Any) -> None:
        return None

    async def release_rollout(self, rollout_id: str) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def main() -> None:
    if not API_KEY:
        raise SystemExit("Set CASTFORM_API_KEY before running.")

    print(f"Platform URL: {BASE_URL}")

    train_data = [{"target": t} for t in TRAIN_TARGETS]
    eval_data = [{"target": t} for t in EVAL_TARGETS]

    run_name = f"guess-benchmax-{uuid.uuid4().hex[:8]}"
    print(f"Uploading bundle + datasets as {run_name!r} ...")
    uploaded = upload_training_run(
        env_class=GuessEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=run_name,
        api_key=API_KEY,
        base_url=BASE_URL,
        constructor_args=None,
        pip_dependencies=["benchmax"],
    )
    for label, path in (
        ("env_cls", uploaded.env_cls_path),
        ("env_metadata", uploaded.env_metadata_path),
        ("train_dataset", uploaded.train_dataset_path),
        ("eval_dataset", uploaded.eval_dataset_path),
    ):
        print(f"  {label:<14}: {path}")

    print("\nLaunching training run ...")
    with TrainerClient(api_key=API_KEY, base_url=BASE_URL) as trainer:
        run_id = trainer.launch_training_run(
            training_run_type="simple",
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            train_dataset_path=uploaded.train_dataset_path,
            eval_dataset_path=uploaded.eval_dataset_path,
            name=run_name,
        )

    print(f"\n✓ Launched run_id={run_id}")
    print(f"  View / cancel at: {config.web_app_url()}/train/{run_id}")


if __name__ == "__main__":
    main()
