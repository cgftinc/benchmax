"""A small arithmetic environment and its castform workflow."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmax.bundle import dump_bundle
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    Dataset,
    DatasetSplit,
    Example,
    JsonlDataset,
    JsonRow,
    Tool,
    canonical_example_id,
)
from benchmax.envs.base import resolve_dataset_path
from benchmax.envs.environment import Environment
from benchmax.rewards import extract_answer_block, extract_completion_text
from castform.platform import ensure_session, upload_assets

SYSTEM_PROMPT = (
    "Use the arithmetic tools to solve the expression. "
    "Return only the final number inside <answer></answer> tags."
)
TOOL_PADDING_PROBABILITY = 0.1
TOOL_PADDING_CHARS = 1_000


class MathEnv(BaseEnv):
    """Solve mixed arithmetic with four tools."""

    max_turns = 3

    def __init__(self) -> None:
        super().__init__()
        self._random = random.Random(0)

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> Dataset[JsonRow]:
        return JsonlDataset(
            resolve_dataset_path(base_dir, f"{split}.jsonl"),
            row_to_example=_to_example,
            max_examples=max_examples,
        )

    async def list_tools(self) -> list[Tool]:
        return list(TOOLS)

    async def run_tool(
        self,
        rollout_id: str,
        tool_name: str,
        **tool_args: Any,
    ) -> str:
        del rollout_id
        operation = OPERATIONS.get(tool_name)
        if operation is None:
            raise ValueError(f"unknown tool: {tool_name}")
        try:
            result = operation(_number(tool_args["a"]), _number(tool_args["b"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            return f"error: {error}"

        response = _format_number(result)
        if self._random.random() < TOOL_PADDING_PROBABILITY:
            response += f"\n{'x' * TOOL_PADDING_CHARS}"
        return response

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        completion = extract_completion_text(rollout.messages)
        prediction = extract_answer_block(completion)
        used_tool = any(
            message.get("role") == "assistant" and message.get("tool_calls")
            for message in rollout.messages
        )
        return {
            "correctness": float(
                bool(used_tool) and _same_number(prediction, rollout.example_args["answer"])
            )
        }


def _to_example(row: JsonRow) -> Example[JsonRow]:
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("math rows require a non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("math rows require a non-empty answer")
    payload: JsonRow = {
        "prompt_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "answer": answer,
    }
    return Example(id=canonical_example_id(payload), payload=payload)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError("expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("expected a finite number")
    return number


def _divide(left: float, right: float) -> float:
    if right == 0:
        raise ZeroDivisionError("division by zero")
    return left / right


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def _same_number(left: object, right: object) -> bool:
    try:
        left_number = _number(left)
        right_number = _number(right)
    except (TypeError, ValueError):
        return False
    return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)


OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "add": lambda left, right: left + right,
    "subtract": lambda left, right: left - right,
    "multiply": lambda left, right: left * right,
    "divide": _divide,
}


def _tool(name: str, description: str) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }


TOOLS: tuple[Tool, ...] = tuple(
    _tool(name, description)
    for name, description in (
        ("add", "add two numbers"),
        ("subtract", "subtract b from a"),
        ("multiply", "multiply two numbers"),
        ("divide", "divide a by b"),
    )
)


DATASET_REPO = "dawidmt/arithmetic50"
TRAIN_EXAMPLES = 40
EVAL_EXAMPLES = 10
DATA_DIR = Path(__file__).parent / "data"
RUN_NAME = "math"
VALIDATION_MODEL = "gpt-5.4-mini"
TRAINING_ARGS = {
    "model": "Qwen/Qwen3.5-4B",
    "max_context_tokens": 4096,
    "num_epochs": 10,
}


def generate_data(*, force: bool = False) -> dict[str, Path]:
    dataset_files = {
        "train.jsonl": DATA_DIR / "train.jsonl",
        "eval.jsonl": DATA_DIR / "eval.jsonl",
    }
    if all(path.exists() for path in dataset_files.values()) and not force:
        print(f"data: using existing {TRAIN_EXAMPLES} train / {EVAL_EXAMPLES} eval examples")
        return dataset_files

    from datasets import load_dataset

    source = load_dataset(DATASET_REPO, split="test")
    total_examples = TRAIN_EXAMPLES + EVAL_EXAMPLES
    if len(source) < total_examples:
        raise RuntimeError(
            f"{DATASET_REPO} returned {len(source)} examples; expected at least {total_examples}"
        )
    rows = [
        {"question": str(row["task"]), "answer": str(row["answer"])}
        for row in source.select(range(total_examples))
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dataset_files["train.jsonl"], rows[:TRAIN_EXAMPLES])
    _write_jsonl(dataset_files["eval.jsonl"], rows[TRAIN_EXAMPLES:])
    print(f"data: wrote {TRAIN_EXAMPLES} train / {EVAL_EXAMPLES} eval examples")
    return dataset_files


def validate(env: MathEnv, uploaded_assets):
    from castform import validate_environment

    report = asyncio.run(
        validate_environment(
            env,
            model=VALIDATION_MODEL,
            split="train",
            base_dir=DATA_DIR,
            remote_assets=uploaded_assets,
            max_context_tokens=2048,
        )
    )
    _print_validation(report)
    return report


def launch(
    uploaded_assets,
    *,
    run_name: str = RUN_NAME,
    assume_yes: bool = False,
) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("launch: cancelled")
            return None

    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=run_name,
            launcher_args=TRAINING_ARGS,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )


def _print_validation(report: Any) -> None:
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
            else:
                mark = (
                    "✅"
                    if outcome.termination_reason in Environment.scorable_termination_reasons
                    else "❌"
                )
                error_suffix = f" error={outcome.error}" if outcome.error else ""
                print(
                    f"{mark} {location} {rollout_id}: "
                    f"{outcome.termination_reason} {dict(outcome.rewards)}{error_suffix}"
                )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def run_cli(
    env_class: type[MathEnv] = MathEnv,
    *,
    run_name: str = RUN_NAME,
    constructor_args: dict[str, Any] | None = None,
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("data", "validate", "launch"),
        default="validate",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the launch confirmation",
    )
    args = parser.parse_args(argv)
    resolved_constructor_args = constructor_args or {}
    total_stages = {
        "data": 1,
        "validate": 4,
        "launch": 5,
    }[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    dataset_files = generate_data(force=args.force)
    if args.action == "data":
        return 0

    ensure_session()
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        env_class,
        constructor_args=resolved_constructor_args,
        pip_dependencies=[],
    )
    print(f"[stage 3/{total_stages}] uploading environment and dataset")
    uploaded_assets = upload_assets(
        bundle=bundled_environment,
        dataset_files=dataset_files,
        run_name=run_name,
    )
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(env_class(**resolved_constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(
            uploaded_assets,
            run_name=run_name,
            assume_yes=args.yes,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv=argv)


if __name__ == "__main__":
    sys.exit(main())
