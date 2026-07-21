"""Canonical math example: prepare arithmetic50, validate, bundle, launch.

The historical "mathenv" runs (e.g. math-reviewable-0716) trained on the
HuggingFace ``dawidmt/arithmetic50`` split: fifty small mixed-operator
expressions written with ``×``/``÷`` symbols and order-of-operations traps
that reliably confuse small models, on top of MathEnv's long-tool-output
padding. This script reproduces that flow end to end:

    uv run --package benchmax-example-math python example.py

Environment knobs:
    MATH_RUN_NAME        override the generated run name
    MATH_MODEL           trainer model (default Qwen/Qwen3.5-4B)
    MATH_VALIDATE_MODEL  validation model served by the Castform LLM proxy
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from benchmax.bundle import Bundle, dump_bundle
from benchmax.envs.identity import canonical_example_id
from benchmax.envs.shared_types import Example
from castform import config, validate_environment
from math_dataset import SYSTEM_PROMPT
from math_env import MathEnv

DATASET_REPO = "dawidmt/arithmetic50"
TRAIN_COUNT = 40
MODEL = os.environ.get("MATH_MODEL", "Qwen/Qwen3.5-4B")
VALIDATE_MODEL = os.environ.get("MATH_VALIDATE_MODEL", "gpt-5.4-mini")
RUN_NAME = os.environ.get("MATH_RUN_NAME")


def get_dataset() -> tuple[list[dict], list[dict]]:
    """Load the fixed arithmetic50 rows and split them 40 train / 10 eval."""

    from datasets import load_dataset

    rows = [
        {"task": row["task"], "answer": str(row["answer"])}
        for row in load_dataset(DATASET_REPO, split="test")
    ]
    if len(rows) <= TRAIN_COUNT:
        raise SystemExit(
            f"{DATASET_REPO} returned {len(rows)} rows; expected more than {TRAIN_COUNT}"
        )
    return rows[:TRAIN_COUNT], rows[TRAIN_COUNT:]


def build_training_bundle(constructor_args: dict[str, str]) -> Bundle:
    return dump_bundle(MathEnv, constructor_args=constructor_args)


def confirm_gpu_launch(run_name: str) -> bool:
    reply = (
        input(f"Launch {run_name!r} on GPUs — this spends credits. Continue? [y/N] ")
        .strip()
        .lower()
    )
    return reply in ("y", "yes")


if __name__ == "__main__":
    from castform.platform import ensure_session
    from castform.platform.client import TrainerClient
    from castform.platform.training_run import upload_training_run

    ensure_session()
    print(f"Platform URL: {config.platform_url()}")
    print(f"LLM URL:      {config.llm_url()}\n")

    train_rows, eval_rows = get_dataset()
    print(f"{len(train_rows)} train / {len(eval_rows)} eval from {DATASET_REPO}")

    # Dataset locations must be known before the bundle is built (constructor
    # args travel inside the pickle), so pin the upload prefix.
    run_name = RUN_NAME or f"mathenv-{uuid.uuid4().hex[:8]}"
    dataset_prefix = f"datasets/{run_name}"
    constructor_args = {
        "train_dataset_path": f"{dataset_prefix}/train.jsonl",
        "eval_dataset_path": f"{dataset_prefix}/eval.jsonl",
    }

    print("\nValidating env (local two-rollout group) ...")
    # Validation-only relaxed turn budget: proxy validation models play
    # tools strictly one-per-turn and cannot finish 3-op tasks in the
    # bundled max_turns=3; validation checks execution mechanics, not
    # turn-efficiency. The uploaded bundle keeps the strict default.
    validation_env = MathEnv(**constructor_args, max_turns=5)
    payload = {
        "prompt_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": train_rows[0]["task"]},
        ],
        "answer": train_rows[0]["answer"],
    }
    report = asyncio.run(
        validate_environment(
            validation_env,
            example=Example(id=canonical_example_id(payload), payload=payload),
            model=VALIDATE_MODEL,
        )
    )
    for rollout_id, outcome in report.local.items():
        print(
            f"  {rollout_id}: termination={outcome.termination_reason} "
            f"rewards={dict(outcome.rewards)}"
        )
    if not report.ok:
        sys.exit("Env validation failed — aborting before launch.")

    if not confirm_gpu_launch(run_name):
        raise SystemExit("Launch aborted.")

    bundle = build_training_bundle(constructor_args)
    print(f"\nUploading bundle + datasets as {run_name!r} ...")
    uploaded = upload_training_run(
        bundle=bundle,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        run_name=run_name,
        dataset_prefix=dataset_prefix,
    )

    print(f"\nLaunching training run (model={MODEL}) ...")
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            train_dataset_path=uploaded.train_dataset_path,
            eval_dataset_path=uploaded.eval_dataset_path,
            name=run_name,
            launcher_args={"model": MODEL, "num_epochs": 10},
        )

    print(f"\n✓ Launched run_id={run_id}")
    print(f"  View / cancel at: {config.web_app_url()}/train/{run_id}")
