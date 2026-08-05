"""Minimal single-turn environment (written by `castform setup`).

Post-trains a model on a single-turn task: given a `prompt`, produce an answer,
scored against `ground_truth`. No tools — `list_tools` returns []. The reward IS
the training signal; it lives here in `compute_reward` so it's yours to read and
edit. This is a deliberately small SEED — grow it into your task: change the
reward, swap the dataset, add tools (see the design-environment skill).

The whole run is reproducible from this file: the reward is below, and the
`VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks bake in the rollout budgets.
`python main.py` drives the loop: data → upload → validate, then STOP
(`launch` is an explicit, confirmed step — it spends GPU credits).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from benchmax.envs.base import resolve_dataset_path

from benchmax.bundle import dump_bundle
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Environment,
    Example,
    JsonlDataset,
    JsonRow,
    canonical_example_id,
)
from benchmax.rewards import extract_completion_text
from castform import validate_environment
from castform.platform.client import TrainerClient
from castform.platform.environment_assets import upload_assets
from castform.platform.login import ensure_session


class CustomEnv(BaseEnv):
    """A single-turn env: the model answers a prompt, scored against ground_truth.

    No tools and no LLM judge — the reward is normalized exact-match correctness
    and needs no separate judge call. Swap in your own reward / tools / judge.
    """

    # Advertised to the model each rollout (optional; default ""). Keep it short.
    system_prompt = "Return only the answer, without explanation."
    max_turns = 1

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            resolve_dataset_path(base_dir, f"{split}.jsonl"),
            row_to_example=self._example_from_row,
            max_examples=max_examples,
        )

    def _example_from_row(self, row: JsonRow) -> Example[JsonRow]:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("dataset rows require a string 'prompt'")
        if not prompt.strip():
            raise ValueError("dataset rows require a non-empty 'prompt'")
        ground_truth = row.get("ground_truth")
        if not isinstance(ground_truth, str):
            raise TypeError("dataset rows require a string 'ground_truth'")
        if not ground_truth.strip():
            raise ValueError("dataset rows require a non-empty 'ground_truth'")
        extra = {
            key: value
            for key, value in row.items()
            if key not in ("prompt", "ground_truth", "prompt_messages")
        }
        payload: JsonRow = {
            **extra,
            "prompt_messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "ground_truth": ground_truth,
        }
        return Example(id=canonical_example_id(payload), payload=payload)

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> dict[str, float]:
        """Score a normalized exact answer against the required ground truth."""
        answer = extract_completion_text(rollout.messages).strip()
        gold = rollout.example_args.get("ground_truth")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError("rollout requires a non-empty string 'ground_truth'")
        normalized_answer = " ".join(answer.casefold().split())
        normalized_gold = " ".join(gold.casefold().split())
        return {"correctness": float(normalized_answer == normalized_gold)}


# ── Run config — validate/launch read these so the run reproduces from this
#    file alone. See `python main.py validate --help` / `launch --help`.
VALIDATE_CONFIG = {
    "model": "gpt-5.4-mini",
    "max_context_tokens": 2048,
    "local_timeout_seconds": 120,
}

LAUNCH_CONFIG = {
    "model": "Qwen/Qwen3.5-4B",
    # Total prompt + response tokens across the WHOLE rollout. Single-turn answers
    # are short; raise it if your task needs longer prompts or tool transcripts.
    "max_context_tokens": 4096,
    "num_epochs": 2,  # eval tends to peak before the overfit tail; keep epochs modest
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}


# ── Runnable entrypoint ──────────────────────────────────────────────────────
# `python main.py [data|validate|launch]` drives the whole loop SDK-directly —
# no CLI needed, and this file stays the reproducible record of the run:
#
#   python main.py data       generate/refresh the datasets (skip if present)
#   python main.py validate   upload, then check the same assets locally and in
#                             a hosted sandbox (the default; never launches)
#   python main.py launch     the same path, then a confirmed GPU launch that
#                             reuses the assets that were just validated
#
# Import-safe: stages run only from the ``if __name__ == "__main__"`` block below,
# so importing this file for tests or environment inspection has no side effects.

TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"

# Dependencies installed in the remote rollout runtime. Keep this declaration at
# the script boundary: add only packages imported by your env/reward/tool code.
RUNTIME_DEPENDENCIES: list[str] = []

# The tiny seed dataset — synthetic, reproducible. `castform setup` also commits
# these as train.jsonl / eval.jsonl so validate runs on day one;
# regenerate them any time with `python main.py data --force`.
_SEED_TRAIN = [
    {"prompt": "What is the capital of France?", "ground_truth": "Paris"},
    {"prompt": "What is 2 + 2?", "ground_truth": "4"},
    {"prompt": "What color is a clear daytime sky?", "ground_truth": "Blue"},
]
_SEED_EVAL = [
    {"prompt": "What is the capital of Japan?", "ground_truth": "Tokyo"},
    {"prompt": "What is 10 minus 3?", "ground_truth": "7"},
]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text("utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_name() -> str:
    return str(LAUNCH_CONFIG.get("name") or CustomEnv.__name__.lower())


def _constructor_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build explicit, serializable kwargs shared by local and remote envs.

    Add task configuration and credentials as CLI arguments, then translate them
    here instead of reading ambient environment variables inside ``CustomEnv``.
    """
    del args
    return {}


def generate_data(force: bool = False) -> bool:
    """Produce `train.jsonl` / `eval.jsonl`.

    Provenance: a tiny synthetic seed, generated inline below (reproducible). Replace
    it with your real task's data — hand-write the jsonl, or generate it and inline
    the gen code here so the dataset stays reproducible from this file. Re-run with
    --force to regenerate; skip-if-exists otherwise.
    """
    datasets = (
        (Path(TRAIN_FILE), _SEED_TRAIN),
        (Path(EVAL_FILE), _SEED_EVAL),
    )
    pending = (
        list(datasets) if force else [(path, rows) for path, rows in datasets if not path.exists()]
    )
    if not pending:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    for path, rows in pending:
        _write_jsonl(str(path), rows)
    written = ", ".join(f"{path} ({len(rows)} rows)" for path, rows in pending)
    print(f"data: wrote {written}")
    return True


def _print_validation(report: Any) -> None:
    for location in ("static", "local", "remote"):
        warnings = getattr(report, f"{location}_warnings", {}) or {}
        for item, messages in warnings.items():
            values = messages if isinstance(messages, list) else [messages]
            for message in values:
                print(f"⚠️ {location} {item}: {message}")
    for item, error in (getattr(report, "static_errors", {}) or {}).items():
        print(f"❌ static {item}: {error}")
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


def validate(env: CustomEnv, uploaded_assets: Any) -> Any:
    """Validate the exact uploaded assets, locally and in a hosted sandbox."""
    report = asyncio.run(
        validate_environment(
            env,
            model=str(VALIDATE_CONFIG["model"]),
            split="train",
            base_dir=Path("."),
            remote_assets=uploaded_assets,
            max_context_tokens=int(VALIDATE_CONFIG["max_context_tokens"]),
            local_timeout_seconds=float(VALIDATE_CONFIG["local_timeout_seconds"]),
        )
    )
    _print_validation(report)
    return report


def launch(uploaded_assets: Any, *, assume_yes: bool = False) -> str | None:
    """Confirm, then train on GPUs with the assets that were just validated."""
    if not assume_yes:
        reply = (
            input(f"Launch '{_run_name()}' on GPUs — this spends credits. Continue? [y/N] ")
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None
    # LAUNCH_CONFIG feeds the launcher, minus the reserved keys: `name` is the
    # run name; `type` is not a wire arg. The server rejects any unknown key.
    launcher_args = {
        key: value for key, value in LAUNCH_CONFIG.items() if key not in ("name", "type")
    }
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=_run_name(),
            launcher_args=launcher_args or None,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → upload → validate → launch.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="validate",
        choices=["data", "validate", "launch"],
        help="data stops after datasets; validate stops after checks; launch trains.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate datasets even if present.")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    if not generate_data(force=args.force):
        return 1
    if args.action == "data":
        return 0

    constructor_args = _constructor_args(args)
    ensure_session()  # data preparation stays usable without platform login
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        CustomEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment and dataset")
    uploaded_assets = upload_assets(
        bundle=bundled_environment,
        train_dataset=_load_jsonl(TRAIN_FILE),
        eval_dataset=(_load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else None),
        run_name=_run_name(),
    )
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(CustomEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        return 0 if launch(uploaded_assets, assume_yes=args.yes) is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
