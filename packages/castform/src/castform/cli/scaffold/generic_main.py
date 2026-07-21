"""Minimal single-turn environment (written by `castform setup`).

Post-trains a model on a single-turn task: given a `prompt`, produce an answer,
scored against `ground_truth`. No tools — `list_tools` returns []. The reward IS
the training signal; it lives here in `compute_reward` so it's yours to read and
edit. This is a deliberately small SEED — grow it into your task: change the
reward, swap the dataset, add tools (see the design-environment skill).

The whole run is reproducible from this file: the reward is below, and the
`VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks bake in the rollout budgets.
`python main.py` drives the loop: data → validate, then STOP
(`launch` is an explicit, confirmed step — it spends GPU credits).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Example,
    JsonRow,
    JsonlDataset,
    canonical_example_id,
)
from benchmax.bundle import dump_bundle
from benchmax.rewards import extract_completion_text
from castform.platform.client import TrainerClient
from castform.platform.login import ensure_session
from castform.platform.training_run import upload_training_run
from castform import validate_environment


class CustomEnv(BaseEnv):
    """A single-turn env: the model answers a prompt, scored against ground_truth.

    No tools and no LLM judge — the reward itself is deterministic string overlap
    and needs no separate judge call. Swap in your own reward / tools / judge.
    """

    # Advertised to the model each rollout (optional; default ""). Keep it short.
    system_prompt = "Answer the question concisely and directly."
    max_turns = 1
    reward_keys = ("overlap", "contains_gold", "answered")

    async def create_dataset(
        self, split: DatasetSplit, base_dir: Path
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            base_dir / f"{split}.jsonl",
            row_to_example=self._example_from_row,
        )

    def _example_from_row(self, row: JsonRow) -> Example[JsonRow]:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("dataset rows require a string 'prompt'")
        payload: JsonRow = {
            "prompt_messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **{key: value for key, value in row.items() if key != "prompt"},
        }
        return Example(id=canonical_example_id(payload), payload=payload)

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> dict[str, float]:
        """Score the answer against `ground_truth`. Return positive scores only; all
        components are SUMMED into one scalar — this is your whole training signal,
        so edit it. (Deterministic overlap here; swap in an LLM judge for open-ended
        answers — see the design-environment skill.)
        """
        t = rollout.example_args
        answer = extract_completion_text(rollout.messages).strip()
        gold = str(t.get("ground_truth") or t.get("answer") or "").strip()
        if not answer:
            return {"overlap": 0.0, "contains_gold": 0.0, "answered": 0.0}
        overlap = (
            difflib.SequenceMatcher(None, answer.lower(), gold.lower()).ratio()
            if gold
            else 0.0
        )
        contains = 1.0 if gold and gold.lower() in answer.lower() else 0.0
        return {
            "overlap": overlap,  # fuzzy match with the gold answer, in [0, 1]
            "contains_gold": 0.5 * contains,  # the gold string appears verbatim
            "answered": 0.1,  # small shaping: produced a non-empty answer
        }


# ── Run config — validate/launch read these so the run reproduces from this
#    file alone. See `python main.py validate --help` / `launch --help`.
VALIDATE_CONFIG = {
    "model": "gpt-5.4-mini",
    # Local always runs. Flip this only after hosted group validation is available.
    "include_remote": False,
}

LAUNCH_CONFIG = {
    # Total tokens across the WHOLE rollout. Single-turn answers are short; raise it
    # if your task needs long outputs (a truncated rollout is dropped from the loss).
    "max_rollout_len": 4096,
    "num_epochs": 2,  # eval tends to peak before the overfit tail; keep epochs modest
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}


# ── Runnable entrypoint ──────────────────────────────────────────────────────
# `python main.py [data|validate|launch|all]` drives the whole loop SDK-directly —
# no CLI needed, and this file stays the reproducible record of the run. Stages are
# isolable and skip work whose output already exists (`--force` to redo):
#
#   python main.py data       generate/refresh the datasets (skip if present)
#   python main.py validate   baseline on a real-rollout subset (no GPU)
#   python main.py launch     validate-gate, then train on GPUs (spends credits)
#   python main.py  (or all)  data → validate, then STOP (never auto-launches)
#
# Import-safe: stages run only from the ``if __name__ == "__main__"`` block below,
# so importing this file for tests or environment inspection has no side effects.

TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
ENV_ARGS: dict[str, Any] = {}  # CustomEnv constructor kwargs (none by default)

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


def generate_data(force: bool = False) -> bool:
    """Produce `train.jsonl` / `eval.jsonl`.

    Provenance: a tiny synthetic seed, generated inline below (reproducible). Replace
    it with your real task's data — hand-write the jsonl, or generate it and inline
    the gen code here so the dataset stays reproducible from this file. Re-run with
    --force to regenerate; skip-if-exists otherwise.
    """
    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    _write_jsonl(TRAIN_FILE, _SEED_TRAIN)
    _write_jsonl(EVAL_FILE, _SEED_EVAL)
    print(f"data: wrote {len(_SEED_TRAIN)} train / {len(_SEED_EVAL)} eval rows")
    return True


def _print_scorecard(report: Any) -> None:
    """Print the two sibling outcomes returned by local group validation."""
    for rollout_id, outcome in report.local.items():
        rewards = dict(outcome.rewards)
        total = sum(rewards.values())
        print(
            f"  {rollout_id}: termination_reason={outcome.termination_reason} "
            f"total={total:.3f} rewards={rewards}"
        )
    print(f"validate: {'PASS' if report.ok else 'FAIL'}")


async def _run_validation(env: CustomEnv) -> Any:
    dataset = await env.create_dataset("train", Path("."))
    if not dataset:
        raise ValueError(f"{TRAIN_FILE} contains no validation examples")
    return await validate_environment(
        env,
        example=dataset[0],
        model=str(VALIDATE_CONFIG["model"]),
        include_remote=bool(VALIDATE_CONFIG.get("include_remote", False)),
    )


def validate() -> Any:
    """Run the real environment group contract with exactly two siblings."""
    env = CustomEnv(**ENV_ARGS)
    report = asyncio.run(_run_validation(env))
    _print_scorecard(report)
    return report


def launch(assume_yes: bool = False) -> str | None:
    """Validate-gate, confirm, then upload + launch a GPU training run (spends
    credits). Returns the run id, or None if gated/aborted."""
    report = validate()  # cheap pre-flight — never spend GPU on a broken env
    if report is None or not report.ok:
        print(
            "launch: validate gate FAILED — fix the env before launching.",
            file=sys.stderr,
        )
        return None
    if not assume_yes:
        reply = (
            input(
                f"Launch '{_run_name()}' on GPUs — this spends credits. Continue? [y/N] "
            )
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None
    train = _load_jsonl(TRAIN_FILE)
    eval_ds = _load_jsonl(EVAL_FILE) if Path(EVAL_FILE).exists() else []
    bundle = dump_bundle(
        CustomEnv,
        constructor_args=ENV_ARGS,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    uploaded = upload_training_run(
        bundle=bundle,
        train_dataset=train,
        eval_dataset=eval_ds,
        run_name=_run_name(),
    )
    # LAUNCH_CONFIG feeds the launcher, minus the reserved keys: `name` is the run
    # name above; `type` is not a wire arg. The server rejects any unknown key.
    launcher_args = {
        k: v for k, v in LAUNCH_CONFIG.items() if k not in ("type", "name")
    }
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=_run_name(),
            launcher_args=launcher_args or None,
            **dataclasses.asdict(uploaded),
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="Stage to run (default: all = data → validate, then STOP).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        ensure_session()  # data preparation remains usable without platform login
        report = validate()
        ok = report is not None and report.ok  # non-zero exit on a failed baseline
    if args.stage == "launch":
        ensure_session()
        ok = launch(assume_yes=args.yes) is not None  # None = gated / aborted / failed
    # `all` / bare `python main.py` STOPS after validate — launch is never automatic
    # (it spends GPU credits); run `python main.py launch` to train.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
