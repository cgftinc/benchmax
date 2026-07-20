"""Env-less SFT dataset project (written by `castform setup --template sft`).

Post-trains a model via supervised fine-tuning on `{"messages": [...]}` rows —
the OpenAI fine-tuning chat format, with optional `tools` and per-assistant-
message `weight` (0/1) for turn masking. There is no environment and no
reward: `TRAINING_MODE = "sft"` is the explicit mode marker `castform` reads
instead of discovering a `BaseEnv` subclass (see `benchmax.cli._project`).

The whole run is reproducible from this file: the seed dataset is below, and
the `VALIDATE_CONFIG` / `LAUNCH_CONFIG` blocks bake in the dataset/launch
knobs so `castform validate` / `castform launch` need no extra flags (a CLI
flag still overrides). `python main.py` drives the loop: data → validate,
then STOP (`launch` is an explicit, confirmed step — it spends GPU credits).

Multimodal: a user message may carry a content-part list (e.g. a `text` part
plus an `image_url` part) instead of a plain string — see `_SEED_MULTIMODAL`
below for a demonstration row. It is opt-in only: a vision base model is
required, so `generate_data()` does not write it by default (see the comment
above `_SEED_MULTIMODAL`).

Weight/masking is experimental: trainer support for per-message `weight` is
unconfirmed, so `launch()` blocks a weight-bearing dataset unless
`LAUNCH_CONFIG["allow_experimental_weights"]` is set. Live launch is also
gated behind `benchmax.platform.client.SFT_LAUNCH_SUPPORTED` (False as of
writing — the platform doesn't yet accept env-less sft launch args); the
upload→launch path below is fully implemented and unit-tested, it just can't
reach a real platform yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmax.platform.client import SFT_LAUNCH_SUPPORTED, TrainerClient
from benchmax.platform.login import ensure_session
from benchmax.platform.training_run import upload_sft_run
from benchmax.sft import SftValidationReport, load_sft_dataset, validate_sft_dataset

# The explicit mode marker (see the module docstring) — read on its own by
# `cli._project.load_project`, before any BaseEnv discovery is attempted.
TRAINING_MODE = "sft"


# ── Run config — validate/launch read these so the run reproduces from this file
#    alone (a CLI flag still overrides). See `castform validate/launch --help`.
VALIDATE_CONFIG = {
    # "max_seq_len": 8192,     # char/4-heuristic token budget flagged as a notice
    # "max_row_bytes": 1 << 20,  # per-row serialized-size notice threshold (1 MiB)
}

LAUNCH_CONFIG = {
    "num_epochs": 2,  # eval tends to peak before the overfit tail; keep epochs modest
    # "allow_experimental_weights": False,  # set True to launch a weight-bearing
    #   dataset anyway — trainer masking support is unconfirmed (see the docstring).
    # "type": "simple",  # GPU pool (gpu4 for 4B / gpu8 for 35B); "simple-cpu" = smoke
}

# LAUNCH_CONFIG keys resolved locally, never forwarded as a launcher arg: `name` is
# the run name (see `_run_name`); `type` is not a wire arg; `allow_experimental_weights`
# is the client-side weight gate override and must never reach the server as an
# unknown launch arg.
_LAUNCH_CONFIG_RESERVED = frozenset({"type", "name", "allow_experimental_weights"})


# ── Runnable entrypoint ──────────────────────────────────────────────────────
# `python main.py [data|validate|launch|all]` drives the whole loop SDK-directly —
# no CLI needed, and this file stays the reproducible record of the run. Stages are
# isolable and skip work whose output already exists (`--force` to redo):
#
#   python main.py data       generate/refresh the datasets (skip if present)
#   python main.py validate   validate the dataset locally (no GPU, no rollouts)
#   python main.py launch     validate-gate, then train on GPUs (spends credits)
#   python main.py  (or all)  data → validate, then STOP (never auto-launches)
#
# Import-safe: this block runs ONLY under `python main.py`. When the castform CLI
# imports this file it execs under the "main" stem, not "__main__", so nothing here
# fires — `castform validate` / `launch` reuse the SAME SDK calls, no drift.

TRAIN_FILE = "train_dataset.jsonl"
EVAL_FILE = "eval_dataset.jsonl"

# A tiny hardcoded 1x1 red-pixel PNG data URI (no pillow/PIL dependency — just a
# literal base64 string). Used only by the opt-in `_SEED_MULTIMODAL` row below.
_TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC"
)

# The tiny seed dataset — synthetic, reproducible, text-only. `castform setup` also
# commits these as train_dataset.jsonl / eval_dataset.jsonl so validate runs on day
# one; regenerate them any time with `python main.py data --force`.
_SEED_TRAIN = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What color is a clear daytime sky?"},
            {"role": "assistant", "content": "A clear daytime sky is blue."},
        ]
    },
]
_SEED_EVAL = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 10 minus 3?"},
            {"role": "assistant", "content": "10 minus 3 equals 7."},
        ]
    },
]

# Opt-in multimodal demonstration — NOT written by `generate_data()` by default (see
# below). Enable it only when training against a VISION base model: append it to
# `_SEED_TRAIN` (e.g. `_SEED_TRAIN + [_SEED_MULTIMODAL]`) before calling
# `generate_data(force=True)`. Enabling it for a text-only base model is undefined
# trainer behavior.
_SEED_MULTIMODAL = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is the square in this image?"},
                {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URI}},
            ],
        },
        {"role": "assistant", "content": "The square is red."},
    ]
}


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")


def _run_name() -> str:
    return LAUNCH_CONFIG.get("name") or "sft-run"


def generate_data(force: bool = False) -> bool:
    """Produce `train_dataset.jsonl` / `eval_dataset.jsonl`.

    Provenance: a tiny synthetic seed, generated inline above (reproducible,
    text-only). Replace it with your real task's data — hand-write the jsonl in
    the OpenAI fine-tuning chat format, or generate it and inline the gen code
    here so the dataset stays reproducible from this file. Re-run with --force
    to regenerate; skip-if-exists otherwise. See `_SEED_MULTIMODAL` above to
    opt into a multimodal demonstration row (vision base models only).
    """
    have = Path(TRAIN_FILE).exists() and Path(EVAL_FILE).exists()
    if have and not force:
        print(f"data: {TRAIN_FILE} / {EVAL_FILE} present — skipping (--force to redo)")
        return True
    _write_jsonl(TRAIN_FILE, _SEED_TRAIN)
    _write_jsonl(EVAL_FILE, _SEED_EVAL)
    print(f"data: wrote {len(_SEED_TRAIN)} train / {len(_SEED_EVAL)} eval rows")
    return True


def _print_scorecard(report: SftValidationReport) -> None:
    """A minimal, SDK-direct scorecard (the CLI's `castform validate` prints a
    richer one)."""
    print(f"  rows: train {report.train_row_count} · eval {report.eval_row_count}")
    stats = report.token_length_stats
    print(
        f"  tokens (char/4 heuristic): min {stats.min_tokens}  max {stats.max_tokens}"
        f"  mean {stats.mean_tokens:.1f}"
    )
    masking = report.masking_summary
    if masking.rows_with_weight:
        print(
            f"  masking: {masking.rows_with_weight} row(s) use per-message weight "
            "(experimental)"
        )
    for issue in report.issues:
        loc = (
            f"{issue.source_path}:{issue.physical_line}"
            if issue.source_path
            else "(dataset)"
        )
        symbol = "✗" if issue.severity == "error" else "⚠"
        print(f"  {symbol} {loc}  {issue.message}")
    print(f"validate: {'PASS' if report.ok else 'FAIL'}")


def validate() -> SftValidationReport:
    """Validate the sft dataset pair locally (no GPU, no rollouts). Returns the
    report."""
    train = load_sft_dataset(TRAIN_FILE)
    eval_dataset = load_sft_dataset(EVAL_FILE) if Path(EVAL_FILE).exists() else None
    report = validate_sft_dataset(
        train,
        eval_dataset,
        **{k: v for k, v in VALIDATE_CONFIG.items() if k in ("max_seq_len", "max_row_bytes")},
    )
    _print_scorecard(report)
    return report


def launch(assume_yes: bool = False) -> str | None:
    """Validate-gate, weight-gate, confirm, then upload + launch an SFT training
    run (spends credits). Returns the run id, or None if gated/aborted."""
    report = validate()  # cheap pre-flight — never spend GPU on a broken dataset
    if report is None or not report.ok:
        print(
            "launch: validate gate FAILED — fix the dataset before launching.",
            file=sys.stderr,
        )
        return None

    # Weight gate: a dataset using per-message `weight` (masking) is launch-blocking
    # until trainer support is confirmed — a separate capability from
    # SFT_LAUNCH_SUPPORTED below; the validate notice alone does not clear it.
    if report.masking_summary.rows_with_weight and not LAUNCH_CONFIG.get(
        "allow_experimental_weights", False
    ):
        print(
            "launch: dataset uses per-message 'weight' (masking) — trainer support "
            "is unconfirmed. Set allow_experimental_weights=True in LAUNCH_CONFIG to "
            "launch anyway.",
            file=sys.stderr,
        )
        return None

    # LAUNCH_CONFIG feeds the launcher, minus the reserved/local-only keys. The
    # server rejects any unknown key.
    launcher_args = {
        k: v for k, v in LAUNCH_CONFIG.items() if k not in _LAUNCH_CONFIG_RESERVED
    }

    # Guard before upload: no orphaned storage artifacts behind an API that cannot
    # succeed yet (see SFT_LAUNCH_SUPPORTED's docstring).
    if not SFT_LAUNCH_SUPPORTED:
        print(
            "launch: the platform does not accept env-less sft runs yet "
            "(benchmax.platform.client.SFT_LAUNCH_SUPPORTED is False).",
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

    train = load_sft_dataset(TRAIN_FILE)
    eval_dataset = load_sft_dataset(EVAL_FILE) if Path(EVAL_FILE).exists() else None
    uploaded = upload_sft_run(train=train, eval=eval_dataset, run_name=_run_name())

    with TrainerClient() as client:
        run_id = client.launch_sft_run(
            name=_run_name(),
            train_dataset_path=uploaded.train_dataset_path,
            eval_dataset_path=uploaded.eval_dataset_path,
            launcher_args=launcher_args or None,
        )
    print(f"launch: started run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this sft dataset: data → validate → launch.",
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

    ensure_session()  # best-effort: no-op if a credential resolves

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        report = validate()
        ok = report is not None and report.ok  # non-zero exit on a failed validate
    if args.stage == "launch":
        ok = launch(assume_yes=args.yes) is not None  # None = gated / aborted / failed
    # `all` / bare `python main.py` STOPS after validate — launch is never automatic
    # (it spends GPU credits); run `python main.py launch` to train.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
