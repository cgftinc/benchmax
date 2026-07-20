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
import os
import sys
from pathlib import Path
from typing import Any

from benchmax.platform.client import SFT_LAUNCH_SUPPORTED, TrainerClient
from benchmax.platform.login import ensure_session
from benchmax.platform.training_run import upload_sft_run
from benchmax.sft import (
    SftValidationReport,
    load_sft_dataset,
    sft_config_bool,
    sft_validate_kwargs,
    validate_sft_dataset,
)

# Explicit mode marker; read before BaseEnv discovery (see cli._project.load_project).
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

# Resolved locally; never forwarded to the server as a launch arg.
_LAUNCH_CONFIG_RESERVED = frozenset({"type", "name", "allow_experimental_weights"})


# ── Runnable entrypoint (see the module docstring + main()'s argparse help) ──

TRAIN_FILE = "train_dataset.jsonl"
EVAL_FILE = "eval_dataset.jsonl"

# A tiny hardcoded 1x1 red-pixel PNG data URI (no pillow/PIL dependency — just a
# literal base64 string). Used only by the opt-in `_SEED_MULTIMODAL` row below.
_TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC"
)

# Tiny synthetic seed, text-only; regenerate with `python main.py data --force`.
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

# Opt-in multimodal demonstration — NOT written by `generate_data()` by default.
# Enable it only against a VISION base model (undefined trainer behavior otherwise)
# by changing `generate_data()`'s `_create_seed_jsonl(TRAIN_FILE, _SEED_TRAIN)` call
# below to `_create_seed_jsonl(TRAIN_FILE, _SEED_TRAIN + [_SEED_MULTIMODAL])`, then
# re-run with `python main.py data --force`.
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


class SftScaffoldError(Exception):
    """`generate_data()` found a project state it refuses to touch automatically."""


def _create_seed_jsonl(path: str, rows: list[dict[str, Any]]) -> bool:
    """Atomically create `path` with `rows` iff nothing is there yet. An exclusive
    ``O_CREAT | O_EXCL`` open (not exists()-then-write_text()) closes the
    check/write race, and — since O_EXCL never follows a symlink at `path`, even
    a dangling one — a stray symlink is refused the same as a real file, not
    silently followed/clobbered. Returns True iff this call wrote the file."""
    payload = "".join(json.dumps(r) + "\n" for r in rows)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    return True


def _run_name() -> str:
    return LAUNCH_CONFIG.get("name") or "sft-run"


def generate_data(force: bool = False) -> bool:
    """Produce `train_dataset.jsonl` / `eval_dataset.jsonl`.

    Provenance: a tiny synthetic seed, generated inline above (reproducible,
    text-only). Replace it with your real task's data — hand-write the jsonl in
    the OpenAI fine-tuning chat format, or generate it and inline the gen code
    here so the dataset stays reproducible from this file. See `_SEED_MULTIMODAL`
    above to opt into a multimodal demonstration row (vision base models only).

    State machine (no `--force`):
      - neither file exists -> create both (the normal first-run case)
      - both exist -> skip both, unchanged
      - train exists, eval doesn't -> skip both; a train-only project is a valid,
        intentional state (eval is optional in SFT) — eval is never fabricated
      - eval exists, train doesn't -> refuse: there's no legitimate reason to have
        eval without train, so this looks like a corrupted project, not a first
        run — raises instead of guessing
    `--force` regenerates both unconditionally, regardless of prior state.
    """
    # lexists (not exists) so a symlink occupying a target path — dangling or
    # not — counts as "there", consistent with the exclusive-create below (which
    # refuses any occupied path without following it).
    train_exists = os.path.lexists(TRAIN_FILE)
    eval_exists = os.path.lexists(EVAL_FILE)

    if not force and eval_exists and not train_exists:
        raise SftScaffoldError(
            f"{EVAL_FILE} exists but {TRAIN_FILE} does not — refusing to guess "
            "what to do here (this looks like a corrupted project, not a first "
            f"run). Fix {TRAIN_FILE} manually, or re-run with --force to "
            "regenerate both from the seed."
        )

    if force:
        Path(TRAIN_FILE).unlink(missing_ok=True)
    if _create_seed_jsonl(TRAIN_FILE, _SEED_TRAIN):
        print(f"data: wrote {len(_SEED_TRAIN)} train rows to {TRAIN_FILE}")
    else:
        print(f"data: {TRAIN_FILE} present — skipping (--force to redo)")

    if not force and train_exists and not eval_exists:
        print(f"data: {EVAL_FILE} absent — leaving it that way (train-only project)")
        return True

    if force:
        Path(EVAL_FILE).unlink(missing_ok=True)
    if _create_seed_jsonl(EVAL_FILE, _SEED_EVAL):
        print(f"data: wrote {len(_SEED_EVAL)} eval rows to {EVAL_FILE}")
    else:
        print(f"data: {EVAL_FILE} present — skipping (--force to redo)")
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
    print(f"validate: {'pass' if report.ok else 'fail'}")


def _load_datasets():
    """Load the train/eval `SftDataset` pair once from disk."""
    train = load_sft_dataset(TRAIN_FILE)
    eval_dataset = load_sft_dataset(EVAL_FILE) if Path(EVAL_FILE).exists() else None
    return train, eval_dataset


def _validate_loaded(train, eval_dataset) -> SftValidationReport:
    """Validate an already-loaded train/eval pair (no disk I/O) and print the
    scorecard."""
    report = validate_sft_dataset(
        train, eval_dataset, **sft_validate_kwargs(VALIDATE_CONFIG)
    )
    _print_scorecard(report)
    return report


def validate() -> SftValidationReport:
    """Validate the sft dataset pair locally (no GPU, no rollouts). Returns the
    report."""
    train, eval_dataset = _load_datasets()
    return _validate_loaded(train, eval_dataset)


def launch(assume_yes: bool = False) -> str | None:
    """Validate-gate, weight-gate, confirm, then upload + launch an SFT training
    run (spends credits). Returns the run id, or None if gated/aborted.

    The train/eval pair is loaded ONCE, up front — the same objects that get
    validated are the ones passed to `upload_sft_run`, so a file edit/swap
    between validate and confirm can't bypass the schema/weight gates."""
    train, eval_dataset = _load_datasets()
    report = _validate_loaded(
        train, eval_dataset
    )  # never spend GPU on a broken dataset
    if report is None or not report.ok:
        print(
            "launch: validate gate failed — fix the dataset before launching.",
            file=sys.stderr,
        )
        return None

    # Weight gate: a separate capability from SFT_LAUNCH_SUPPORTED below — the
    # validate notice alone does not clear it. Strict bool (not truthiness) --
    # a typo'd string value must fail loudly, never silently clear the gate.
    if report.masking_summary.rows_with_weight and not sft_config_bool(
        LAUNCH_CONFIG, "allow_experimental_weights"
    ):
        print(
            "launch: dataset uses per-message 'weight' (masking) — trainer support "
            "is unconfirmed. set allow_experimental_weights=True in LAUNCH_CONFIG to "
            "launch anyway.",
            file=sys.stderr,
        )
        return None

    # Minus the reserved/local-only keys — the server rejects any unknown key.
    launcher_args = {
        k: v for k, v in LAUNCH_CONFIG.items() if k not in _LAUNCH_CONFIG_RESERVED
    }

    # Guard before upload — avoids orphaned storage behind an API that can't succeed yet.
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
                f"launch '{_run_name()}' on GPUs — this spends credits. continue? [y/N] "
            )
            .strip()
            .lower()
        )
        if reply not in ("y", "yes"):
            print("launch: aborted.")
            return None

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
        description="run the castform loop for this sft dataset: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="stage to run (default: all = data → validate, then stop).",
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the launch confirmation (it spends GPU credits).",
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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
