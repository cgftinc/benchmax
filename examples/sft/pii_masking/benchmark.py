"""Command line for the PII-masking benchmark workflow.

Five commands, each a separate consent boundary:

``prepare``            read the pinned sources; write samples and the protocol
``launch``             upload the datasets and start exactly one SFT run
``preflight-adapter``  load and attest the terminal adapter, without generating
``evaluate``           fill missing smoke, pilot, or full request identities
``score``              recompute the report from the journal, offline

This module stays shallow on purpose. It parses arguments, enforces the
interlocks, and delegates; selection, scoring, and inference each own their own
logic so that reading this file tells you what is *permitted*, not how any of it
works.

A note on the flags. ``--allow-network`` and ``--yes`` are safety interlocks
against accident — a mistyped command, a rerun of shell history. They are **not**
authorization. An agent running this workflow must obtain human approval for
each networked, mutating, or paid step regardless of what flags it is able to
type.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from .benchmark_protocol import (
    BENCHMARK_SOURCES,
    OUTPUT_ROOT,
    Protocol,
    ProtocolError,
)

PHASES = ("smoke", "pilot", "full")


def _load(module_name: str) -> Any:
    """Import a sibling workflow module, lazily.

    Keeps ``--help``, ``score``, and every refusal path free of the source and
    model clients, so they need neither network nor credentials.
    """
    try:
        return importlib.import_module(f".{module_name}", __package__)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{module_name} is unavailable in this build: {exc}") from exc


def _require(flag_value: bool, flag: str, action: str) -> None:
    """Refuse an action whose interlock flag is absent."""
    if not flag_value:
        raise SystemExit(
            f"refusing to {action} without {flag}. This flag guards against accidental "
            f"invocation; it does not by itself constitute approval for the action."
        )


def _protocol(directory: Path) -> Protocol:
    return Protocol.load(directory)


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_prepare(args: argparse.Namespace) -> int:
    """Freeze samples and write the protocol. Networked but free."""
    _require(args.allow_network, "--allow-network", "read the pinned sources")
    selection = _load("benchmark_selection")
    directory = selection.prepare(
        benchmark_source=args.benchmark_source,
        output_root=args.output_root,
    )
    print(f"wrote protocol to {directory}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """Upload the datasets and start exactly one paid SFT run."""
    _require(args.yes, "--yes", "upload datasets and start a PAID training run")
    protocol = _protocol(args.protocol_dir)
    launcher = _load("benchmark_launch")
    run_id = launcher.launch(protocol, args.protocol_dir)
    print(f"launched SFT run {run_id}")
    return 0


def cmd_preflight_adapter(args: argparse.Namespace) -> int:
    """Load and attest the terminal adapter without generating anything.

    Idempotent, but externally mutating: it populates a shared serving replica's
    adapter cache. It is never described as read-only.
    """
    _require(args.yes, "--yes", "mutate the shared serving adapter cache")
    protocol = _protocol(args.protocol_dir)
    inference = _load("benchmark_inference")
    attestation = inference.preflight_adapter(protocol, args.protocol_dir)
    print(f"adapter attested: {attestation}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Fill the missing request identities for one phase. Paid."""
    _require(args.yes, "--yes", f"issue PAID {args.phase} inference requests")
    protocol = _protocol(args.protocol_dir)
    inference = _load("benchmark_inference")
    summary = inference.evaluate(protocol, args.protocol_dir, phase=args.phase)
    print(f"{args.phase}: {summary}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Recompute the report from the journal. Offline, no model calls."""
    protocol = _protocol(args.protocol_dir)
    scoring = _load("benchmark_scoring")
    report = scoring.score(protocol, args.protocol_dir, final=args.final)
    print(report)
    return 0


# ── parser ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="freeze samples and write the protocol (networked, free)"
    )
    prepare.add_argument("--benchmark-source", choices=BENCHMARK_SOURCES, required=True)
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    prepare.add_argument(
        "--allow-network",
        action="store_true",
        help="permit reads of the pinned sources; guards against accidental invocation only",
    )
    prepare.set_defaults(handler=cmd_prepare)

    launch = sub.add_parser("launch", help="upload datasets and start one PAID SFT run")
    launch.add_argument("--protocol-dir", type=Path, required=True)
    launch.add_argument("--yes", action="store_true", help="confirm the paid, mutating action")
    launch.set_defaults(handler=cmd_launch)

    preflight = sub.add_parser(
        "preflight-adapter",
        help="load and attest the terminal adapter (mutates the shared serving cache)",
    )
    preflight.add_argument("--protocol-dir", type=Path, required=True)
    preflight.add_argument("--yes", action="store_true", help="confirm the mutating action")
    preflight.set_defaults(handler=cmd_preflight_adapter)

    evaluate = sub.add_parser(
        "evaluate", help="fill missing request identities for one phase (PAID)"
    )
    evaluate.add_argument("--protocol-dir", type=Path, required=True)
    evaluate.add_argument("--phase", choices=PHASES, required=True)
    evaluate.add_argument("--yes", action="store_true", help="confirm the paid action")
    evaluate.set_defaults(handler=cmd_evaluate)

    score = sub.add_parser("score", help="recompute the report from the journal (offline)")
    score.add_argument("--protocol-dir", type=Path, required=True)
    score.add_argument(
        "--final",
        action="store_true",
        help="require 100%% completeness and emit a final metric",
    )
    score.set_defaults(handler=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
