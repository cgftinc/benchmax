"""castform run-control verbs (slice 1.3).

Top-level ``castform stop <id>`` → ``POST /v1/train/runs/{id}/cancel``. Owner-only
(403 otherwise). A launched run emits ``training.cancelling`` + cancels its
launcher job; a run with no job is marked complete directly (no cancelling event).
"""

from __future__ import annotations

import argparse

from castform.cli._client import handle_errors, trainer_client


@handle_errors
def _cmd_stop(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        result = client.cancel_run(args.run_id)
    message = result.get("message") or "Cancellation requested"
    print(f"✓ {message}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `stop` verb."""
    p_stop = sub.add_parser("stop", help="Stop (cancel) a run you own")
    p_stop.add_argument("run_id")
    p_stop.set_defaults(func=_cmd_stop)
