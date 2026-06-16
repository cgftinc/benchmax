"""castform ``runs`` command group — read-only run lifecycle (slice 1.2).

Thin GETs over platform REST. ``list`` needs login (hard 401); the per-run reads
are optionalAuth, so public runs work logged-out. All commands take ``--json``
for raw output (agent/script friendly). Run-scoped reads (``logs``, ``scalars``)
live under ``runs`` rather than top-level for a single coherent group.
"""

from __future__ import annotations

import argparse
import textwrap

from benchmax import config
from benchmax.cli._client import handle_errors, trainer_client
from benchmax.cli._output import fmt_value, print_json, render_table


def _run_url(run_id: str) -> str:
    return f"{config.web_app_url()}/train/{run_id}"


def _print_run(run: dict) -> None:
    """Key-value view of a run, mirroring the web-app run-detail fields."""
    fields = [
        ("ID", run.get("id")),
        ("Name", run.get("name")),
        ("Status", run.get("status")),
        ("Created", run.get("createdAt")),
        ("Updated", run.get("updatedAt")),
        ("Public", run.get("isPublic")),
        ("Owner", run.get("isOwner")),
        ("Has launcher job", run.get("hasLauncherJob")),
        ("Total steps", run.get("totalSteps")),
        ("Latest activity", run.get("latestActivityMessage")),
    ]
    width = max(len(label) for label, _ in fields)
    for label, value in fields:
        if value is not None:
            print(f"{label:<{width}}  {value}")
    if run.get("launcherArgs"):
        print(f"{'Launcher args':<{width}}  {run['launcherArgs']}")
    print(f"{'URL':<{width}}  {_run_url(run.get('id', ''))}")


@handle_errors
def _cmd_runs_list(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        runs = client.list_runs()
    if args.json:
        print_json(runs)
        return 0
    if not runs:
        print("No runs found.")
        return 0
    rows = [
        [
            r.get("id", ""),
            r.get("name") or "-",
            r.get("status", ""),
            r.get("createdAt", ""),
        ]
        for r in runs
    ]
    render_table(["ID", "NAME", "STATUS", "CREATED"], rows)
    return 0


@handle_errors
def _cmd_runs_get(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        run = client.get_run(args.run_id, include_config=args.config)
    if args.json:
        print_json(run)
        return 0
    _print_run(run)
    return 0


@handle_errors
def _cmd_runs_status(args: argparse.Namespace) -> int:
    # No dedicated /status route — status is the run object's `status` field;
    # /details adds latestStep for progress.
    with trainer_client() as client:
        run = client.get_run(args.run_id)
        details = client.get_run_details(args.run_id)
    if args.json:
        print_json({"status": run.get("status"), **details})
        return 0
    latest_step = details.get("latestStep")
    total = run.get("totalSteps")
    print(f"Status:  {run.get('status')}")
    if latest_step is not None and total:
        print(f"Step:    {latest_step} / {total - 1}")
    elif latest_step is not None:
        print(f"Step:    {latest_step}")
    if run.get("latestActivityMessage"):
        print(f"Latest:  {run['latestActivityMessage']}")
    if details.get("errorCount"):
        print(f"Errors:  {details['errorCount']}")
    print(f"URL:     {_run_url(args.run_id)}")
    return 0


@handle_errors
def _cmd_runs_scalars(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        mode = args.mode
        if mode is None:
            modes = client.get_run_details(args.run_id).get("modes") or []
            if not modes:
                print("No scalars yet (the run has produced no scalar modes).")
                return 0
            mode = "train" if "train" in modes else modes[0]
        scalars = client.get_run_scalars(args.run_id, mode)
    if args.json:
        print_json({"mode": mode, "scalars": scalars})
        return 0
    if not scalars:
        print(f"No scalars for mode '{mode}'.")
        return 0
    print(f"Scalars (mode={mode}):")
    rows = []
    for name, series in scalars.items():  # server returns keys sorted
        last = series[-1] if series else {}
        rows.append(
            [name, len(series), last.get("step", ""), fmt_value(last.get("value"))]
        )
    render_table(["SCALAR", "POINTS", "LAST STEP", "LAST VALUE"], rows)
    return 0


@handle_errors
def _cmd_runs_logs(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        logs = client.get_environment_logs(args.run_id, rollout_id=args.rollout_id)
    if args.json:
        print_json(logs)
        return 0
    if not logs:
        print("No logs.")
        return 0
    for entry in logs:
        print(
            f"{entry.get('createdAt', '')} [{entry.get('level', '')}] {entry.get('content', '')}"
        )
        if entry.get("traceback"):
            print(textwrap.indent(entry["traceback"], "    "))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the `runs` group to the top-level subparsers."""
    runs = sub.add_parser("runs", help="Inspect training runs")
    runs_sub = runs.add_subparsers(
        dest="runs_command", required=True, metavar="<subcommand>"
    )

    p_list = runs_sub.add_parser("list", help="List your training runs")
    p_list.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_list.set_defaults(func=_cmd_runs_list)

    p_get = runs_sub.add_parser("get", help="Show a run's details")
    p_get.add_argument("run_id")
    p_get.add_argument("--config", action="store_true", help="Include launcher args")
    p_get.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_get.set_defaults(func=_cmd_runs_get)

    p_status = runs_sub.add_parser("status", help="Show a run's status + progress")
    p_status.add_argument("run_id")
    p_status.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_status.set_defaults(func=_cmd_runs_status)

    p_scalars = runs_sub.add_parser("scalars", help="Show a run's scalar metrics")
    p_scalars.add_argument("run_id")
    p_scalars.add_argument(
        "--mode", help="Scalar mode (default: train, else first available)"
    )
    p_scalars.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_scalars.set_defaults(func=_cmd_runs_scalars)

    p_logs = runs_sub.add_parser("logs", help="Show a run's environment logs")
    p_logs.add_argument("run_id")
    p_logs.add_argument("--rollout-id", dest="rollout_id", help="Filter to one rollout")
    p_logs.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_logs.set_defaults(func=_cmd_runs_logs)
