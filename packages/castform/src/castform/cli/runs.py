"""castform ``runs`` command group — read-only run lifecycle (slice 1.2).

Thin GETs over platform REST. ``list`` needs login (hard 401); the per-run reads
are optionalAuth, so public runs work logged-out. All commands take ``--json``
for raw output (agent/script friendly). Run-scoped reads (``logs``, ``scalars``)
live under ``runs`` rather than top-level for a single coherent group.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from castform import config
from castform.cli._client import handle_errors, trainer_client
from castform.cli._output import (
    final_answer,
    fmt_value,
    print_json,
    render_table,
    truncate,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from *path* for the optional gold join."""
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _row_question_and_gold(row: Any) -> tuple[object, object]:
    """Return the question and gold answer from supported local dataset rows."""
    if not isinstance(row, dict):
        return None, None
    question = row.get("prompt") or row.get("question")
    if isinstance(question, list):
        question = next(
            (
                message.get("content")
                for message in reversed(question)
                if isinstance(message, dict)
                and message.get("role") == "user"
                and message.get("content")
            ),
            None,
        )
    gold = row.get("ground_truth")
    if gold is None:
        gold = row.get("answer")
    return question, gold


def _run_url(run_id: str) -> str:
    return f"{config.web_app_url()}/train/{run_id}"


# --- stored-rollout helpers (runs rollouts / rollout) -----------------------
# `truncate` + `final_answer` are shared with the validate audit (castform.cli._output).


def _user_prompt(messages: list | None) -> str | None:
    """The last user-turn content from a promptMessages list."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            c = m["content"]
            return c if isinstance(c, str) else str(c)
    return None


def _gold_index(dataset_paths: list[str]) -> dict[str, object]:
    """normalized question text → gold, from the first readable local dataset.

    Gold isn't in the rollout payload; join it back from the local jsonl by
    question text (the env built the rollout prompt from the row's question)."""
    for path in dataset_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            rows = _load_jsonl(p)
        except (OSError, json.JSONDecodeError):
            continue
        index: dict[str, object] = {}
        for r in rows:
            q, gold = _row_question_and_gold(r)
            if isinstance(q, str) and gold is not None:
                index[" ".join(q.split())] = gold
        if index:
            return index
    return {}


def _match_gold(prompt_text: str | None, index: dict[str, object]) -> object:
    """Gold lookup by EXACT normalized question match. Deliberately not fuzzy: the
    flagship env's rollout user turn IS the raw dataset question (exact hits), and a
    substring/containment fallback confidently attaches the WRONG gold when the
    rollout's true question isn't in the local file — worse than showing none."""
    if not prompt_text or not index:
        return None
    return index.get(" ".join(prompt_text.split()))


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
        # details spread first so the run's own status field wins, not /details'.
        print_json({**details, "status": run.get("status")})
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


@handle_errors
def _cmd_runs_rollouts(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        if args.example:
            # One example's rollouts across steps (heatmap). Ids feed `runs rollout`.
            rollouts = client.get_rollout_heatmap(
                args.run_id, args.example, mode=args.mode
            )
            if args.json:
                print_json(rollouts)
                return 0
            if not rollouts:
                print(f"No rollouts for example {args.example} (mode={args.mode}).")
                return 0
            rows = [
                [r.get("id", ""), r.get("step", ""), fmt_value(r.get("totalReward"))]
                for r in rollouts
            ]
            render_table(["ROLLOUT ID", "STEP", "REWARD"], rows)
            print(f"\nInspect one:  castform runs rollout {args.run_id} <ROLLOUT ID>")
            return 0

        summary = client.get_rollout_summary(
            args.run_id, mode=args.mode, limit=args.limit
        )
        avg = client.get_rollout_mode_average(args.run_id, mode=args.mode)
    if args.json:
        print_json({"mode": args.mode, "mode_average": avg, "examples": summary})
        return 0
    if not summary:
        print(f"No {args.mode} rollouts yet.")
        return 0
    mean = avg.get("avg") if isinstance(avg, dict) else None
    head = f"Rollout examples (mode={args.mode}"
    head += f", avg reward {fmt_value(mean)})" if mean is not None else ")"
    print(head + ":")
    rows = []
    for g in summary:
        hist = g.get("rewardHistory") or []
        last = hist[-1].get("meanReward") if hist else None
        rows.append(
            [
                g.get("promptMessageId", ""),
                fmt_value(last) if last is not None else "-",
                truncate(g.get("promptText"), 60),
            ]
        )
    render_table(["EXAMPLE ID", "MEAN REWARD", "PROMPT"], rows)
    print(
        f"\nList an example's rollouts:  "
        f"castform runs rollouts {args.run_id} --example <EXAMPLE ID>"
    )
    return 0


@handle_errors
def _cmd_runs_rollout(args: argparse.Namespace) -> int:
    with trainer_client() as client:
        details = client.get_rollout_details(args.run_id, args.rollout_id)

    prompt = _user_prompt(details.get("promptMessages"))
    datasets = (
        [args.dataset]
        if args.dataset
        else ["eval_dataset.jsonl", "train_dataset.jsonl"]
    )
    gold = _match_gold(prompt, _gold_index(datasets))

    if args.json:
        print_json({**details, "gold": gold})
        return 0

    print(
        f"Rollout {args.rollout_id}  "
        f"(step {details.get('step', '?')}, "
        f"total {fmt_value(details.get('totalReward'))})"
    )
    if prompt:
        print(f"\nQ:    {truncate(prompt, 400)}")
    if gold is not None:
        print(f"gold: {truncate(gold, 400)}")
    else:
        print("gold: (not found locally — pass --dataset to join ground truth)")
    answer = final_answer(details.get("messages"))
    if answer:
        print(f"\nanswer:\n{answer}")
    rewards = details.get("rewards") or []
    if rewards:
        print("\nper-component rewards:")
        for r in rewards:
            print(f"  {str(r.get('name', '')):<24} {fmt_value(r.get('value'))}")
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

    p_rollouts = runs_sub.add_parser(
        "rollouts", help="List stored rollout examples (or one example's rollouts)"
    )
    p_rollouts.add_argument("run_id")
    p_rollouts.add_argument(
        "--mode", default="eval", help="train | eval | external-eval (default: eval)"
    )
    p_rollouts.add_argument(
        "--example",
        help="Show one example's rollouts across steps (its promptMessageId)",
    )
    p_rollouts.add_argument(
        "--limit", type=int, default=50, help="Max examples to list (default: 50)"
    )
    p_rollouts.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_rollouts.set_defaults(func=_cmd_runs_rollouts)

    p_rollout = runs_sub.add_parser(
        "rollout", help="Show one stored rollout: transcript + rewards + gold"
    )
    p_rollout.add_argument("run_id")
    p_rollout.add_argument("rollout_id")
    p_rollout.add_argument(
        "--dataset",
        help="Local jsonl to join gold from (default: eval_dataset.jsonl, then train)",
    )
    p_rollout.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_rollout.set_defaults(func=_cmd_runs_rollout)
