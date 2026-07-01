"""castform ``runs`` command group — read-only run lifecycle (slice 1.2).

Thin GETs over platform REST. ``list`` needs login (hard 401); the per-run reads
are optionalAuth, so public runs work logged-out. All commands take ``--json``
for raw output (agent/script friendly). Run-scoped reads (``logs``, ``scalars``)
live under ``runs`` rather than top-level for a single coherent group.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from benchmax import config
from benchmax.cli._client import handle_errors, trainer_client
from benchmax.cli._output import fmt_value, print_json, render_table
from benchmax.cli._project import ProjectError, _load_jsonl


def _run_url(run_id: str) -> str:
    return f"{config.web_app_url()}/train/{run_id}"


def _trunc(text: object, n: int) -> str:
    """Collapse whitespace and clip to ``n`` chars for one-line table cells."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --- stored-rollout helpers (runs rollouts / rollout) -----------------------


def _user_prompt(messages: list | None) -> str | None:
    """The last user-turn content from a promptMessages list."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            c = m["content"]
            return c if isinstance(c, str) else str(c)
    return None


def _final_answer(messages: list | None) -> str | None:
    """The last assistant-turn content — the model's committed answer."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return None


def _gold_index(dataset_paths: list[str]) -> dict[str, object]:
    """normalized prompt text → gold, from the first readable local dataset.

    Gold isn't in the rollout payload; join it back from the local jsonl by
    prompt text (the env built the rollout prompt from the row's ``prompt``)."""
    for path in dataset_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            rows = _load_jsonl(p)
        except ProjectError:
            continue
        index: dict[str, object] = {}
        for r in rows:
            # Generic datasets key the question under 'prompt'; the flagship RAG
            # datasets (qa-gen output) use 'question' / 'answer'. Accept both.
            q = r.get("prompt")
            if not q:
                q = r.get("question")
            if isinstance(q, list):
                q = _user_prompt(q)
            gold = r.get("ground_truth")
            if gold is None:
                gold = r.get("answer")
            if isinstance(q, str) and gold is not None:
                index[" ".join(q.split())] = gold
        if index:
            return index
    return {}


def _match_gold(prompt_text: str | None, index: dict[str, object]) -> object:
    """Best-effort gold lookup: exact normalized match, else the LONGEST dataset
    question that appears inside the rollout prompt (the prompt may wrap the
    question in a template). Longest-wins so 'reset password' can't shadow the
    more specific 'reset password on mobile'; one-directional so a short prompt
    can't spuriously match a longer question."""
    if not prompt_text or not index:
        return None
    key = " ".join(prompt_text.split())
    if key in index:
        return index[key]
    best_k, best_v = "", None
    for k, v in index.items():
        if k and k in key and len(k) > len(best_k):
            best_k, best_v = k, v
    return best_v


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
                _trunc(g.get("promptText"), 60),
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

    if args.view:
        from benchmax.cli.dataview import build_view_model, write_html
        from benchmax.platform import browser

        record = {
            "id": args.rollout_id,
            "messages": details.get("messages") or [],
            "scores": {
                r.get("name"): r.get("value") for r in details.get("rewards") or []
            },
            "metadata": {
                "run_id": args.run_id,
                "step": details.get("step"),
                "total_reward": details.get("totalReward"),
                "gold": gold,
            },
        }
        model = build_view_model(
            [record], source=f"rollout {args.rollout_id}", type_override="traces"
        )
        out = write_html(model, Path(f"rollout-{args.rollout_id}.html"))
        print(f"Wrote {out}")
        browser.maybe_open_browser(out.as_uri())
        return 0

    if args.json:
        print_json({**details, "gold": gold})
        return 0

    print(
        f"Rollout {args.rollout_id}  "
        f"(step {details.get('step', '?')}, "
        f"total {fmt_value(details.get('totalReward'))})"
    )
    if prompt:
        print(f"\nQ:    {_trunc(prompt, 400)}")
    if gold is not None:
        print(f"gold: {_trunc(gold, 400)}")
    else:
        print("gold: (not found locally — pass --dataset to join ground truth)")
    answer = _final_answer(details.get("messages"))
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
    p_rollout.add_argument(
        "--view", action="store_true", help="Open the rollout in the HTML viewer"
    )
    p_rollout.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_rollout.set_defaults(func=_cmd_runs_rollout)
