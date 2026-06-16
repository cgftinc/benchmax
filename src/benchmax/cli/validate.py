"""castform validate — verify an env + surface rewards and errors (slice 1.4).

Runs ``validate_env``'s real-rollout subset (remote by default — no local deps
needed, matching the scaffold's ``local=False``) and prints what that path used
to discard: per-rollout reward values + mean, and the across-sibling mean group
reward. Reward-fn errors (incl. a bad judge key, incl. the group path) already
fail the report — this surfaces them instead of swallowing.

group_reward None ≠ failure: the env may not override ``compute_group_reward``,
or the server may have skipped it (``/rollout/batch/stream`` undeployed) — and on
a server predating ``group_reward_error`` a green group means "no failure
observed", not "verified". The output says which.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmax.cli._client import handle_errors
from benchmax.cli._output import fmt_value, print_json, render_table
from benchmax.cli._project import ProjectError, load_project
from benchmax.platform.client import _mean_rewards


def _parse_env_args(pairs: list[str] | None) -> dict:
    """``--env-arg key=value`` (repeatable); value parsed as JSON, else string."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--env-arg must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _fmt_rewards(rewards: dict | None) -> str:
    if not rewards:
        return "(none)"
    return ", ".join(f"{k}={fmt_value(v)}" for k, v in rewards.items())


def _report_to_dict(report) -> dict:
    remote = report.remote
    return {
        "ok": report.ok,
        "local_ran": report.local_ran,
        "local_passed": report.local_passed,
        "local_failed": report.local_failed,
        "remote_ran": report.remote_ran,
        "examples": [
            {"index": e.index, "ok": e.ok, "error": e.error, "rewards": e.rewards}
            for e in (remote.examples if remote else [])
        ],
        "group_reward": (
            None
            if not remote or remote.group_reward is None
            else {
                "ok": remote.group_reward.ok,
                "error": remote.group_reward.error,
                "rewards": remote.group_reward.rewards,
            }
        ),
    }


def _print_report(report) -> None:
    if report.local_ran:
        status = "passed" if report.local_ok else "FAILED"
        print(
            f"Local contract checks: {status} "
            f"({report.local_passed} passed, {report.local_failed} failed)"
        )

    remote = report.remote
    if remote is None:
        print("Remote rollout subset did not run (no reward values to show).")
    else:
        print("\nPer-rollout rewards:")
        rows = []
        ok_rewards = []
        for ex in remote.examples:
            if ex.ok:
                ok_rewards.append(ex.rewards or {})
                rows.append([ex.index, "ok", _fmt_rewards(ex.rewards)])
            else:
                rows.append([ex.index, "FAILED", ex.error or "rollout failed"])
        render_table(["EXAMPLE", "RESULT", "REWARDS / ERROR"], rows)
        mean = _mean_rewards(ok_rewards)
        if mean:
            print(f"Mean reward: {_fmt_rewards(mean)}")

        group = remote.group_reward
        if group is None:
            print(
                "\nGroup reward: not run — env doesn't override compute_group_reward, "
                "or the server skipped it (group path not verified)."
            )
        elif not group.ok:
            print(f"\nGroup reward: FAILED — {group.error}")
        else:
            print(f"\nGroup reward: ok — mean {_fmt_rewards(group.rewards)}")

    print()
    print("✓ validate passed" if report.ok else "✗ validate failed")


@handle_errors
def _cmd_validate(args: argparse.Namespace) -> int:
    from benchmax.platform.validation import validate_env

    try:
        project = load_project(
            directory=args.dir,
            run_file=args.run_file,
            module_path=args.module,
            env_class_name=args.env_class,
            train_file=args.train,
            eval_file=args.eval,
        )
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Default: remote rollout subset (local=False) — runs compute_reward and
    # compute_group_reward for real, no local deps required. --local-only does
    # the offline contract checks instead (needs the env's deps installed here).
    report = validate_env(
        env_class=project.env_class,
        env_args=_parse_env_args(args.env_arg),
        train_dataset=project.train_dataset,
        eval_dataset=project.eval_dataset or None,
        local_modules=[project.module] if project.from_file else None,
        local=args.local_only,
        api_key=None,  # device session via the bearer seam
        remote_examples=args.examples,
        group_reward_samples=args.group_samples,
        llm_model=args.model,
        # The SDK streams rollout events live regardless; --verbose adds
        # validate_env's own summary on top. Default off — our summary is below.
        verbose=args.verbose,
    )

    if args.json:
        print_json(_report_to_dict(report))
        return 0 if report.ok else 1
    print()
    _print_report(report)
    return 0 if report.ok else 1


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `validate` verb."""
    p = sub.add_parser(
        "validate", help="Verify an env on a real-rollout subset; show rewards + errors"
    )
    p.add_argument("--dir", default=".", help="Project directory (default: .)")
    p.add_argument("--run-file", default="run.py", help="Env file (default: run.py)")
    p.add_argument(
        "--module", help="Import an env from a module path instead of --run-file"
    )
    p.add_argument(
        "--env-class", help="Env class name (when the module defines several)"
    )
    p.add_argument(
        "--train", default="train_dataset.jsonl", help="Train dataset (jsonl)"
    )
    p.add_argument("--eval", default="eval_dataset.jsonl", help="Eval dataset (jsonl)")
    p.add_argument(
        "--env-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Env constructor arg (repeatable)",
    )
    p.add_argument(
        "--model", help="LLM model for the rollout subset (default: cheap nano)"
    )
    p.add_argument(
        "--examples", type=int, default=2, help="Examples to roll out (default: 2)"
    )
    p.add_argument(
        "--group-samples",
        type=int,
        default=2,
        help="Group-reward sibling count (default: 2)",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Run offline contract checks only (no rollouts; needs env deps installed)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Also show validate_env's own progress summary (rollout events always stream)",
    )
    p.add_argument("--json", action="store_true", help="Emit raw JSON")
    p.set_defaults(func=_cmd_validate)
