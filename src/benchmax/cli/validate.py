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
import contextlib
import json
import sys

from benchmax.cli._client import handle_errors
from benchmax.cli._output import fmt_value, print_json
from benchmax.cli._project import ProjectError, load_project
from benchmax.cli._providers import provider_choices, resolve_pip_dependencies
from benchmax.platform.client import _VALIDATION_MODEL, _mean_rewards


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


# --- scorecard rendering (same shape every run) -------------------------

_CARD_WIDTH = 52  # header-rule width; the card body is hand-aligned, not boxed


def _rule(label: str = "") -> str:
    """A ``─── label ───`` header rule (plain rule when label is empty)."""
    if not label:
        return "─" * _CARD_WIDTH
    prefix = f"─── {label} "
    return prefix + "─" * max(3, _CARD_WIDTH - len(prefix))


def _std(values: list[float]) -> float:
    """Population std of one reward component across the sampled rollouts."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def _check(symbol: str, label: str, detail: str = "") -> str:
    """One aligned check row: ``  ✓  label                 detail``."""
    return f"  {symbol}  {label:<29}{detail}".rstrip()


def _distinct_errors(remote) -> list[str]:
    """Unique rollout-failure messages, in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for ex in remote.examples:
        if not ex.ok and ex.error and ex.error not in seen:
            seen.add(ex.error)
            out.append(ex.error)
    return out


def _print_rewards(ok_rewards: list[dict]) -> None:
    """Reward components as ``avg``/``std`` rows + a summed total."""
    mean = _mean_rewards(ok_rewards) or {}
    if not mean:
        print("  reward     n/a — no numeric reward components to score")
        return
    comps = sorted(mean)
    name_w = max(len("reward component"), len("total reward"), *(len(c) for c in comps))

    def row(name: str, avg: str, std: str) -> str:
        return f"  {name:<{name_w}}   {avg:>7}  {std:>7}"

    print(row("reward component", "avg", "std"))
    for k in comps:
        vals = [
            r[k]
            for r in ok_rewards
            if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
        ]
        print(row(k, fmt_value(mean[k]), fmt_value(_std(vals))))
    print("  " + "─" * (name_w + 19))
    print(f"  {'total reward':<{name_w}}   {fmt_value(sum(mean.values())):>7}")


def _print_checks(remote, ok_rewards: list[dict]) -> None:
    """The fixed checks block: errors, reward variance, group reward."""
    print("  checks")
    n = len(remote.examples)
    n_ok = sum(1 for ex in remote.examples if ex.ok)
    n_failed = n - n_ok
    if n_failed == 0:
        print(_check("✓", "no reward errors", f"{n_ok}/{n} rollouts ok"))
    else:
        print(
            _check(
                "⚠", "reward errors", f"{n_failed}/{n} rollouts failed (see transcript)"
            )
        )
        for err in _distinct_errors(remote):
            print(f"       {err}")

    # Headline keys off the per-rollout TOTAL (robust to ragged/empty component
    # keys); the per-component list is a softer warning when the total varies.
    flat_total = _constant_total(ok_rewards)
    constant = [
        (name, value)
        for name, value in _constant_components(ok_rewards)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(ok_rewards) < 2:
        print(
            _check("·", "rewards vary across rollouts", "not assessed (<2 ok rollouts)")
        )
    elif flat_total is not None:
        print(
            _check(
                "⚠",
                "rewards DON'T vary",
                f"total reward constant (= {fmt_value(flat_total)})",
            )
        )
    elif constant:
        names = ", ".join(repr(name) for name, _ in constant)
        print(
            _check("⚠", "some components constant", f"{names} never vary (no gradient)")
        )
    else:
        print(_check("✓", "rewards vary across rollouts"))

    group = remote.group_reward
    if group is None:
        print(_check("·", "group reward", "not run (no compute_group_reward)"))
    elif not group.ok:
        print(_check("⚠", "group reward", f"FAILED — {group.error}"))
    else:
        print(_check("✓", "group reward", f"mean {_fmt_rewards(group.rewards)}"))


def _recommendation(report, ok_rewards: list[dict]) -> str:
    """The single decision line. Keys off variance + errors, not just report.ok —
    so a technically-passing run with no signal (the hollow green) is loud.
    Same ``_constant_total`` gate as the checks block, so the two never disagree."""
    if not report.ok:
        return "→ NOT passing — fix the errors above, then re-validate."
    if _constant_total(ok_rewards) is not None:
        return (
            "⚠ green, but NO training signal — the total reward never varies across "
            "rollouts.\n"
            "  Likely a hollow pass: rows too easy/hard, or the env "
            "(retrieval/judge) is failing.\n"
            "  Read the per-rollout transcript (--full-messages) for a swallowed "
            "Error: before trusting this baseline.\n"
            "  For a provider RAG env, add --provider <name> (or --pip <sdk>) so its "
            "search SDK is in the sandbox."
        )
    return "→ GREEN baseline — iterate (improve reward/data) or launch."


def _ok_rewards(remote) -> list[dict]:
    """Reward dicts for the rollouts that succeeded (basis for mean + variance)."""
    return [ex.rewards or {} for ex in remote.examples if ex.ok]


def _constant_components(ok_rewards: list[dict]) -> list[tuple[str, object]]:
    """Reward components that never vary across the ok rollouts — a constant
    component gives no gradient, so training can't learn from it (the classic
    "all-zero reward" footgun). Needs >=2 rollouts to mean anything; returns
    ``(component, value)`` per offending component, sorted by name."""
    if len(ok_rewards) < 2:
        return []
    keys = sorted({k for r in ok_rewards for k in r})
    constant = []
    for k in keys:
        values = [r[k] for r in ok_rewards if k in r]
        if len(values) >= 2 and len(set(values)) == 1:
            constant.append((k, values[0]))
    return constant


def _constant_total(ok_rewards: list[dict]) -> float | None:
    """The per-rollout TOTAL numeric reward if it never varies across the ok
    rollouts (>=2), else None. A constant total = no advantage to learn from —
    the true hollow-green test. Sums numeric components per rollout, so it catches
    ragged keys / empty reward dicts that a per-component check (which needs a key
    present in >=2 rollouts) silently misses. Bools are excluded, like the table."""
    if len(ok_rewards) < 2:
        return None
    totals = [
        sum(
            v
            for v in r.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        for r in ok_rewards
    ]
    return totals[0] if len(set(totals)) == 1 else None


def _report_to_dict(report) -> dict:
    remote = report.remote
    ok_rewards = _ok_rewards(remote) if remote else []
    return {
        "ok": report.ok,
        "local_ran": report.local_ran,
        "local_passed": report.local_passed,
        "local_failed": report.local_failed,
        "remote_ran": report.remote_ran,
        "examples": [
            {
                "index": e.index,
                "ok": e.ok,
                "error": e.error,
                "rewards": e.rewards,
                "messages": e.messages,
            }
            for e in (remote.examples if remote else [])
        ],
        "warnings": [
            {
                "kind": "constant_reward_component",
                "component": name,
                "value": value,
                "rollouts": len(ok_rewards),
            }
            for name, value in _constant_components(ok_rewards)
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


def _print_report(report, *, env_label: str, model: str, train_label: str) -> None:
    """Render the fixed scorecard: header → reward avg/std → checks → one line."""
    remote = report.remote
    print(_rule("castform validate"))
    print(f"  env        {env_label}")
    if remote is not None:
        n = len(remote.examples)
        print(f"  model      {model}  (cheap eval, no GPU)")
        print(f"  rollouts   {n} example{'' if n == 1 else 's'} · {train_label}")
    print()

    if report.local_ran:
        status = "passed" if report.local_ok else "FAILED"
        print(
            f"  contract checks  {status} "
            f"({report.local_passed} passed, {report.local_failed} failed)"
        )
        print()

    if remote is None:
        if not report.local_ran:
            print("  remote rollout subset did not run — no reward values to show.")
            print()
        print("  ✓ validate passed" if report.ok else "  ✗ validate failed")
        return

    ok_rewards = _ok_rewards(remote)
    _print_rewards(ok_rewards)
    print()
    _print_checks(remote, ok_rewards)
    print()
    print("  ✓ validate passed" if report.ok else "  ✗ validate failed")
    print("  " + _recommendation(report, ok_rewards))


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

    # The SDK streams rollout events to stdout regardless. In --json mode keep
    # stdout clean (machine-readable, pipeable to jq) by routing that stream to
    # stderr; we emit only the JSON object on stdout below.
    stream_sink = (
        contextlib.redirect_stdout(sys.stderr)
        if args.json
        else contextlib.nullcontext()
    )

    # Default: remote rollout subset (local=False) — runs compute_reward and
    # compute_group_reward for real, no local deps required. --local-only does
    # the offline contract checks instead (needs the env's deps installed here).
    with stream_sink:
        report = validate_env(
            env_class=project.env_class,
            env_args=_parse_env_args(args.env_arg),
            train_dataset=project.train_dataset,
            eval_dataset=project.eval_dataset or None,
            local_modules=[project.module] if project.from_file else None,
            pip_dependencies=resolve_pip_dependencies(
                args.pip, project.env_class, args.provider
            ),
            local=args.local_only,
            api_key=None,  # device session via the bearer seam
            remote_examples=args.examples,
            group_reward_samples=args.group_samples,
            llm_model=args.model,
            # Rollout budget — raise both to match an env that advertises a larger
            # search/tool budget (e.g. SearchEnv MAX_SEARCH_CALLS=6) so the rollout
            # isn't truncated below what the system prompt instructs.
            max_turns=args.max_turns,
            max_tool_calls=args.max_tool_calls,
            # --verbose adds validate_env's own summary on top; default off.
            verbose=args.verbose,
            # --full-messages prints untruncated tool/transcript text — needed to
            # read a swallowed search error (e.g. a missing provider SDK) behind a
            # hollow green.
            full_messages=args.full_messages,
        )

    if args.json:
        print_json(_report_to_dict(report))
        return 0 if report.ok else 1
    source = args.run_file if project.from_file else (args.module or "module")
    print()
    _print_report(
        report,
        env_label=f"{project.env_class.__name__} · {source}",
        model=args.model or _VALIDATION_MODEL,
        train_label=args.train,
    )
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
        "--pip",
        action="append",
        metavar="DEP",
        help="Extra pip dependency for the rollout sandbox (repeatable). Needed "
        "for provider RAG envs whose search client imports a provider SDK "
        "(e.g. turbopuffer) — the sandbox bundles only run.py + benchmax.",
    )
    p.add_argument(
        "--provider",
        choices=provider_choices(),
        help="Inject this provider's SDK into the rollout sandbox; does NOT "
        "configure the corpus (the env's search client reads its config from "
        "run.py). Shorthand for the right --pip deps; differs from qa-gen's "
        "--provider, which reads a corpus.",
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
        "--max-turns",
        type=int,
        default=4,
        help="Max conversation turns per rollout (default: 4). Raise it to match a "
        "multi-turn env's budget — e.g. a SearchEnv with MAX_SEARCH_CALLS=6 needs "
        "~7 — or the rollout truncates below what the system prompt instructs",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=8,
        help="Max tool calls per rollout (default: 8). Each search is one tool "
        "call, so raise it alongside --max-turns for tool-heavy envs",
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
    p.add_argument(
        "--full-messages",
        action="store_true",
        help="Print untruncated tool/transcript text — use to read a swallowed "
        "search error (e.g. a missing provider SDK) behind a hollow green",
    )
    p.add_argument("--json", action="store_true", help="Emit raw JSON")
    p.set_defaults(func=_cmd_validate)
