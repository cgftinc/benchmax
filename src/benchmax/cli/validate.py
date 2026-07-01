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
from benchmax.cli._output import final_answer, fmt_value, print_json, truncate
from benchmax.cli._preflight import print_project_error
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


# --- reward audit (--reward-audit): the pre-launch reward inspection -------------
# `truncate` + `final_answer` are shared with `runs rollout` (benchmax.cli._output).


def _example_gold(row: dict | None) -> tuple[object, object]:
    """(question, gold) from a local dataset row — the ground truth the scorecard
    hides. Handles a string prompt or a chat-list prompt; gold from
    ``ground_truth`` (or ``answer``)."""
    if not isinstance(row, dict):
        return None, None
    # Generic datasets key the question under 'prompt'; the flagship RAG datasets
    # (qa-gen output, which this audit most often runs on) use 'question' / 'answer'.
    q = row.get("prompt")
    if not q:
        q = row.get("question")
    if isinstance(q, list):  # chat-style prompt → last user turn
        q = next(
            (m.get("content") for m in reversed(q) if m.get("role") == "user"), None
        )
    gold = row.get("ground_truth")
    if gold is None:
        gold = row.get("answer")
    return q, gold


def _numeric(reward: dict, key: str) -> float | None:
    v = reward.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _correctness_key(components) -> str | None:
    """The component representing answer correctness (the gate). Secondary
    components are ``* correctness``, so redundancy is judged against this one.
    Prefer an exact name, else any key containing 'correct'."""
    keys = list(components)
    for exact in ("answer_correctness", "correctness"):
        if exact in keys:
            return exact
    for k in keys:
        if "correct" in k.lower():
            return k
    return None


def _mirrors_correctness(comp_vals: list[float], corr_vals: list[float]) -> bool:
    """Heuristic: does a component add NO signal beyond correctness? True when it's
    constant *within* every correctness stratum (a deterministic function of
    correctness — e.g. a redundant second judge), given correctness itself varies
    and we saw a stratum with >=2 rollouts. A genuinely-independent gated term
    (``recall * correctness``) varies within a stratum, so it is NOT flagged."""
    from collections import defaultdict

    if len(set(corr_vals)) < 2 or len(set(comp_vals)) < 2:
        return False
    strata: dict[float, list[float]] = defaultdict(list)
    for c, v in zip(corr_vals, comp_vals):
        strata[round(c, 6)].append(round(v, 6))
    if not any(len(v) >= 2 for v in strata.values()):
        return False  # no repeated correctness value → can't conclude
    return all(len(set(v)) == 1 for v in strata.values())


def _audit_components(
    ok_rewards: list[dict], group_names: set[str]
) -> tuple[list[dict], str | None]:
    """Per-component audit rows (avg/std/note) + the detected correctness key.

    ``note`` flags a component as: group-scored (N/A per-example), constant (no
    gradient), mirrors-correctness (redundant), or primary (the gate)."""
    mean = _mean_rewards(ok_rewards) or {}
    corr_key = _correctness_key(mean)
    rows: list[dict] = []
    for k in sorted(mean):
        vals = [v for v in (_numeric(r, k) for r in ok_rewards) if v is not None]
        std = _std(vals)
        note = ""
        if k in group_names:
            note = "group-scored (N/A per-example)"
        elif len(vals) >= 2 and std == 0.0:
            note = "constant — no gradient"
        elif corr_key and k == corr_key:
            note = "primary (gate)"
        elif corr_key:
            pairs = [
                (_numeric(r, corr_key), _numeric(r, k))
                for r in ok_rewards
                if _numeric(r, corr_key) is not None and _numeric(r, k) is not None
            ]
            if len(pairs) >= 2 and _mirrors_correctness(
                [v for _, v in pairs], [c for c, _ in pairs]
            ):
                note = "mirrors correctness — no signal beyond it"
        rows.append({"component": k, "avg": mean[k], "std": std, "note": note})
    return rows, corr_key


def _group_component_names(remote) -> set[str]:
    grp = getattr(remote, "group_reward", None)
    if grp is not None and grp.ok and grp.rewards:
        return set(grp.rewards)
    return set()


def _print_reward_audit(remote, ok_rewards: list[dict], *, dataset, pip_deps) -> None:
    """The pre-launch reward audit: per-component discrimination + N real
    transcripts (question, gold, answer, rewards) — catching redundant/constant
    components and no-answer/hash-citation rollouts the scorecard hides."""
    print(_rule("reward audit"))
    n_ok = len(ok_rewards)
    rows, corr_key = _audit_components(ok_rewards, _group_component_names(remote))
    print(
        f"  {n_ok} ok rollout{'' if n_ok == 1 else 's'} · "
        f"{len(rows)} reward component{'' if len(rows) == 1 else 's'}"
    )
    print(f"  pip bundle   {', '.join(pip_deps) if pip_deps else '(none — base env)'}")
    print()
    if not rows:
        print("  (no numeric reward components to audit)")
    else:
        name_w = max(len("component"), *(len(r["component"]) for r in rows))
        print(f"  {'component':<{name_w}}   {'avg':>7}  {'std':>7}   note")
        for r in rows:
            print(
                (
                    f"  {r['component']:<{name_w}}   {fmt_value(r['avg']):>7}  "
                    f"{fmt_value(r['std']):>7}   {r['note']}"
                ).rstrip()
            )
        if corr_key is None:
            print("  (no correctness component detected — redundancy check skipped)")

    print()
    examples = remote.examples
    print(f"  transcripts ({len(examples)} rolled out)")
    for ex in examples:
        row = dataset[ex.index] if dataset and 0 <= ex.index < len(dataset) else None
        q, gold = _example_gold(row)
        total = sum(
            v
            for v in (ex.rewards or {}).values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        print(f"  ── example {ex.index}  ·  reward {fmt_value(total)} ──")
        if q is not None:
            print(f"     Q:    {truncate(q, 200)}")
        if gold is not None:
            print(f"     gold: {truncate(gold, 200)}")
        if ex.ok:
            ans = final_answer(ex.messages)
            print(f"     ans:  {truncate(ans, 300) if ans else '(no answer captured)'}")
            print(f"     rewards: {_fmt_rewards(ex.rewards)}")
        else:
            print(f"     ✗ failed: {ex.error}")


def _report_to_dict(report, *, audit: bool = False, dataset=None) -> dict:
    remote = report.remote
    ok_rewards = _ok_rewards(remote) if remote else []
    out = {
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
    if audit and remote:
        rows, corr_key = _audit_components(ok_rewards, _group_component_names(remote))
        out["audit"] = {"correctness_component": corr_key, "components": rows}
        # Join gold by index (the rollout of example i uses dataset[i]) so the JSON
        # is a self-sufficient reward audit — gold isn't otherwise in the payload.
        for e in out["examples"]:
            row = (
                dataset[e["index"]]
                if dataset and 0 <= e["index"] < len(dataset)
                else None
            )
            _, e["gold"] = _example_gold(row)
    return out


def _print_report(
    report,
    *,
    env_label: str,
    model: str,
    train_label: str,
    reward_audit: bool = False,
    dataset=None,
    pip_deps=None,
) -> None:
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

    if reward_audit:
        print()
        _print_reward_audit(remote, ok_rewards, dataset=dataset, pip_deps=pip_deps)


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
        print_project_error(exc)
        return 1

    reward_audit = args.reward_audit
    # The audit shows real transcripts (needs message capture) and a clean report,
    # so it implies --full-messages and routes the live rollout stream to stderr.
    full_messages = args.full_messages or reward_audit
    pip_deps = resolve_pip_dependencies(args.pip, project.env_class, args.provider)

    # Resolve each rollout knob: explicit CLI flag → run.py VALIDATE_CONFIG → default.
    # The config block lets a multi-turn env bake in its budget so `castform
    # validate` reproduces the intended run without remembering flags.
    vc = project.validate_config

    def _int_cfg(cli_val, key, default):
        """Resolve an int knob (CLI flag → VALIDATE_CONFIG → default), validating
        the config value's type — a str budget in run.py would otherwise crash
        deep in the SDK instead of failing loudly here."""
        val = cli_val if cli_val is not None else vc.get(key, default)
        if not isinstance(val, int) or isinstance(val, bool):
            raise SystemExit(f"VALIDATE_CONFIG['{key}'] must be an int, got {val!r}")
        return val

    max_turns = _int_cfg(args.max_turns, "max_turns", 4)
    max_tool_calls = _int_cfg(args.max_tool_calls, "max_tool_calls", 8)
    examples = _int_cfg(args.examples, "examples", 2)
    group_samples = _int_cfg(args.group_samples, "group_samples", 2)
    model = args.model if args.model is not None else vc.get("model")

    # The SDK streams rollout events to stdout regardless. In --json (and audit)
    # mode keep stdout clean (machine-readable / the report) by routing that stream
    # to stderr; we emit only the JSON object / scorecard on stdout below.
    stream_sink = (
        contextlib.redirect_stdout(sys.stderr)
        if (args.json or reward_audit)
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
            pip_dependencies=pip_deps,
            local=args.local_only,
            api_key=None,  # device session via the bearer seam
            remote_examples=examples,
            group_reward_samples=group_samples,
            llm_model=model,
            # Rollout budget — raise both to match an env that advertises a larger
            # search/tool budget (e.g. SearchEnv MAX_SEARCH_CALLS=6) so the rollout
            # isn't truncated below what the system prompt instructs. Resolved from
            # the CLI flag, else run.py's VALIDATE_CONFIG, else the default.
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            # --verbose adds validate_env's own summary on top; default off.
            verbose=args.verbose,
            # --full-messages prints untruncated tool/transcript text — needed to
            # read a swallowed search error (e.g. a missing provider SDK) behind a
            # hollow green. --reward-audit implies it (captures answers to show).
            full_messages=full_messages,
        )

    if args.json:
        print_json(
            _report_to_dict(report, audit=reward_audit, dataset=project.train_dataset)
        )
        return 0 if report.ok else 1
    source = args.run_file if project.from_file else (args.module or "module")
    print()
    _print_report(
        report,
        env_label=f"{project.env_class.__name__} · {source}",
        model=model or _VALIDATION_MODEL,
        train_label=args.train,
        reward_audit=reward_audit,
        dataset=project.train_dataset,
        pip_deps=pip_deps,
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
        "--examples",
        type=int,
        default=None,
        help="Examples to roll out (default: run.py VALIDATE_CONFIG, else 2)",
    )
    p.add_argument(
        "--group-samples",
        type=int,
        default=None,
        help="Group-reward sibling count (default: VALIDATE_CONFIG, else 2)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max conversation turns per rollout (default: run.py VALIDATE_CONFIG, "
        "else 4). Raise it to match a multi-turn env's budget — e.g. a SearchEnv "
        "with MAX_SEARCH_CALLS=6 needs ~7 — or the rollout truncates below what the "
        "system prompt instructs",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help="Max tool calls per rollout (default: run.py VALIDATE_CONFIG, else 8). "
        "Each search is one tool call, so raise it alongside --max-turns for "
        "tool-heavy envs",
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
    p.add_argument(
        "--reward-audit",
        action="store_true",
        help="Pre-launch reward audit: per-component avg/std, flags for "
        "constant/redundant (mirrors-correctness) components, and the real "
        "question/gold/answer/rewards per rollout. Implies --full-messages. "
        "Raise --examples for a sharper read.",
    )
    p.add_argument("--json", action="store_true", help="Emit raw JSON")
    p.set_defaults(func=_cmd_validate)
