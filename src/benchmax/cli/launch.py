"""castform launch — validate, upload, and launch a training run (slice 1.5).

Wraps the SDK launch flow (validate_env → upload_training_run →
launch_training_run) from a project dir. The accepted launcher args + their
defaults/soft-caps are driven off ``TrainerClient.list_launch_args()`` (never
hardcoded) — the wire knob is ``max_rollout_len``, not ``max_response_len``, and
the server rejects unknown keys. Launch-response warnings are surfaced.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

from benchmax import config
from benchmax.cli._client import handle_errors
from benchmax.cli._output import print_json
from benchmax.cli._preflight import print_project_error
from benchmax.cli._project import ProjectError, load_project
from benchmax.cli._providers import provider_choices, resolve_pip_dependencies
from benchmax.cli.validate import _parse_env_args
from benchmax.platform.client import LaunchArgSpec, TrainerClient


def _coerce_arg(spec: LaunchArgSpec, raw: str):
    try:
        if spec.type == "integer":
            return int(raw)
        if spec.type == "number":
            return float(raw)
        if spec.type == "boolean":
            return raw.strip().lower() in ("1", "true", "yes", "on")
    except ValueError:
        raise SystemExit(f"--set {spec.name}: expected {spec.type}, got {raw!r}")
    return raw


def _check_launcher_value(spec: LaunchArgSpec, key: str, value, *, source: str) -> None:
    """Validate one launcher value against its spec (enum/min/max), warning on the
    soft cap. ``source`` (e.g. ``--set`` or ``LAUNCH_CONFIG``) tags the error."""
    if spec.enum and str(value) not in spec.enum:
        raise SystemExit(f"{source} {key}: must be one of {list(spec.enum)}")
    if spec.min is not None and isinstance(value, (int, float)) and value < spec.min:
        raise SystemExit(f"{source} {key}: below min {spec.min}")
    if spec.max is not None and isinstance(value, (int, float)) and value > spec.max:
        raise SystemExit(f"{source} {key}: above max {spec.max}")
    if (
        spec.warn_above is not None
        and isinstance(value, (int, float))
        and value > spec.warn_above
    ):
        print(
            f"⚠ {key}={value} exceeds the soft cap {spec.warn_above}", file=sys.stderr
        )


def _build_launcher_args(specs: list[LaunchArgSpec], pairs: list[str] | None) -> dict:
    """Validate/coerce ``--set key=value`` against the platform's arg schema."""
    index = {s.name: s for s in specs}
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        spec = index.get(key)
        if spec is None:
            known = ", ".join(sorted(index)) or "(none)"
            raise SystemExit(f"Unknown launch arg {key!r}. Accepted: {known}")
        value = _coerce_arg(spec, raw)
        _check_launcher_value(spec, key, value, source="--set")
        out[key] = value
    return out


# LAUNCH_CONFIG keys resolved on their own (not launcher args passed to the server).
# `model` is the TRAINING model and IS a real launcher arg — it must flow through to
# the server, so it is NOT reserved. (`type` is a removed knob; filtered if present.)
_LAUNCH_CONFIG_RESERVED = frozenset({"type", "name"})


def _launcher_args_from_config(specs: list[LaunchArgSpec], config: dict) -> dict:
    """Launcher args baked into main.py's ``LAUNCH_CONFIG`` (already typed), validated
    against the schema. Unknown keys warn + skip (forgiving, unlike ``--set``) so the
    file can carry a knob a given server hasn't shipped yet; out-of-range hard-fails."""
    index = {s.name: s for s in specs}
    out: dict = {}
    for key, value in config.items():
        if key in _LAUNCH_CONFIG_RESERVED:
            continue
        spec = index.get(key)
        if spec is None:
            print(
                f"⚠ main.py LAUNCH_CONFIG: unknown launch arg {key!r} — ignoring "
                "(see --list-args)",
                file=sys.stderr,
            )
            continue
        _check_launcher_value(spec, key, value, source="LAUNCH_CONFIG")
        out[key] = value
    return out


# Coarse per-turn token budget for the generic fallback estimate (when the env
# provides no estimate_rollout_tokens()). Env-supplied estimates are preferred.
_GENERIC_TOKENS_PER_TURN = 1024


def _env_token_estimate(env_class: type, env_args: dict) -> int | None:
    """The env's own `estimate_rollout_tokens()` if it overrides it — else None.
    Constructs the env best-effort (no network at __init__ for the scaffold envs);
    any failure degrades to None so the launch guard falls back to a coarse estimate."""
    from benchmax.envs.base_env import BaseEnv

    if not (
        isinstance(env_class, type)
        and issubclass(env_class, BaseEnv)
        and env_class.estimate_rollout_tokens is not BaseEnv.estimate_rollout_tokens
    ):
        return None
    try:
        est = env_class(**(env_args or {})).estimate_rollout_tokens()
        return int(est) if isinstance(est, (int, float)) and est > 0 else None
    except Exception:
        return None


def _estimate_rollout_tokens(
    env_class: type, env_args: dict, launcher_args: dict
) -> tuple[int | None, str]:
    """(estimate, source) for the pre-launch truncation guard. Prefer the env's
    own (env-type-aware) estimate; else a coarse per-turn fallback from max_turns."""
    est = _env_token_estimate(env_class, env_args)
    if est is not None:
        return est, "env estimate"
    max_turns = launcher_args.get("max_turns")
    if isinstance(max_turns, int) and max_turns > 0:
        return max_turns * _GENERIC_TOKENS_PER_TURN, "rough per-turn estimate"
    return None, ""


def _print_token_budget_guard(
    env_class: type, env_args: dict, launcher_args: dict
) -> None:
    """Warn (before the launch confirm) if the estimated rollout tokens exceed
    max_rollout_len — a truncated rollout is silently dropped from the loss."""
    budget = launcher_args.get("max_rollout_len")
    if not isinstance(budget, int):
        return
    est, source = _estimate_rollout_tokens(env_class, env_args, launcher_args)
    if est is None:
        return
    if est > budget:
        print(
            f"⚠ estimated rollout ~{est} tokens ({source}) EXCEEDS max_rollout_len "
            f"{budget} — rollouts will truncate and DROP from the loss. Raise "
            f"max_rollout_len (--set) or shrink the env's context.",
            file=sys.stderr,
        )
    else:
        print(f"  token budget: ~{est} est ({source}) / {budget} max_rollout_len")


def _sha256_rows(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head(base: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (
            out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
        )
    except Exception:
        return None


def _write_run_manifest(
    *,
    directory: str,
    run_file: str,
    module_path: str | None,
    from_file: bool,
    run_id: str,
    run_name: str,
    train_dataset: list[dict],
    eval_dataset: list[dict],
    launcher_args: dict,
) -> Path | None:
    """Write an in-repo run manifest (env + dataset hashes, row counts, launcher
    args, run id, back-referencing the git commit) so `git` answers 'what data /
    reward / budget produced this run'. Best-effort — never fails the launch."""
    try:
        base = Path(directory)
        # Record the ACTUAL env source: the run_file for a file-loaded project, else
        # the importable --module path. Only the file case has bytes to hash.
        env_sha256 = None
        if from_file:
            env_source = run_file
            env_path = base / run_file
            if env_path.exists():
                env_sha256 = hashlib.sha256(env_path.read_bytes()).hexdigest()
        else:
            env_source = module_path or "module"
        manifest = {
            "run_id": run_id,
            "name": run_name,
            "commit": _git_head(base),
            "env_source": env_source,
            "env_from_file": from_file,
            "env_sha256": env_sha256,
            "train_sha256": _sha256_rows(train_dataset),
            "eval_sha256": _sha256_rows(eval_dataset),
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset),
            "launcher_args": launcher_args,
        }
        out_dir = base / ".castform" / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize the server-supplied run_id for the filename (defense-in-depth —
        # no path traversal); the real id is preserved in the manifest body.
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", run_id) or "run"
        path = out_dir / f"{safe_id}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
        return path
    except Exception:
        return None


@handle_errors
def _cmd_launch(args: argparse.Namespace) -> int:
    from benchmax.platform.training_run import upload_training_run
    from benchmax.platform.validation import validate_env

    with TrainerClient() as client:
        if args.list_args:
            client.print_launch_args()
            return 0
        specs = client.list_launch_args()

        try:
            project = load_project(
                directory=args.dir,
                run_file=args.run_file,
                module_path=args.module,
                env_class_name=args.env_class,
                train_file=args.train,
                eval_file=args.eval,
                require_eval=True,
            )
        except ProjectError as exc:
            print_project_error(exc)
            return 1

        env_args = _parse_env_args(args.env_arg)
        # Launcher args: main.py's LAUNCH_CONFIG provides defaults; --set overrides
        # key-by-key — so `castform launch` with no flags reproduces the run baked
        # into the file (budgets, epochs), and a flag still wins for a one-off.
        lc = project.launch_config
        launcher_args = {
            **_launcher_args_from_config(specs, lc),
            **_build_launcher_args(specs, args.set),
        }
        # Effective view: the schema defaults the server will apply, overlaid by what
        # this run explicitly sets — so the token guard sees the real max_rollout_len
        # even when it's omitted, and the manifest records the full config.
        effective_args = {
            **{s.name: s.default for s in specs if s.default is not None},
            **launcher_args,
        }
        # name resolves from the flag, else LAUNCH_CONFIG, else the default.
        run_name = args.name or lc.get("name") or project.env_class.__name__.lower()
        # Pre-flight validate uses the cheap chat model from --model / VALIDATE_CONFIG
        # (NOT LAUNCH_CONFIG's training model), matching `castform validate`.
        validate_model = args.model or project.validate_config.get("model")
        # Resolve once so the pre-flight validate and the upload install the SAME
        # deps (--pip + the env's PIP_DEPENDENCIES slot + --provider's SDK).
        pip_deps = resolve_pip_dependencies(args.pip, project.env_class, args.provider)
        # max_turns defaults to 4 server-side and the trainer never consults
        # recommended_max_* — warn on omit so multi-turn envs aren't silently capped.
        if "max_turns" not in launcher_args:
            print(
                "Note: max_turns not set (no --set max_turns / LAUNCH_CONFIG) — "
                "defaults to 4 (max_tool_calls 8). Set it for multi-turn envs.",
                file=sys.stderr,
            )

        # Pre-confirm truncation guard: warn if the rollout likely blows the token
        # budget (so the user can abort / raise it before spending GPU).
        _print_token_budget_guard(project.env_class, env_args, effective_args)

        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    "Refusing to launch (real GPU spend) without confirmation. "
                    "Re-run with --yes.",
                    file=sys.stderr,
                )
                return 1
            reply = input(f"Launch '{run_name}' — incurs GPU cost. Continue? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 1

        if not args.skip_validate:
            print("Validating env on a rollout subset before launch…")
            report = validate_env(
                env_class=project.env_class,
                env_args=env_args,
                train_dataset=project.train_dataset,
                eval_dataset=project.eval_dataset,
                local_modules=[project.module] if project.from_file else None,
                pip_dependencies=pip_deps,
                local=False,
                api_key=None,
                remote_examples=2,
                group_reward_samples=2,
                llm_model=validate_model,
                # Smoke-test at the SAME turn budget the run will use, so the
                # pre-flight doesn't truncate (and flag a false problem) on a
                # multi-turn/search env. max_tool_calls isn't a launch --set knob,
                # so it stays the server default.
                max_turns=launcher_args.get("max_turns", 4),
                verbose=False,
            )
            if not report.ok:
                print(
                    "✗ Validation failed — fix the env/dataset first "
                    "(`castform validate` for detail).",
                    file=sys.stderr,
                )
                return 1
            print("✓ Validation passed.")

        print("Uploading env + datasets…")
        uploaded = upload_training_run(
            env_class=project.env_class,
            train_dataset=project.train_dataset,
            eval_dataset=project.eval_dataset,
            run_name=run_name,
            constructor_args=env_args,
            pip_dependencies=pip_deps,
            local_modules=[project.module] if project.from_file else None,
        )

        print("Launching…")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_id = client.launch_training_run(
                name=run_name,
                launcher_args=launcher_args or None,
                **dataclasses.asdict(uploaded),
            )
        for w in caught:
            print(f"⚠ {w.message}", file=sys.stderr)

    # In-repo run manifest — the reproducible "what produced this run" record.
    manifest_path = _write_run_manifest(
        directory=args.dir,
        run_file=args.run_file,
        module_path=args.module,
        from_file=project.from_file,
        run_id=run_id,
        run_name=run_name,
        train_dataset=project.train_dataset,
        eval_dataset=project.eval_dataset,
        launcher_args=effective_args,
    )

    url = f"{config.web_app_url()}/train/{run_id}"
    if args.json:
        out = {"run_id": run_id, "name": run_name, "url": url}
        if manifest_path is not None:
            out["manifest"] = str(manifest_path)
        print_json(out)
    else:
        print(f"\n✓ Launched run {run_id}")
        print(f"  {url}")
        print(f"  Track:  castform runs status {run_id}")
        if manifest_path is not None:
            print(
                f"  Manifest: {manifest_path}  (commit it — reproducible run record; "
                "it records launcher_args, so keep secrets out of LAUNCH_CONFIG/--set)"
            )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `launch` verb."""
    p = sub.add_parser("launch", help="Validate, upload, and launch a training run")
    p.add_argument("--dir", default=".", help="Project directory (default: .)")
    # Flag name stays --run-file for back-compat; the convention is now main.py.
    p.add_argument("--run-file", default="main.py", help="Env file (default: main.py)")
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
    p.add_argument("--name", help="Run name (default: the env class name)")
    p.add_argument(
        "--env-arg", action="append", metavar="KEY=VALUE", help="Env constructor arg"
    )
    p.add_argument("--pip", action="append", metavar="DEP", help="Extra pip dependency")
    p.add_argument(
        "--provider",
        choices=provider_choices(),
        help="Inject this provider's SDK into the rollout sandbox; does NOT "
        "configure the corpus (the env reads its config from main.py). Shorthand "
        "for the right --pip deps.",
    )
    p.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Launcher arg (see --list-args)",
    )
    p.add_argument(
        "--list-args", action="store_true", help="List accepted launcher args and exit"
    )
    p.add_argument("--model", help="LLM model for the pre-flight validate")
    p.add_argument(
        "--skip-validate", action="store_true", help="Skip the pre-flight validate"
    )
    p.add_argument(
        "--yes", action="store_true", help="Skip the GPU-cost confirmation prompt"
    )
    p.add_argument("--json", action="store_true", help="Emit raw JSON")
    p.set_defaults(func=_cmd_launch)
