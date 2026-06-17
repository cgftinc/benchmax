"""castform setup — scaffold a project for an agent-driven RL run (slice 1.8).

Logs you in (no-op if already authed), then writes the agent scaffold from the
packaged templates (``benchmax/cli/scaffold``): CLAUDE.md / AGENTS.md, the
per-stage skills into ``.claude/skills/``, a starter prompt, and a working
starter env (``run.py`` + ``train_dataset.jsonl`` / ``eval_dataset.jsonl``) so the
first ``castform validate`` is green out of the box. Does NOT open the agent.
The scaffold prose duplicates the web-app generator (``buildAgentContextBody``)
for now — accepted divergence debt; keep aligned.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from benchmax import config
from benchmax.cli._client import handle_errors
from benchmax.platform import credentials

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

# Starter env + datasets, written to the project root (unless --no-template). A
# working run.py so the first `castform validate` is green; see scaffold/run.py.
_TEMPLATES = (
    "run.py",
    "train_dataset.jsonl",
    "eval_dataset.jsonl",
)


def _scaffold():
    return resources.files("benchmax.cli.scaffold")


def _write(dest: Path, text: str, *, force: bool, log: list[str]) -> None:
    if dest.exists() and not force:
        log.append(f"  skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    log.append(f"  wrote {dest}")


def _login_first(skip: bool) -> None:
    if skip:
        return
    try:
        credentials.platform_bearer()
        jwt = credentials._session_jwt()
        who = (
            credentials._jwt_claims(jwt).get("email") if jwt else None
        ) or "your account"
        print(f"Signed in as {who} ({config.base_domain()}).")
    except RuntimeError:
        print("Signing in…")
        from benchmax.platform.login import _login

        _login()
        print(f"✓ Signed in to {config.base_domain()}.")


def _choose_agents(arg: str | None) -> set[str]:
    if arg:
        return {"claude", "codex"} if arg == "both" else {arg}
    if sys.stdin.isatty():
        reply = (
            input("Which coding agent? [claude/codex/both] (default both): ")
            .strip()
            .lower()
        )
        if reply in ("claude", "codex"):
            return {reply}
    return {"claude", "codex"}  # default: write for both (same prose, harmless)


@handle_errors
def _cmd_setup(args: argparse.Namespace) -> int:
    target = Path(args.dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    # `--template rag` must not silently leave a stale non-rag run.py in place — it
    # would still `validate` green and masquerade as a rag baseline. Fail loudly
    # (require a clean dir or --force) instead of the usual skip-if-exists.
    run_py = target / "run.py"
    if (
        args.template == "rag"
        and not args.no_template
        and run_py.exists()
        and not args.force
    ):
        print(
            f"Error: {run_py} already exists — refusing to leave a non-rag run.py "
            "in place for --template rag (it would still validate green and look "
            "like a working rag baseline). Re-run with --force to replace it, or "
            "use a clean directory.",
            file=sys.stderr,
        )
        return 1

    _login_first(args.skip_login)

    agents = _choose_agents(args.agent)
    root = _scaffold()
    instructions = (root / "CLAUDE.md").read_text(encoding="utf-8")
    starter = (root / "STARTER.md").read_text(encoding="utf-8")

    log: list[str] = []
    if "claude" in agents:
        _write(target / "CLAUDE.md", instructions, force=args.force, log=log)
        for name in _SKILLS:
            skill = (root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
            _write(
                target / ".claude" / "skills" / name / "SKILL.md",
                skill,
                force=args.force,
                log=log,
            )
    if "codex" in agents:
        _write(target / "AGENTS.md", instructions, force=args.force, log=log)
    _write(target / "GETTING_STARTED.md", starter, force=args.force, log=log)

    # Starter env (the fast-path core). --no-template = docs only, for seasoned
    # users bringing their own run.py.
    if not args.no_template:
        if args.template == "rag":
            # rag: write only run.py (the SearchEnv). The datasets come from
            # `castform data qa-gen` — generic QA rows are the wrong shape for
            # retrieval. The existing-run.py case already failed fast above.
            _write(
                target / "run.py",
                (root / "rag_run.py").read_text(encoding="utf-8"),
                force=args.force,
                log=log,
            )
        else:
            # generic: run.py + starter datasets, all skip-if-exists.
            for name in _TEMPLATES:
                _write(
                    target / name,
                    (root / name).read_text(encoding="utf-8"),
                    force=args.force,
                    log=log,
                )

    print(f"\nScaffolded {target} for: {', '.join(sorted(agents))}")
    print("\n".join(log))
    print("\n" + "─" * 60)
    print(starter)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `setup` verb."""
    p = sub.add_parser(
        "setup", help="Sign in + scaffold this project for a coding agent"
    )
    p.add_argument(
        "--dir", default=".", help="Project directory to scaffold (default: .)"
    )
    p.add_argument(
        "--agent",
        choices=["claude", "codex", "both"],
        help="Coding agent (default: ask, else both)",
    )
    p.add_argument(
        "--force", action="store_true", help="Overwrite existing scaffold files"
    )
    p.add_argument(
        "--template",
        choices=["generic", "rag"],
        default="generic",
        help="Starter env: 'generic' single-turn QA, or 'rag' SearchEnv "
        "(default: generic)",
    )
    p.add_argument(
        "--no-template",
        action="store_true",
        help="Docs only — skip the starter run.py + datasets",
    )
    p.add_argument(
        "--skip-login", action="store_true", help="Don't sign in (scaffold only)"
    )
    p.set_defaults(func=_cmd_setup)
