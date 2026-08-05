"""Scaffold Castform guidance for an agent-driven RL project.

Logs you in (no-op if already authed), then writes CLAUDE.md / AGENTS.md, the
per-stage skills into each agent's skills dir (claude → ``.claude/skills/``,
codex → ``.agents/skills/``, with path references retargeted), and a starter
prompt. Environment code is intentionally not generated: the agent selects and
adapts the closest maintained Benchmax example. Does NOT open the agent.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from castform import config
from castform.cli._client import handle_errors
from castform.cli._output import (
    _GREY,
    BLUE,
    ORANGE,
    paint,
    rule_label,
    term_width,
)
from castform.platform import credentials

_SKILLS = (
    "design-environment",
    "generate-data",
    "verify-environment",
    "launch-run",
    "view-progress",
)

# Where each agent looks for the per-stage skills. Claude Code auto-loads
# `.claude/skills/`; codex has no skills auto-load, so we drop the same files
# under `.agents/skills/` and point AGENTS.md at them explicitly.
_SKILLS_DIR = {"claude": ".claude/skills", "codex": ".agents/skills"}


def _retarget(text: str, agent: str) -> str:
    """Rewrite the scaffold's ``.claude/skills`` references to ``agent``'s dir."""
    return text.replace(".claude/skills", _SKILLS_DIR[agent])


# The one prompt we surface in-terminal — kept in sync with GETTING_STARTED.md's
# generic variant and the web onboarding copy. The other variants (rag / traces)
# stay in GETTING_STARTED.md so the terminal stays a single clear call to action.
_PRIMARY_PROMPT = (
    "i want to improve a model on <your task>. inspect the maintained Benchmax "
    "examples, choose the closest task shape, adapt it into this project, validate "
    "the environment, review the results, and propose next steps before launching."
)

# (command, what it does) — the few verbs worth surfacing right after setup.
_QUICK_COMMANDS = (
    ("python main.py validate", "baseline · local group of 2 · no GPU"),
    ("python main.py launch", "train on GPUs · spends credits"),
    ("castform runs status <id>", "monitor a launched run"),
    ("castform guide", "full walkthrough + more prompts"),
)


def _scaffold():
    return resources.files("castform.cli.scaffold")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def _print_get_started() -> None:
    """Render the get-started block: an ``ask your agent`` divider over the one
    prompt to paste (plain indented lines — no box, so it copy-pastes clean),
    then a ``helpful commands`` divider over an unboxed command list. Command and
    prompt text stay uncolored for legibility; only dividers carry brand color.
    Degrades to plain text without color/TTY.
    """
    width = min(term_width() - 2, 72)

    print()
    print("  " + rule_label("ask your agent", ORANGE, width))
    for ln in _wrap(_PRIMARY_PROMPT, width - 2):
        print("    " + paint(ln, italic=True))

    print()
    print("  " + rule_label("helpful commands", BLUE, width))
    cmd_w = max(len(c) for c, _ in _QUICK_COMMANDS)
    for cmd, desc in _QUICK_COMMANDS:
        print("    " + cmd.ljust(cmd_w) + "   " + paint(desc, _GREY))
    print()


def _write(dest: Path, text: str, *, force: bool, log: list[str]) -> bool:
    """Write ``dest`` unless it exists (and not forced). Returns True if written."""
    if dest.exists() and not force:
        log.append(f"  skip (exists): {dest}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    log.append(f"  wrote {dest}")
    return True


def _group_status(label: str, writes: list[bool], detail: str, label_w: int) -> str:
    """One high-level scaffolding line: ✓ when files were written, dim · when all
    were already present, or an ``N new, M kept`` tally for a partial re-run."""
    name = label.ljust(label_w)
    wrote, total = sum(writes), len(writes)
    if wrote == 0:
        dim = paint(name, dim=True)
        return f"  {paint('·', _GREY)} {dim}   {paint('already present', _GREY)}"
    summary = detail if wrote == total else f"{wrote} new, {total - wrote} kept"
    return f"  {paint('✓', BLUE)} {name}   {paint(summary, _GREY)}"


def _login_first(skip: bool) -> None:
    if skip:
        return
    try:
        credentials.platform_bearer()
        jwt = credentials._session_jwt()
        who = (credentials._jwt_claims(jwt).get("email") if jwt else None) or "your account"
        print(f"Signed in as {who} ({config.profile_target()}).")
    except RuntimeError:
        print("Signing in…")
        from castform.platform.login import _login

        _login()
        print(f"✓ Signed in to {config.profile_target()}.")


def _choose_agents(arg: str | None) -> set[str]:
    if arg:
        return {"claude", "codex"} if arg == "both" else {arg}
    return {"claude", "codex"}  # default: scaffold for every supported agent


@handle_errors
def _cmd_setup(args: argparse.Namespace) -> int:
    target = Path(args.dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    print()
    _login_first(args.skip_login)

    agents = _choose_agents(args.agent)
    root = _scaffold()
    instructions = (root / "CLAUDE.md").read_text(encoding="utf-8")
    starter = (root / "STARTER.md").read_text(encoding="utf-8")

    print()
    print(paint(f"Scaffolding {target} for your coding agent…", bold=True))

    log: list[str] = []

    def w(dest: Path, text: str) -> bool:
        return _write(dest, text, force=args.force, log=log)

    def prep(text: str, agent: str) -> str:
        return _retarget(text, agent)

    bodies = [(a, f) for a, f in (("claude", "CLAUDE.md"), ("codex", "AGENTS.md")) if a in agents]

    # 1) agent guides — instruction file(s) + GETTING_STARTED. GETTING_STARTED
    #    references the skills dir, so point it at the primary agent.
    primary = "claude" if "claude" in agents else "codex"
    guide_writes = [w(target / f, prep(instructions, a)) for a, f in bodies]
    guide_writes.append(w(target / "GETTING_STARTED.md", prep(starter, primary)))

    # 2) agent skills — the same per-stage skills under each agent's dir.
    skill_writes: list[bool] = []
    for agent, _f in bodies:
        skills_dir = _SKILLS_DIR[agent].split("/")
        for name in _SKILLS:
            skill = (root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
            dest = target.joinpath(*skills_dir, name, "SKILL.md")
            skill_writes.append(w(dest, prep(skill, agent)))

    if args.verbose:
        print("\n".join(log))
    else:
        n_ag = len(agents)
        groups = [
            ("agent guides", guide_writes, "instructions + getting-started"),
            (
                "agent skills",
                skill_writes,
                f"{len(_SKILLS)} stages × {n_ag} agent{'s' * (n_ag != 1)}",
            ),
        ]
        label_w = max(len(label) for label, _, _ in groups)
        for label, writes, detail in groups:
            print(_group_status(label, writes, detail, label_w))

    print()
    print(paint(f"{target} has been set up for castform and your coding agent.", bold=True))

    _print_get_started()
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `setup` verb."""
    p = sub.add_parser("setup", help="Sign in + scaffold this project for a coding agent")
    p.add_argument("--dir", default=".", help="Project directory to scaffold (default: .)")
    p.add_argument(
        "--agent",
        choices=["claude", "codex", "both"],
        help="Coding agent to scaffold for (default: both)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing scaffold files")
    p.add_argument("--skip-login", action="store_true", help="Don't sign in (scaffold only)")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List every file written/skipped instead of the grouped summary",
    )
    p.set_defaults(func=_cmd_setup)
