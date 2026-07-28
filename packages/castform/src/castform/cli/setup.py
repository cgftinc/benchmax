"""castform setup — scaffold a project for an agent-driven RL run (slice 1.8).

Logs you in (no-op if already authed), then writes the agent scaffold from the
packaged templates (``castform.cli.scaffold``): CLAUDE.md / AGENTS.md, the
per-stage skills into each agent's skills dir (claude → ``.claude/skills/``,
codex → ``.agents/skills/``, with the body's path references retargeted), a
starter prompt, and a standalone ``pyproject.toml`` + runnable seed ``main.py`` +
tiny seed datasets per template (``generic`` → a minimal single-turn env,
``rag`` → a hosted-corpus search env) so ``python main.py validate`` runs on day
one. ``--no-template`` skips the seed (docs + skills only; the agent writes
``main.py`` from the design-environment skill). Does NOT open the agent.
The scaffold prose duplicates the web-app generator (``buildAgentContextBody``)
for now — accepted divergence debt; keep aligned.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
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

# Per-template seed files in the scaffold dir: a runnable `main.py` + tiny seed
# datasets, copied into the project so `python main.py validate` runs day one.
_TEMPLATE_SEEDS = {
    "generic": {
        "main": "generic_main.py",
        "train": "generic_train_dataset.jsonl",
        "eval": "generic_eval_dataset.jsonl",
        "tests": "generic_env_tests.py",
    },
    "rag": {
        "main": "rag_main.py",
        "train": "rag_train_dataset.jsonl",
        "eval": "rag_eval_dataset.jsonl",
    },
}


def _project_toml(template: str) -> str:
    benchmax_requirement = _installed_requirement("benchmax")
    castform_name = "castform[rag]" if template == "rag" else "castform"
    castform_requirement = _installed_requirement(
        castform_name, distribution="castform"
    )
    return f'''[project]
name = "castform-environment"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "{benchmax_requirement}",
    "{castform_requirement}",
]
'''


def _installed_requirement(name: str, *, distribution: str | None = None) -> str:
    """Pin scaffolds to the package pair that generated them."""

    distribution = distribution or name
    try:
        installed = version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"cannot scaffold without an installed {distribution!r} distribution"
        ) from error
    return f"{name}=={installed}"


def _retarget(text: str, agent: str) -> str:
    """Rewrite the scaffold's ``.claude/skills`` references to ``agent``'s dir."""
    return text.replace(".claude/skills", _SKILLS_DIR[agent])


# Env-conditional surfacing: content between ``<!-- rag:start -->`` and
# ``<!-- rag:end -->`` (HTML comments, invisible when rendered) is RAG-specific —
# a single-source doc that the setup mechanism tailors per template.
_RAG_BLOCK_RE = re.compile(
    r"^[ \t]*<!--\s*rag:start\s*-->.*?^[ \t]*<!--\s*rag:end\s*-->[ \t]*\n?",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_RAG_MARKER_RE = re.compile(
    r"^[ \t]*<!--\s*rag:(?:start|end)\s*-->[ \t]*\n?",
    re.MULTILINE | re.IGNORECASE,
)


def _apply_template_conditionals(text: str, template: str) -> str:
    """Tailor a scaffold doc to the env template. ``--template rag`` KEEPS the
    RAG-specific blocks (dropping just the delimiter comments); every other template
    STRIPS them, so a generic scaffold carries no RAG-specific guidance."""
    if template == "rag":
        return _RAG_MARKER_RE.sub("", text)
    return _RAG_BLOCK_RE.sub("", text)


# The one prompt we surface in-terminal — kept in sync with GETTING_STARTED.md's
# generic variant and the web onboarding copy. The other variants (rag / traces)
# stay in GETTING_STARTED.md so the terminal stays a single clear call to action.
_PRIMARY_PROMPT = (
    "i want to start a training run to improve a model on <your task>. create a "
    "reasonable environment with relevant tools, generate a small synthetic "
    "dataset, run a baseline eval, review the results, and propose next steps to "
    "either iterate or launch."
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
        who = (
            credentials._jwt_claims(jwt).get("email") if jwt else None
        ) or "your account"
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

    # A seed main.py must not silently overwrite (or be masked by) a stale one — the
    # old env would still `validate` green and masquerade as a working baseline. Fail
    # loudly (require a clean dir or --force) instead of the usual skip-if-exists.
    main_py = target / "main.py"
    if not args.no_template and main_py.exists() and not args.force:
        print(
            f"Error: {main_py} already exists — refusing to overwrite it (a stale env "
            "would still validate green and mask your task). Re-run with --force to "
            "replace it, or use a clean directory.",
            file=sys.stderr,
        )
        return 1

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
        # Tailor to the env template (strip RAG blocks for non-rag), then retarget
        # the skills-dir references to the agent.
        return _retarget(_apply_template_conditionals(text, args.template), agent)

    bodies = [
        (a, f)
        for a, f in (("claude", "CLAUDE.md"), ("codex", "AGENTS.md"))
        if a in agents
    ]

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

    # 3) env template — every template ships a standalone pyproject, runnable
    #    main.py, and tiny seed datasets so `python main.py validate` runs on day
    #    one; the agent then
    #    tailors them. --no-template skips the seed (docs + skills only). main.py
    #    honors --force (the guard above cleared it); the datasets ALWAYS
    #    skip-if-exists — --force is only for the main.py guard, and real prepared
    #    data must never be clobbered by the placeholder.
    env_writes: list[bool] = []
    if not args.no_template:
        seed = _TEMPLATE_SEEDS[args.template]
        env_writes.append(
            _write(
                target / "pyproject.toml",
                _project_toml(args.template),
                force=False,
                log=log,
            )
        )
        env_writes.append(
            w(target / "main.py", (root / seed["main"]).read_text("utf-8"))
        )
        env_writes.append(
            _write(
                target / "train.jsonl",
                (root / seed["train"]).read_text("utf-8"),
                force=False,
                log=log,
            )
        )
        # tests/ mirrors the examples' layout: conftest pins the import path,
        # and templates with a deterministic reward seed a reward test to grow.
        env_writes.append(
            _write(
                target / "tests" / "conftest.py",
                (root / "tests_conftest.py").read_text("utf-8"),
                force=False,
                log=log,
            )
        )
        if "tests" in seed:
            env_writes.append(
                _write(
                    target / "tests" / "test_env.py",
                    (root / seed["tests"]).read_text("utf-8"),
                    force=False,
                    log=log,
                )
            )
        env_writes.append(
            _write(
                target / "eval.jsonl",
                (root / seed["eval"]).read_text("utf-8"),
                force=False,
                log=log,
            )
        )

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
        if env_writes:  # every template ships a seed main.py + datasets
            groups.append(
                (
                    "env template",
                    env_writes,
                    f"pyproject + main.py + datasets + tests ({args.template})",
                )
            )
        label_w = max(len(label) for label, _, _ in groups)
        for label, writes, detail in groups:
            print(_group_status(label, writes, detail, label_w))

    print()
    print(
        paint(
            f"{target} has been set up for castform and your coding agent.", bold=True
        )
    )

    _print_get_started()
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
        help="Coding agent to scaffold for (default: both)",
    )
    p.add_argument(
        "--force", action="store_true", help="Overwrite existing scaffold files"
    )
    p.add_argument(
        "--template",
        choices=["generic", "rag"],
        default="generic",
        help="Env seed: 'generic' = a minimal single-turn env, 'rag' = a hosted-"
        "corpus search env (both ship a pyproject, runnable main.py, and tiny "
        "datasets; default: generic)",
    )
    p.add_argument(
        "--no-template",
        action="store_true",
        help="Skip the seed main.py + datasets (scaffold docs + skills only)",
    )
    p.add_argument(
        "--skip-login", action="store_true", help="Don't sign in (scaffold only)"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List every file written/skipped instead of the grouped summary",
    )
    p.set_defaults(func=_cmd_setup)
