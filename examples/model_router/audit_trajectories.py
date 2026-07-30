#!/usr/bin/env python3
"""Scan harbor trial trajectories for ground-truth-cheat attempts.

A rollout can defeat the local leak guard by fetching the upstream repo
(or a released package of it) over the network. This scans every trial's
agent files for *executed-command-shaped* cheat patterns and reports
tainted trials, so their rewards can be excluded or attempt-labeled.

Deliberately command-focused: bare mentions of github URLs occur
legitimately (repo files reference issues/PRs and agents read them), so
URL fetches only count when driven by a fetching tool (git/curl/wget/pip).
harbor's own agent-install script lives under agent/setup/ and is skipped.

Usage: python audit_trajectories.py <jobs_dir...> [--package click]
Exit code 1 if any tainted trial is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REAL_AGENTS = {"claude-code", "codex", "mini-swe-agent", "openhands", "aider"}


def patterns(package: str | None) -> list[tuple[str, re.Pattern[str]]]:
    pats = [
        ("git-remote-read",
         re.compile(r"git\s+(?:--?\S+\s+)*(fetch|clone|ls-remote|pull)\b")),
        ("git-remote-add", re.compile(r"git\s+remote\s+(add|set-url)\b")),
        ("url-fetch",
         re.compile(r"(curl|wget)\s+\S*(github\.com|githubusercontent\.com|"
                    r"codeload\.github\.com|pypi\.org|files\.pythonhosted\.org)",
                    re.I)),
    ]
    if package:
        pats.append((
            "package-install",
            re.compile(rf"(pip3?|uv pip|npm|yarn)\s+(install|download|add)\s+"
                       rf"(-\S+\s+)*{re.escape(package)}\b", re.I),
        ))
    return pats


def trial_agent(trial_dir: Path) -> str:
    try:
        info = json.loads((trial_dir / "result.json").read_text())
        return (info.get("agent_info") or {}).get("name", "?")
    except Exception:
        return "?"


_COMMAND_FIELD = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"')


def extract_commands(text: str) -> list[str]:
    """Pull executed-command strings out of trajectory/session JSON.

    Both codex (command_execution items) and claude-code (Bash tool_use
    input) serialize commands under a "command" key. Scanning only these
    avoids false positives from instruction text quoted in the trajectory
    (our Rules section literally contains "git fetch").
    """
    out = []
    for m in _COMMAND_FIELD.finditer(text):
        try:
            out.append(json.loads(f'"{m.group(1)}"'))
        except Exception:
            out.append(m.group(1))
    return out


def audit_trial(trial_dir: Path,
                pats: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    findings: dict[str, str] = {}
    agent_dir = trial_dir / "agent"
    n_commands = 0
    for f in agent_dir.rglob("*"):
        if not f.is_file() or "setup" in f.relative_to(agent_dir).parts:
            continue
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        for cmd in extract_commands(text):
            n_commands += 1
            for label, pat in pats:
                if label in findings:
                    continue
                m = pat.search(cmd)
                if m:
                    findings[label] = (
                        f"{label}: {m.group(0)[:80]!r} "
                        f"(cmd: {' '.join(cmd.split())[:120]!r}) in {f.name}"
                    )
    if n_commands == 0:
        findings["no-commands-extracted"] = (
            "no-commands-extracted: unknown trajectory format, audit blind"
        )
    return list(findings.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_dirs", nargs="+", type=Path)
    ap.add_argument("--package", default=None,
                    help="repo's package name, e.g. click: flags installing "
                         "it from an index (released code contains the fix)")
    args = ap.parse_args()
    pats = patterns(args.package)

    tainted = n_trials = 0
    for jobs_dir in args.jobs_dirs:
        for result in sorted(jobs_dir.rglob("result.json")):
            trial_dir = result.parent
            if not (trial_dir / "agent").is_dir():
                continue
            if trial_agent(trial_dir) not in REAL_AGENTS:
                continue
            n_trials += 1
            findings = audit_trial(trial_dir, pats)
            if findings:
                tainted += 1
                print(f"TAINTED {trial_dir} ({trial_agent(trial_dir)})")
                for fi in findings:
                    print(f"  {fi}")
    print(f"\naudited {n_trials} real-agent trials: {tainted} tainted")
    return 1 if tainted else 0


if __name__ == "__main__":
    raise SystemExit(main())
