#!/usr/bin/env python3
"""Prompted-LLM router baseline, adapted from Agent-as-a-Router.

Zero-shot template adapted from arXiv:2606.22902 (Apache-2.0,
src/routing/prompts.py): a router LLM sees the task + a hand-written
model pool description (no historical stats) and picks one model.

Runs over either a gated Harbor task directory or a dataset JSONL. Dataset
mode discovers the available routes and uses the same temporal split as the
other router rungs.

Usage:
  python baseline_router.py harbor_tasks/click [--router-model sonnet]
  python baseline_router.py dataset.jsonl --split-strategy repo-temporal
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# Our pool. Descriptions are deliberately prior-only (public positioning +
# price tier), NOT tuned on our collected outcomes - that information
# belongs to the few-shot/stats baseline rung, not zero-shot.
PROMPT_HEADER = """\
You are a coding task router. Your objective is to maximize the \
performance-cost trade-off: choose the model that achieves the best quality \
for its cost on this task.

## Available Models (sorted by cost, high to low)
"""

PROMPT_FOOTER = """\
## Instructions

Analyze the task and choose the model that maximizes quality relative to cost.
Consider the task's difficulty, language, and complexity.
Prefer cheaper models when quality is comparable.

Respond with ONLY a JSON object:
{"model": "<model_name>", "reasoning": "<brief explanation>"}
"""

MODEL_PRIORS = {
    "claude-opus-5": (4, "Premium. Anthropic's strongest coding model. "
                      "Excels at complex, multi-file changes requiring deep "
                      "reasoning about unfamiliar code."),
    "gpt-5.6-sol": (3, "High. OpenAI's flagship coding model. Strong general "
                    "software engineering and careful test-driven work."),
    "claude-sonnet-4-6": (2, "Mid. Fast and capable on routine engineering "
                          "tasks, balancing speed and quality."),
    "gpt-5.6-terra": (1, "Low. Cost-efficient variant. Competitive on "
                      "well-scoped tasks; weaker on subtle multi-step changes."),
    "claude-haiku-4-5": (0, "Lowest-cost Anthropic model. Best suited to "
                          "simple, tightly scoped changes."),
    "gpt-5.6-luna": (0, "Lowest-cost OpenAI variant. Best suited to simple, "
                     "tightly scoped changes."),
}
MODELS = {"claude-opus-5", "gpt-5.6-sol", "claude-sonnet-4-6", "gpt-5.6-terra"}


def build_prior_prompt(routes: set[str]) -> str:
    """Prior-only model-pool prompt for exactly the evaluated routes."""
    unknown = routes - MODEL_PRIORS.keys()
    if unknown:
        raise ValueError("missing prior description for: " + ", ".join(sorted(unknown)))
    ordered = sorted(routes, key=lambda m: (-MODEL_PRIORS[m][0], m))
    entries = [f"{i}. **{model}**: {MODEL_PRIORS[model][1]}"
               for i, model in enumerate(ordered, 1)]
    return PROMPT_HEADER + "\n".join(entries) + "\n\n" + PROMPT_FOOTER


ROUTER_SYSTEM_PROMPT = build_prior_prompt(MODELS)


def build_task_prompt(task_dir: Path, max_chars: int = 6000) -> str:
    cfg = tomllib.loads((task_dir / "task.toml").read_text())
    md = cfg["metadata"]
    instruction = (task_dir / "instruction.md").read_text()[:max_chars]
    return (
        "## Task to Route\n\n"
        f"**Repository**: {md.get('repo', 'unknown')}\n"
        f"**Difficulty**: {md.get('difficulty', 'unknown')}\n"
        "**Language**: python\n"
        "**Evaluation**: repository test suite (agentic coding task)\n\n"
        f"**Task**:\n{instruction}"
    )


def route(task_prompt: str, router_model: str,
          system_prompt: str | None = None,
          allowed: set[str] | None = None) -> dict:
    """One router call. Returns {"model", "reasoning", "router_cost_usd"}.

    system_prompt/allowed default to the zero-shot pool above; the profile
    router (P1) passes its own stats-bearing prompt and route set.
    """
    full_prompt = (system_prompt or ROUTER_SYSTEM_PROMPT) + "\n\n" + task_prompt
    allowed = allowed or MODELS
    cost = None
    if router_model.startswith("gpt"):
        codex_bin = os.environ.get("CODEX_BIN", "codex")
        res = subprocess.run(
            [codex_bin, "exec", "-m", router_model, "--skip-git-repo-check",
             "-s", "read-only", full_prompt],
            capture_output=True, text=True, timeout=180, cwd="/tmp",
        )
        if res.returncode != 0:
            detail = res.stderr.strip() or res.stdout.strip() or "no output"
            raise RuntimeError(f"codex CLI failed: {detail[:200]}")
        text = res.stdout
    else:
        res = subprocess.run(
            ["claude", "-p", full_prompt,
             "--model", router_model, "--output-format", "json"],
            capture_output=True, text=True, timeout=180,
        )
        if res.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {res.stderr[:200]}")
        payload = json.loads(res.stdout)
        text = payload.get("result", "")
        cost = payload.get("total_cost_usd")
    m = None
    for cand in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        if '"model"' in cand.group(0):
            m = cand
    if not m:
        raise ValueError(f"no JSON in router response: {text[:200]}")
    pick = json.loads(m.group(0))
    if pick.get("model") not in allowed:
        raise ValueError(f"router picked unknown model: {pick}")
    pick["router_cost_usd"] = cost
    return pick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path,
                    help="gated Harbor tasks directory or dataset JSONL")
    ap.add_argument("--router-model", default="claude-sonnet-4-6")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-strategy",
                    choices=["global-temporal", "repo-temporal"],
                    default="global-temporal")
    ap.add_argument("--route-split", choices=["test", "train"], default="test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.source.is_file():
        from scoreboard import load_matrix, split_tasks, split_tasks_by_repo

        rows = [json.loads(line) for line in args.source.read_text().splitlines()
                if line.strip()]
        _, tasks, routes, dates, repos = load_matrix(args.source)
        if args.split_strategy == "repo-temporal":
            train, test = split_tasks_by_repo(tasks, dates, repos)
        else:
            train, test = split_tasks(tasks, dates, args.test_frac)
        task_ids = train if args.route_split == "train" else test
        task_dirs = {r["task_id"]: Path(r["task_dir"]) for r in rows}
        allowed = set(routes)
        system_prompt = build_prior_prompt(allowed)
        out_path = args.out or Path("picks_baseline.jsonl")
    else:
        manifest = json.loads((args.source / "manifest.json").read_text())
        task_ids = sorted(t for t, e in manifest.items() if e["verdict"] == "pass")
        task_dirs = {tid: args.source / tid for tid in task_ids}
        allowed = MODELS
        system_prompt = ROUTER_SYSTEM_PROMPT
        out_path = args.out or (args.source / "router_picks.jsonl")

    picks = []
    with out_path.open("w") as out:
        for tid in task_ids:
            prompt = build_task_prompt(task_dirs[tid])
            try:
                pick = route(prompt, args.router_model,
                             system_prompt=system_prompt, allowed=allowed)
            except Exception as e:
                print(f"{tid}: ERROR {e}", file=sys.stderr)
                continue
            row = {"task_id": tid, "router_model": args.router_model, **pick}
            out.write(json.dumps(row) + "\n")
            picks.append(row)
            print(f"{tid}: {pick['model']:<18} {pick['reasoning'][:80]}")

    from collections import Counter
    dist = Counter(p["model"] for p in picks)
    print(f"\n{len(picks)}/{len(task_ids)} routed -> {out_path}")
    print("pick distribution:", dict(dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
