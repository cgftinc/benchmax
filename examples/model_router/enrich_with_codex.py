#!/usr/bin/env python3
"""Enrich Codeprobe SDLC task instructions through the local Codex CLI.

Mine with ``codeprobe mine --no-llm``, then run this script before converting
the tasks to Harbor. Preview mode is the default: it writes one JSON artifact
per task without changing the mined task. ``--in-place`` atomically updates
``metadata.json``, ``instruction.md``, and an existing ``instruction_mcp.md``.

Example:
  python enrich_with_codex.py numpy/.codeprobe/tasks \
      --task-id 5e403264 --out-dir codex_enrichment
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.6-sol"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "heading": {"type": "string"},
        "team_context": {"type": "string"},
        "access_scope": {"type": "string"},
        "problem": {"type": "string"},
        "reproduction": {"type": "string"},
        "requirements": {"type": "string"},
        "success_criteria": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
    },
    "required": [
        "heading",
        "team_context",
        "access_scope",
        "problem",
        "reproduction",
        "requirements",
        "success_criteria",
        "difficulty",
    ],
    "additionalProperties": False,
}


def batch_schema(task_ids: list[str]) -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "enum": task_ids},
            **OUTPUT_SCHEMA["properties"],
        },
        "required": ["task_id", *OUTPUT_SCHEMA["required"]],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": item,
                "minItems": len(task_ids),
                "maxItems": len(task_ids),
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }

PROMPT_TEMPLATE = """\
You write coding-agent benchmark tasks from real pull-request evidence.

Outcome:
- Produce a self-contained problem statement and acceptance contract.
- Tell the agent what observable behavior is required without revealing the
  implementation, patch, newly introduced symbols, or exact code edits.
- Preserve facts from the supplied evidence. Do not invent reproduction
  output, APIs, ownership boundaries, or success criteria. Use an empty string
  when the evidence does not support a field.

Critical anti-leak transformation:
- A PR narrative often explains the author's implementation. Treat those
  details only as evidence of the underlying bug or required behavior.
- Do not repeat internal algorithms, helper names, lookup strategies, exact
  conditions, dtype-slot/view mechanics, or instructions about how to edit a
  file. Translate them into externally observable behavior.
- The changed-file list may define access scope and broad team context, but
  it is not evidence that every listed file needs a particular edit or test.
- Requirements and success criteria must be supported by the narrative or
  linked issue. Do not invent a regression-test scenario merely because a
  test file appears in changed_files.

Remove PR-template noise, reviewer discussion, AI-disclosure notes, backport
bookkeeping, and solution-oriented prose.

Return exactly the requested JSON schema. Markdown is allowed inside string
fields. Requirements and success_criteria must be Markdown bullet lists stored
as strings. Keep the problem to 3-6 sentences and the heading concise.

Repository: {repo}
Language: {language}
Raw PR title: {title}
Raw PR narrative:
{description}

Linked issue title:
{issue_title}

Linked issue body:
{issue_body}

Changed files:
{changed_files}
"""

BATCH_HEADER = """\
Process every task below independently. Return exactly one entry per task_id;
do not combine evidence or requirements across tasks.
"""


def task_evidence(task_dir: Path) -> tuple[dict[str, Any], str]:
    metadata_path = task_dir / "metadata.json"
    ground_truth_path = task_dir / "tests" / "ground_truth.json"
    metadata = json.loads(metadata_path.read_text())
    task_meta = metadata["metadata"]
    description = str(task_meta.get("description", "")).strip()
    title = description.splitlines()[0] if description else task_meta.get("name", "")

    # Once Codeprobe has enriched a task, issue_title/body contain generated
    # prose rather than source evidence. A forced comparison run must not feed
    # that prose back into Codex and mistake it for a linked issue.
    source = str(task_meta.get("enrichment_source", ""))
    already_generated = "llm" in source.split("+") or "codex" in source.split("+")
    issue_title = "" if already_generated else str(task_meta.get("issue_title", ""))
    issue_body = "" if already_generated else str(task_meta.get("issue_body", ""))

    changed_files: list[str] = []
    if ground_truth_path.exists():
        ground_truth = json.loads(ground_truth_path.read_text())
        changed_files = [str(path) for path in ground_truth.get("changed_files", [])]

    prompt = PROMPT_TEMPLATE.format(
        repo=metadata.get("repo", task_dir.parent.parent.name),
        language=task_meta.get("language", "unknown"),
        title=title,
        description=description,
        issue_title=issue_title or "(none)",
        issue_body=issue_body or "(none)",
        changed_files="\n".join(f"- {path}" for path in changed_files) or "(none)",
    )
    return metadata, prompt


def validate_enrichment(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Codex output is not a JSON object")
    expected = set(OUTPUT_SCHEMA["required"])
    if set(value) != expected:
        raise ValueError(
            f"Codex output keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    if any(not isinstance(value[key], str) for key in expected):
        raise ValueError("every Codex output field must be a string")
    if not value["heading"].strip() or not value["problem"].strip():
        raise ValueError("heading and problem must be non-empty")
    if value["difficulty"] not in {"easy", "medium", "hard"}:
        raise ValueError(f"invalid difficulty: {value['difficulty']!r}")
    return {key: value[key].strip() for key in OUTPUT_SCHEMA["required"]}


def validate_batch(value: Any, task_ids: list[str]) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"tasks"}:
        raise ValueError("batched Codex output must contain only a tasks array")
    if not isinstance(value["tasks"], list):
        raise ValueError("batched Codex tasks field is not an array")
    result: dict[str, dict[str, str]] = {}
    for item in value["tasks"]:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise ValueError("each batched result needs a string task_id")
        task_id = item["task_id"]
        if task_id in result:
            raise ValueError(f"duplicate batched task_id: {task_id}")
        result[task_id] = validate_enrichment(
            {key: value for key, value in item.items() if key != "task_id"}
        )
    if set(result) != set(task_ids):
        raise ValueError(
            f"batched task IDs differ: missing={sorted(set(task_ids) - set(result))}, "
            f"extra={sorted(set(result) - set(task_ids))}"
        )
    return result


def compose_issue_body(enrichment: dict[str, str]) -> str:
    sections = [enrichment["problem"]]
    for heading, key in (
        ("Context", "team_context"),
        ("Access Scope", "access_scope"),
        ("Steps to Reproduce", "reproduction"),
        ("Requirements", "requirements"),
        ("Success Criteria", "success_criteria"),
    ):
        if enrichment[key]:
            sections.append(f"## {heading}\n\n{enrichment[key]}")
    return "\n\n".join(sections)


def render_instruction(metadata: dict[str, Any], enrichment: dict[str, str],
                       repo_root: Path) -> str:
    return (
        f"# {enrichment['heading']}\n\n"
        f"**Repository:** {metadata.get('repo', repo_root.name)}\n"
        f"**Language:** {metadata['metadata'].get('language', 'unknown')}\n\n"
        "## Problem\n\n"
        f"{compose_issue_body(enrichment)}\n\n"
        "## Task Contract\n\n"
        f"- `TASK_REPO_ROOT={repo_root.resolve()}`\n\n"
        "## Task\n\n"
        "Implement the fix or feature described above. "
        "The test script will verify correctness.\n"
    )


def find_codex_binary(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("CODEX_BIN") or shutil.which("codex")
    if not candidate:
        raise RuntimeError("codex CLI not found; pass --codex-bin or set CODEX_BIN")
    return candidate


def run_codex_json(prompt: str, schema: dict[str, Any], *, codex_bin: str,
                   model: str, reasoning_effort: str,
                   timeout: int) -> tuple[Any, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="codeprobe-codex-enrich-") as raw_tmp:
        temp_dir = Path(raw_tmp)
        schema_path = temp_dir / "schema.json"
        output_path = temp_dir / "output.json"
        schema_path.write_text(json.dumps(schema, indent=2) + "\n")
        cmd = [
            codex_bin,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--json",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        started = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=temp_dir,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"codex exec timed out after {timeout}s") from exc
        duration = time.monotonic() - started
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RuntimeError(f"codex exec failed ({result.returncode}): {detail[-1000:]}")
        if not output_path.exists():
            raise RuntimeError("codex exec succeeded without --output-last-message output")

        output = json.loads(output_path.read_text())
        events: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        completed = next(
            (event for event in reversed(events) if event.get("type") == "turn.completed"),
            {},
        )
        run_meta = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "duration_seconds": round(duration, 3),
            "usage": completed.get("usage", {}),
            "stderr_tail": result.stderr[-2000:],
        }
        return output, run_meta


def run_codex_batch(entries: list[tuple[str, str]], *, codex_bin: str,
                    model: str, reasoning_effort: str,
                    timeout: int) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    task_ids = [task_id for task_id, _ in entries]
    sections = [BATCH_HEADER]
    for task_id, prompt in entries:
        sections.append(f"\n===== TASK {task_id} =====\n{prompt}")
    output, run_meta = run_codex_json(
        "\n".join(sections),
        batch_schema(task_ids),
        codex_bin=codex_bin,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    run_meta["batch_task_ids"] = task_ids
    return validate_batch(output, task_ids), run_meta


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def apply_enrichment(task_dir: Path, metadata: dict[str, Any],
                     enrichment: dict[str, str], instruction: str) -> None:
    task_meta = metadata["metadata"]
    source_parts = [part for part in str(task_meta.get("enrichment_source", "")).split("+") if part]
    if "codex" not in source_parts:
        source_parts.append("codex")
    task_meta.update(
        issue_title=enrichment["heading"],
        issue_body=compose_issue_body(enrichment),
        difficulty=enrichment["difficulty"],
        enrichment_source="+".join(source_parts) or "codex",
    )

    mcp_path = task_dir / "instruction_mcp.md"
    mcp_content: str | None = None
    if mcp_path.exists():
        current = mcp_path.read_text()
        marker = "\n# MCP Capabilities\n"
        if marker not in current:
            raise ValueError(f"cannot preserve MCP suffix in {mcp_path}")
        suffix = current[current.index(marker):]
        mcp_content = instruction.rstrip() + suffix

    atomic_write(task_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    atomic_write(task_dir / "instruction.md", instruction)
    if mcp_content is not None:
        atomic_write(mcp_path, mcp_content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("--task-id", action="append", default=[],
                        help="task ID to enrich; repeatable (required unless --all)")
    parser.add_argument("--all", action="store_true",
                        help="enrich every not-yet-generated task")
    parser.add_argument("--out-dir", type=Path, default=Path("codex_enrichment"))
    parser.add_argument("--in-place", action="store_true",
                        help="atomically update mined task artifacts after validation")
    parser.add_argument("--force", action="store_true",
                        help="allow an already LLM/Codex-enriched task")
    parser.add_argument("--resume", action="store_true",
                        help="skip explicitly requested tasks already enriched by LLM/Codex")
    parser.add_argument("--model", default=os.environ.get("CODEPROBE_CODEX_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--codex-bin", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="tasks per codex exec call (default: 5)")
    parser.add_argument("--no-single-fallback", action="store_true",
                        help="stop if a multi-task batch fails instead of retrying singly")
    args = parser.parse_args()

    if args.all == bool(args.task_id):
        parser.error("choose exactly one of --all or one or more --task-id")
    task_dirs = sorted(
        path for path in args.tasks_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").exists()
    )
    if args.task_id:
        requested = set(args.task_id)
        task_dirs = [path for path in task_dirs if path.name in requested]
        missing = requested - {path.name for path in task_dirs}
        if missing:
            parser.error("unknown task IDs: " + ", ".join(sorted(missing)))

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    codex_bin = find_codex_binary(args.codex_bin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = args.tasks_dir.parent.parent
    pending: list[tuple[Path, dict[str, Any], str, str]] = []
    for task_dir in task_dirs:
        metadata, prompt = task_evidence(task_dir)
        source = str(metadata["metadata"].get("enrichment_source", ""))
        generated = {"llm", "codex"} & set(source.split("+"))
        if not args.force and generated:
            if args.all or args.resume:
                print(f"skipping {task_dir.name}: already enriched ({source})")
                continue
            raise RuntimeError(
                f"{task_dir.name} is already enriched ({source}); pass --force to compare"
            )
        pending.append((task_dir, metadata, prompt, source))

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        entries = [(task_dir.name, prompt) for task_dir, _, prompt, _ in batch]
        ids = [task_id for task_id, _ in entries]
        print(
            f"enriching batch {offset // args.batch_size + 1}: {', '.join(ids)} "
            f"with {args.model} ({args.reasoning_effort})"
        )
        try:
            enrichments, run_meta = run_codex_batch(
                entries,
                codex_bin=codex_bin,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
            results = [(task_id, enrichments[task_id], run_meta) for task_id in ids]
        except Exception as exc:
            if len(batch) == 1 or args.no_single_fallback:
                raise
            print(f"batch failed ({exc}); retrying {len(batch)} tasks singly")
            results = []
            for task_id, prompt in entries:
                single, single_meta = run_codex_batch(
                    [(task_id, prompt)],
                    codex_bin=codex_bin,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout,
                )
                results.append((task_id, single[task_id], single_meta))

        by_id = {task_dir.name: (task_dir, metadata, prompt, source)
                 for task_dir, metadata, prompt, source in batch}
        for task_id, enrichment, run_meta in results:
            task_dir, metadata, prompt, source = by_id[task_id]
            instruction = render_instruction(metadata, enrichment, repo_root)
            artifact = {
                "task_id": task_id,
                "source_enrichment": source,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "run": run_meta,
                "enrichment": enrichment,
                "instruction": instruction,
            }
            artifact_path = args.out_dir / f"{task_id}.json"
            atomic_write(
                artifact_path,
                json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            )
            if args.in_place:
                apply_enrichment(task_dir, metadata, enrichment, instruction)
            print(
                f"artifact -> {artifact_path}"
                + ("; applied" if args.in_place else "; preview only")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
