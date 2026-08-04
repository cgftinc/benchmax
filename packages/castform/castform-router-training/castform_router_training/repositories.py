"""Clone approved repositories and turn recent changes into routing tasks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from castform_router_training.project import Project


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def prepare_repositories(project: Project, output: Path) -> tuple[Path, ...]:
    """Clone or update each configured repository at its pinned revision."""

    repos_dir = output / "repositories"
    repos_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    for repository in project.repositories:
        destination = repos_dir / repository.name
        if destination.exists():
            if not (destination / ".git").is_dir():
                raise ValueError(f"existing path is not a Git repository: {destination}")
            _git("fetch", "--all", "--prune", cwd=destination)
        else:
            _git("clone", "--no-checkout", repository.url, str(destination))
        _git("checkout", "--detach", repository.revision, cwd=destination)
        prepared.append(destination)
    return tuple(prepared)


def mine_tasks(project: Project, output: Path, *, limit_per_repo: int = 100) -> Path:
    """Export recent non-merge commits as task seeds for rollout collection."""

    if limit_per_repo < 1:
        raise ValueError("limit_per_repo must be positive")
    repositories = prepare_repositories(project, output)
    tasks_path = output / "tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as stream:
        for spec, repository in zip(project.repositories, repositories, strict=True):
            records = _git(
                "log",
                "--no-merges",
                f"--max-count={limit_per_repo}",
                "--format=%H%x1f%P%x1f%s%x1f%b%x1e",
                cwd=repository,
            )
            for record in records.strip("\n\x1e").split("\x1e"):
                if not record.strip():
                    continue
                revision, parents, subject, body = record.strip("\n").split("\x1f", 3)
                parent = parents.split()[0] if parents else None
                task = {
                    "task_id": f"{spec.name}:{revision}",
                    "repository": spec.name,
                    "revision": revision,
                    "base_revision": parent,
                    "task": "\n\n".join(part for part in (subject, body.strip()) if part),
                }
                stream.write(json.dumps(task, sort_keys=True) + "\n")
    return tasks_path
