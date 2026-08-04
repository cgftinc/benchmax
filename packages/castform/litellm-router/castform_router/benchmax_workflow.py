"""Orchestrate Benchmax's existing examples/model_router workflow.

This module intentionally does not reimplement mining, conversion, gating,
auditing, dataset construction, routing baselines, or scoring. It builds and
optionally executes commands against the authoritative Benchmax scripts.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCHMAX_REPOSITORY = "https://github.com/castform-ai/benchmax.git"
BENCHMAX_WORKFLOW_REF = "model-router"
STAGES = (
    "setup",
    "mine",
    "convert",
    "gate",
    "collect",
    "dataset",
    "router",
    "scoreboard",
)
REQUIRED_SCRIPTS = (
    "convert_to_harbor.py",
    "gate_tasks.py",
    "audit_trajectories.py",
    "build_dataset.py",
    "baseline_router.py",
    "profile_router.py",
    "knn_router.py",
    "scoreboard.py",
)


@dataclass(slots=True)
class WorkflowStep:
    stage: str
    name: str
    description: str
    command: list[str]
    cwd: str
    spends_model_credits: bool = False
    env_refs: dict[str, str] = field(default_factory=dict)
    skip_if_exists: str | None = None


def build_benchmax_plan(
    workspace: Path,
    *,
    workflow_dir: Path | None = None,
    agent_network: str = "allowlist",
    gate_k: int = 3,
    concurrency: int = 4,
    router_rung: str = "knn",
    router_model: str = "claude-sonnet-4-6",
    codeprobe_llm: bool = False,
) -> dict[str, Any]:
    """Build commands for the authoritative Benchmax model-router workflow."""

    workspace = workspace.resolve()
    if agent_network not in {"allowlist", "public"}:
        raise ValueError("agent_network must be allowlist or public")
    if not 1 <= gate_k <= 10:
        raise ValueError("gate_k must be between 1 and 10")
    if not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be between 1 and 64")
    if router_rung not in {"knn", "profile", "baseline"}:
        raise ValueError("router_rung must be knn, profile, or baseline")

    manifest = _read_object(workspace / "manifest.json")
    spec = _read_object(workspace / "project.spec.json")
    repositories = manifest.get("repositories")
    routes = manifest.get("candidate_routes")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("workspace manifest has no repositories")
    if not isinstance(routes, list) or not routes:
        raise ValueError("workspace manifest has no candidate routes")
    if router_rung == "baseline" and len(repositories) != 1:
        raise ValueError(
            "the existing baseline_router.py accepts one task directory; "
            "use knn or profile for a multi-repository project"
        )

    root = workspace / "benchmax" / "model_router"
    checkout_root = root / "upstream"
    detected_workflow = (
        workflow_dir.resolve()
        if workflow_dir is not None
        else _find_local_workflow(workspace)
    )
    needs_checkout = detected_workflow is None
    authoritative_dir = (
        checkout_root / "examples" / "model_router"
        if needs_checkout
        else detected_workflow
    )
    if not needs_checkout:
        _validate_workflow_dir(authoritative_dir)

    repos_root = root / "repos"
    tasks_root = root / "harbor_tasks"
    gate_runs_root = root / "gate_runs"
    runs_root = root / "harbor_runs"
    outputs_root = root / "router_outputs"
    dataset = root / "dataset.jsonl"
    repetitions = int(spec.get("benchmark", {}).get("repetitions", 1))
    task_limit = int(spec.get("pull_requests", {}).get("limit_per_repo", 20))
    test_fraction = float(spec.get("pull_requests", {}).get("eval_ratio", 0.2))
    uv = "uv"
    # CodeProbe 0.13 pins rich<14 while Harbor 0.18 pins rich>=14.1.
    # Keep them in separate workspace-local environments instead of asking
    # one resolver environment to satisfy an impossible dependency set.
    codeprobe_venv = root / ".venv-codeprobe"
    harbor_venv = authoritative_dir / ".venv"
    uv_python = [str(harbor_venv / "bin" / "python")]
    uv_codeprobe = [str(codeprobe_venv / "bin" / "codeprobe")]
    uv_harbor = [str(harbor_venv / "bin" / "harbor")]
    steps: list[WorkflowStep] = []
    github_check_env = next(
        (
            refs
            for repository in repositories
            if (refs := _github_env_refs(repository))
        ),
        {},
    )

    if needs_checkout:
        steps.append(
            WorkflowStep(
                stage="setup",
                name="checkout-benchmax-model-router",
                description=(
                    "Check out the authoritative Benchmax model-router workflow."
                ),
                command=[
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    BENCHMAX_WORKFLOW_REF,
                    BENCHMAX_REPOSITORY,
                    str(checkout_root),
                ],
                cwd=str(root),
                skip_if_exists=str(authoritative_dir / "README.md"),
            )
        )
    steps.extend(
        [
            WorkflowStep(
                stage="setup",
                name="prepare-workspace-directories",
                description="Create the standard Benchmax artifact directories.",
                command=[
                    "mkdir",
                    "-p",
                    str(repos_root),
                    str(tasks_root),
                    str(gate_runs_root),
                    str(runs_root),
                    str(outputs_root),
                ],
                cwd=str(root),
            ),
            WorkflowStep(
                stage="setup",
                name="create-codeprobe-environment",
                description="Create CodeProbe's isolated dependency environment.",
                command=[
                    uv,
                    "venv",
                    str(codeprobe_venv),
                    "--python",
                    "3.12",
                ],
                cwd=str(root),
                skip_if_exists=str(codeprobe_venv / "bin" / "python"),
            ),
            WorkflowStep(
                stage="setup",
                name="install-codeprobe",
                description="Install CodeProbe without Harbor's conflicting Rich pin.",
                command=[
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(codeprobe_venv / "bin" / "python"),
                    "codeprobe>=0.13,<0.14",
                ],
                cwd=str(root),
                skip_if_exists=str(codeprobe_venv / "bin" / "codeprobe"),
            ),
            WorkflowStep(
                stage="setup",
                name="create-harbor-environment",
                description="Create Harbor's isolated dependency environment.",
                command=[
                    uv,
                    "venv",
                    str(harbor_venv),
                    "--python",
                    "3.12",
                ],
                cwd=str(root),
                skip_if_exists=str(harbor_venv / "bin" / "python"),
            ),
            WorkflowStep(
                stage="setup",
                name="install-harbor",
                description="Install Harbor separately from CodeProbe.",
                command=[
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(harbor_venv / "bin" / "python"),
                    "harbor>=0.18.0,<0.19",
                ],
                cwd=str(root),
                skip_if_exists=str(harbor_venv / "bin" / "harbor"),
            ),
            WorkflowStep(
                stage="setup",
                name="check-github-auth",
                description=(
                    "Verify GitHub CLI auth used by CodeProbe for PR narratives."
                ),
                command=["gh", "auth", "status"],
                cwd=str(root),
                env_refs=github_check_env,
            ),
        ]
    )

    repository_paths: list[tuple[dict[str, Any], Path, Path]] = []
    names: set[str] = set()
    for repository in repositories:
        name = str(repository["name"])
        if name in names:
            raise ValueError(
                "repository names must be unique for the existing dataset builder: "
                f"{name}"
            )
        names.add(name)
        repo_dir = repos_root / name
        tasks_dir = tasks_root / name
        repository_paths.append((repository, repo_dir, tasks_dir))
        env_refs = _github_env_refs(repository)
        clone_command = (
            ["git", "clone", str(repository["html_url"]), str(repo_dir)]
            if repository.get("auth", {}).get("strategy", "public") == "public"
            else ["gh", "repo", "clone", str(repository["full_name"]), str(repo_dir)]
        )
        steps.extend(
            [
                WorkflowStep(
                    stage="mine",
                    name=f"clone-{name}",
                    description=(
                        f"Create the full-history clone required to mine {name}."
                    ),
                    command=clone_command,
                    cwd=str(repos_root),
                    env_refs=env_refs,
                    skip_if_exists=str(repo_dir / ".git"),
                ),
                WorkflowStep(
                    stage="mine",
                    name=f"checkout-{name}-revision",
                    description=(
                        f"Select {repository.get('default_branch') or 'main'} "
                        f"as the mining revision for {name}."
                    ),
                    command=[
                        "git",
                        "-C",
                        str(repo_dir),
                        "checkout",
                        str(repository.get("default_branch") or "main"),
                    ],
                    cwd=str(repos_root),
                ),
                WorkflowStep(
                    stage="mine",
                    name=f"mine-{name}",
                    description=(
                        f"Mine and quality-rank historical coding tasks from {name}."
                    ),
                    command=[
                        *uv_codeprobe,
                        "mine",
                        ".",
                        "--goal",
                        "quality",
                        "--count",
                        str(task_limit),
                        "--no-interactive",
                        *([] if codeprobe_llm else ["--no-llm"]),
                    ],
                    cwd=str(repo_dir),
                    env_refs=env_refs,
                    spends_model_credits=codeprobe_llm,
                ),
                WorkflowStep(
                    stage="convert",
                    name=f"convert-{name}",
                    description=(
                        "Create Harbor tasks with PR-era test overlays and "
                        "leak-guarded base checkouts."
                    ),
                    command=[
                        *uv_python,
                        str(authoritative_dir / "convert_to_harbor.py"),
                        str(repo_dir),
                        str(tasks_dir),
                        "--repo-url",
                        str(repository["html_url"]),
                        "--agent-network",
                        agent_network,
                    ],
                    cwd=str(root),
                    env_refs=env_refs,
                ),
                WorkflowStep(
                    stage="gate",
                    name=f"gate-{name}",
                    description=(
                        "Keep tasks only when every oracle run passes and every "
                        "nop run fails."
                    ),
                    command=[
                        *uv_python,
                        str(authoritative_dir / "gate_tasks.py"),
                        str(tasks_dir),
                        "-k",
                        str(gate_k),
                        "--jobs-dir",
                        str(gate_runs_root / name),
                        "-n",
                        str(concurrency),
                    ],
                    cwd=str(root),
                ),
            ]
        )

    collection_dirs: list[Path] = []
    for repository, _repo_dir, tasks_dir in repository_paths:
        for route in routes:
            route_slug = _safe_name(str(route["route_id"]))
            jobs_dir = runs_root / str(repository["name"]) / route_slug
            collection_dirs.append(jobs_dir)
            command = [
                *uv_harbor,
                "run",
                "-p",
                str(tasks_dir),
                "-a",
                str(route.get("harbor_agent") or route["harness"]),
                "-m",
                str(route.get("harbor_model") or route["model"]),
                "-k",
                str(repetitions),
                "-n",
                str(concurrency),
                "--jobs-dir",
                str(jobs_dir),
            ]
            command.extend(_harbor_auth_arguments(route))
            steps.append(
                WorkflowStep(
                    stage="collect",
                    name=(
                        f"collect-{repository['name']}-{route_slug}"
                    ),
                    description=(
                        f"Collect Harbor outcomes for {route['route_id']}."
                    ),
                    command=command,
                    cwd=str(root),
                    spends_model_credits=True,
                )
            )

    dataset_command = [
        *uv_python,
        str(authoritative_dir / "build_dataset.py"),
        *(str(path) for path in collection_dirs),
    ]
    for _repository, _repo_dir, tasks_dir in repository_paths:
        dataset_command.extend(["--tasks-dir", str(tasks_dir)])
    dataset_command.extend(
        [
            "--out",
            str(dataset),
            "--repo-root",
            str(repos_root),
        ]
    )
    steps.append(
        WorkflowStep(
            stage="dataset",
            name="build-audited-dataset",
            description=(
                "Audit trajectories, drop tainted or ungated trials, and build "
                "the temporal dataset."
            ),
            command=dataset_command,
            cwd=str(authoritative_dir),
        )
    )

    picks = outputs_root / f"picks_{router_rung}.jsonl"
    if router_rung == "knn":
        router_command = [
            *uv_python,
            str(authoritative_dir / "knn_router.py"),
            str(dataset),
            "-k",
            "3",
            "--test-frac",
            str(test_fraction),
            "--out",
            str(picks),
        ]
        router_spend = False
    elif router_rung == "profile":
        router_command = [
            *uv_python,
            str(authoritative_dir / "profile_router.py"),
            str(dataset),
            "--router-model",
            router_model,
            "--test-frac",
            str(test_fraction),
            "--out",
            str(picks),
        ]
        router_spend = True
    else:
        router_command = [
            *uv_python,
            str(authoritative_dir / "baseline_router.py"),
            str(repository_paths[0][2]),
            "--router-model",
            router_model,
            "--out",
            str(picks),
        ]
        router_spend = True
    steps.append(
        WorkflowStep(
            stage="router",
            name=f"run-{router_rung}-router-rung",
            description=(
                f"Run Benchmax's existing {router_rung} router baseline."
            ),
            command=router_command,
            cwd=str(authoritative_dir),
            spends_model_credits=router_spend,
        )
    )
    steps.append(
        WorkflowStep(
            stage="scoreboard",
            name="score-router",
            description=(
                "Compare the router with always-route, random, and oracle "
                "policies on the temporal test split."
            ),
            command=[
                *uv_python,
                str(authoritative_dir / "scoreboard.py"),
                str(dataset),
                "--split",
                "test",
                "--test-frac",
                str(test_fraction),
                "--picks",
                str(picks),
            ],
            cwd=str(authoritative_dir),
        )
    )

    return {
        "schema_version": "1",
        "implementation": "benchmax/examples/model_router",
        "source": {
            "repository": BENCHMAX_REPOSITORY,
            "ref": BENCHMAX_WORKFLOW_REF,
            "workflow_dir": str(authoritative_dir),
            "uses_local_checkout": not needs_checkout,
        },
        "workspace": str(workspace),
        "stages": list(STAGES),
        "router_rung": router_rung,
        "router_model": router_model,
        "codeprobe_llm": codeprobe_llm,
        "agent_network": agent_network,
        "gate_k": gate_k,
        "steps": [asdict(step) for step in steps],
    }


def write_benchmax_plan(workspace: Path, plan: dict[str, Any]) -> Path:
    path = workspace / "benchmax" / "model_router" / "workflow-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = workspace / "manifest.json"
    manifest = _read_object(manifest_path)
    manifest["status"] = "ready_for_benchmax_mining"
    manifest["benchmax_workflow"] = {
        "implementation": plan["implementation"],
        "plan": str(path.relative_to(workspace)),
        "stages": plan["stages"],
        "router_rung": plan["router_rung"],
        "router_model": plan["router_model"],
        "codeprobe_llm": plan.get("codeprobe_llm", False),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def execute_benchmax_plan(
    workspace: Path,
    plan: dict[str, Any],
    *,
    from_stage: str = "setup",
    through_stage: str = "scoreboard",
) -> dict[str, Any]:
    """Execute a selected contiguous range of the generated workflow."""

    if from_stage not in STAGES or through_stage not in STAGES:
        raise ValueError(f"stages must be one of: {', '.join(STAGES)}")
    start = STAGES.index(from_stage)
    end = STAGES.index(through_stage)
    if start > end:
        raise ValueError("from_stage must not come after through_stage")

    workspace = workspace.resolve()
    trace_path = (
        workspace / "benchmax" / "model_router" / "workflow-trace.jsonl"
    )
    selected = [
        step
        for step in plan["steps"]
        if start <= STAGES.index(step["stage"]) <= end
    ]
    completed = 0
    for step in selected:
        cwd = Path(step["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        skipped = bool(
            step.get("skip_if_exists")
            and Path(step["skip_if_exists"]).exists()
        )
        started_at = datetime.now(UTC)
        _append_trace(
            trace_path,
            {
                "event": "step_started",
                "timestamp": started_at.isoformat(),
                "stage": step["stage"],
                "name": step["name"],
                "command": step["command"],
                "cwd": step["cwd"],
                "skipped": skipped,
            },
        )
        if skipped:
            completed += 1
            continue
        env = os.environ.copy()
        for target, source in step.get("env_refs", {}).items():
            value = os.getenv(source)
            if not value:
                raise ValueError(
                    f"{step['name']} requires environment variable {source}"
                )
            env[target] = value
        result = subprocess.run(
            step["command"],
            cwd=step["cwd"],
            env=env,
            check=False,
        )
        _append_trace(
            trace_path,
            {
                "event": "step_finished",
                "timestamp": datetime.now(UTC).isoformat(),
                "stage": step["stage"],
                "name": step["name"],
                "returncode": result.returncode,
            },
        )
        if result.returncode:
            raise ValueError(
                f"Benchmax step failed ({step['name']}): "
                f"exit code {result.returncode}"
            )
        completed += 1

    manifest_path = workspace / "manifest.json"
    manifest = _read_object(manifest_path)
    manifest["status"] = f"benchmax_{through_stage}_complete"
    manifest.setdefault("benchmax_workflow", {})["last_completed_stage"] = (
        through_stage
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": manifest["status"],
        "completed_steps": completed,
        "trace": str(trace_path),
    }


def select_plan_steps(
    plan: dict[str, Any],
    *,
    from_stage: str,
    through_stage: str,
) -> list[dict[str, Any]]:
    if from_stage not in STAGES or through_stage not in STAGES:
        raise ValueError(f"stages must be one of: {', '.join(STAGES)}")
    start = STAGES.index(from_stage)
    end = STAGES.index(through_stage)
    if start > end:
        raise ValueError("from_stage must not come after through_stage")
    return [
        step
        for step in plan["steps"]
        if start <= STAGES.index(step["stage"]) <= end
    ]


def _find_local_workflow(workspace: Path) -> Path | None:
    for base in (Path.cwd(), workspace, *workspace.parents):
        candidate = base / "examples" / "model_router"
        if all((candidate / script).is_file() for script in REQUIRED_SCRIPTS):
            return candidate.resolve()
    return None


def _validate_workflow_dir(path: Path) -> None:
    missing = [
        script for script in REQUIRED_SCRIPTS if not (path / script).is_file()
    ]
    if missing:
        raise ValueError(
            f"Benchmax workflow directory is missing: {', '.join(missing)}"
        )


def _github_env_refs(repository: dict[str, Any]) -> dict[str, str]:
    auth = repository.get("auth", {})
    strategy = auth.get("strategy", "public")
    if strategy == "public":
        return {}
    source = (
        auth.get("token_env")
        if strategy == "token_env"
        else auth.get("installation_token_env")
    )
    if not isinstance(source, str):
        raise ValueError(
            f"{repository['full_name']} needs installation_token_env for "
            "Benchmax mining"
        )
    return {"GH_TOKEN": source}


def _harbor_auth_arguments(route: dict[str, Any]) -> list[str]:
    agent = str(route.get("harbor_agent") or route.get("harness"))
    provider = str(route.get("provider"))
    if agent == "claude-code" and provider == "anthropic":
        return ["--ae", "CLAUDE_FORCE_OAUTH=1"]
    if agent == "codex" and provider == "openai":
        return ["--ae", "CODEX_FORCE_AUTH_JSON=1"]
    return []


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in value.lower()
    ).strip("-")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required workspace file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _append_trace(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
