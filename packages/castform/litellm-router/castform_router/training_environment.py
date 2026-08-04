"""Generate a local Castform + Benchmax router-training workspace."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class TrainingRoute:
    route_id: str
    label: str
    harness: str
    model: str
    provider: str
    family: str
    harbor_agent: str
    harbor_model: str


TRAINING_ROUTE_CATALOG = (
    TrainingRoute(
        route_id="claude-code/opus@anthropic",
        label="Claude Code · Opus",
        harness="claude-code",
        model="opus",
        provider="anthropic",
        family="Claude",
        harbor_agent="claude-code",
        harbor_model="claude-opus-5",
    ),
    TrainingRoute(
        route_id="claude-code/sonnet@anthropic",
        label="Claude Code · Sonnet",
        harness="claude-code",
        model="sonnet",
        provider="anthropic",
        family="Claude",
        harbor_agent="claude-code",
        harbor_model="claude-sonnet-4-6",
    ),
    TrainingRoute(
        route_id="claude-code/haiku@anthropic",
        label="Claude Code · Haiku",
        harness="claude-code",
        model="haiku",
        provider="anthropic",
        family="Claude",
        harbor_agent="claude-code",
        harbor_model="claude-haiku-4-5",
    ),
    TrainingRoute(
        route_id="codex/5.6-fast@openai",
        label="Codex · 5.6 Fast",
        harness="codex",
        model="5.6-fast",
        provider="openai",
        family="OpenAI 5.6",
        harbor_agent="codex",
        harbor_model="gpt-5.6-luna",
    ),
    TrainingRoute(
        route_id="codex/5.6-balanced@openai",
        label="Codex · 5.6 Balanced",
        harness="codex",
        model="5.6-balanced",
        provider="openai",
        family="OpenAI 5.6",
        harbor_agent="codex",
        harbor_model="gpt-5.6-terra",
    ),
    TrainingRoute(
        route_id="codex/5.6-deep@openai",
        label="Codex · 5.6 Deep",
        harness="codex",
        model="5.6-deep",
        provider="openai",
        family="OpenAI 5.6",
        harbor_agent="codex",
        harbor_model="gpt-5.6-sol",
    ),
    TrainingRoute(
        route_id="claude-code/glm-5.1@zai",
        label="Claude Code · GLM 5.1",
        harness="claude-code",
        model="glm-5.1",
        provider="zai",
        family="GLM",
        harbor_agent="claude-code",
        harbor_model="glm-5.1",
    ),
)

_GITHUB_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")


def route_catalog_json() -> list[dict[str, Any]]:
    return [asdict(route) for route in TRAINING_ROUTE_CATALOG]


def parse_github_repo(value: object) -> dict[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("repository must be a GitHub URL or owner/repo")
    normalized = value.strip()
    if "://" in normalized:
        parsed = urlparse(normalized)
        if parsed.scheme != "https" or parsed.hostname not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError("only https://github.com repository URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in normalized.split("/") if part]
    if len(parts) != 2:
        raise ValueError("repository must have the shape owner/repo")
    owner, name = parts
    name = name.removesuffix(".git")
    if not _GITHUB_PART.fullmatch(owner) or not _GITHUB_PART.fullmatch(name):
        raise ValueError("repository owner or name contains unsupported characters")
    return {
        "owner": owner,
        "name": name,
        "full_name": f"{owner}/{name}",
        "html_url": f"https://github.com/{owner}/{name}",
    }


def build_training_workspace(
    root: Path,
    *,
    repositories: list[dict[str, Any]],
    selected_route_ids: list[str],
    tasks_per_repo: int,
    repetitions: int,
    average_run_cost_usd: float,
    privacy_mode: str,
) -> dict[str, Any]:
    if not repositories:
        raise ValueError("select at least one repository")
    if not 1 <= tasks_per_repo <= 20:
        raise ValueError("tasks_per_repo must be between 1 and 20")
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be between 1 and 10")
    if not 0 <= average_run_cost_usd <= 100:
        raise ValueError("average_run_cost_usd must be between 0 and 100")
    if privacy_mode not in {"customer_runner", "castform_hosted"}:
        raise ValueError("unsupported privacy_mode")

    catalog = {route.route_id: route for route in TRAINING_ROUTE_CATALOG}
    if len(set(selected_route_ids)) != len(selected_route_ids):
        raise ValueError("selected routes must not contain duplicates")
    try:
        routes = [catalog[route_id] for route_id in selected_route_ids]
    except KeyError as error:
        raise ValueError(f"unknown route: {error.args[0]}") from error
    if len(routes) < 2:
        raise ValueError("select at least two routes so the router has a choice")

    normalized_repositories: list[dict[str, Any]] = []
    seen_repositories: set[str] = set()
    for repository in repositories:
        parsed = parse_github_repo(repository.get("html_url") or repository.get("full_name"))
        if parsed["full_name"] in seen_repositories:
            continue
        seen_repositories.add(parsed["full_name"])
        normalized_repositories.append(
            {
                **parsed,
                "default_branch": repository.get("default_branch") or "main",
                "visibility": repository.get("visibility") or "unknown",
                "verification": repository.get("verification") or "configured",
                "auth": _normalize_repository_auth(repository.get("auth")),
            }
        )

    created_at = datetime.now(UTC)
    workspace_id = (
        f"router-{created_at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    workspace = root / workspace_id
    workspace.mkdir(parents=True, exist_ok=False)
    for directory in (
        workspace / "benchmax" / "tasks",
        workspace / "benchmax" / "artifacts",
        workspace / "router" / "checkpoints",
        workspace / "router" / "reports",
        workspace / "litellm",
    ):
        directory.mkdir(parents=True)

    task_count = len(normalized_repositories) * tasks_per_repo
    planned_rollouts = task_count * len(routes) * repetitions
    estimated_evaluation_cost_usd = round(
        planned_rollouts * average_run_cost_usd,
        2,
    )
    manifest = {
        "schema_version": "1",
        "workspace_id": workspace_id,
        "created_at": created_at.isoformat(),
        "status": "awaiting_task_extraction",
        "privacy_mode": privacy_mode,
        "repositories": normalized_repositories,
        "candidate_routes": [asdict(route) for route in routes],
        "benchmark": {
            "environment": "harbor",
            "sandbox_provider": "modal",
            "verifier": "benchmax_model_router_gate",
            "tasks_per_repo": tasks_per_repo,
            "repetitions": repetitions,
            "planned_tasks": task_count,
            "planned_rollouts": planned_rollouts,
            "average_run_cost_usd": average_run_cost_usd,
            "estimated_evaluation_cost_usd": estimated_evaluation_cost_usd,
            "eval_ratio": 0.2,
        },
        "router": {
            "architecture": "benchmax_router_rungs_then_small_llm",
            "output_schema": "router/training_contract.json",
            "selection_policy": "cheapest_above_quality_threshold",
            "training_method": "supervised_fine_tuning",
            "base_model": "Qwen/Qwen3.5-0.8B",
        },
        "benchmax_workflow": {
            "implementation": "examples/model_router",
            "stages": [
                "mine",
                "convert",
                "gate",
                "collect",
                "build_dataset",
                "router_rung",
                "scoreboard",
            ],
        },
    }

    _write_json(workspace / "manifest.json", manifest)
    _write_json(
        workspace / "benchmax" / "environment.json",
        {
            "environment": "harbor",
            "dataset_source": "codeprobe_repository_history",
            "sandbox": {
                "provider": "modal",
                "isolation": "one_ephemeral_sandbox_per_rollout",
            },
            "verifier": {
                "type": "oracle_nop_gate",
                "required": True,
                "oracle_runs": 3,
                "nop_runs": 3,
                "pr_test_overlay": True,
                "trajectory_audit": True,
            },
            "splits": {
                "strategy": "temporal_and_repository_held_out",
                "eval_ratio": 0.2,
            },
        },
    )
    _write_json(
        workspace / "router" / "training_contract.json",
        _router_contract(routes),
    )
    _write_json(
        workspace / "router" / "training_config.json",
        {
            "schema_version": "1",
            "method": "supervised_fine_tuning",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "base_model_ablation": "Qwen/Qwen3.5-0.8B-Base",
            "adapter": "lora",
            "epochs": 3,
            "learning_rate": 0.0002,
            "max_sequence_length": 8192,
            "train_file": "router/data/train.jsonl",
            "eval_file": "router/data/eval.jsonl",
            "output_dir": "router/checkpoints/qwen35-08b-sft-v1",
            "objective": (
                "predict per-route success probability and token classes; "
                "selection remains deterministic policy code"
            ),
        },
    )
    _write_json(
        workspace / "litellm" / "route_registry.json",
        {
            "schema_version": "1",
            "routes": [
                {
                    **asdict(route),
                    "credentials": f"env:{route.provider.upper()}_API_KEY",
                }
                for route in routes
            ],
        },
    )
    _write_json(
        workspace / "benchmax" / "task_schema.json",
        {
            "required": [
                "task_id",
                "repository",
                "base_commit",
                "task_text",
                "verifier",
            ],
            "properties": {
                "task_id": {"type": "string"},
                "repository": {"type": "string"},
                "base_commit": {"type": "string"},
                "task_text": {"type": "string"},
                "verifier": {"type": "object"},
            },
        },
    )
    (workspace / "benchmax" / "tasks" / "train.jsonl").touch()
    (workspace / "benchmax" / "tasks" / "eval.jsonl").touch()
    (workspace / "NEXT_STEPS.md").write_text(
        _next_steps(workspace_id, privacy_mode),
        encoding="utf-8",
    )

    files = sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    )
    return {
        "workspace_id": workspace_id,
        "workspace_path": str(workspace),
        "status": manifest["status"],
        "summary": manifest["benchmark"],
        "files": files,
        "next_command": (
            f"castform-router benchmax {workspace} --through gate --execute"
        ),
    }


def _router_contract(routes: list[TrainingRoute]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "learned_model_input": {
            "schema_version": "1",
            "request_id": "<string>",
            "task": {
                "text": "<pre-solve task text>",
                "domain": "software_engineering",
            },
            "user_context": {},
            "workspace_context": {
                "repository": "<owner/repo>",
                "language": "<language when known>",
                "tools": ["repository", "tests"],
            },
            "candidate_routes": [
                {
                    "route_id": route.route_id,
                    "harness": route.harness,
                    "model": route.model,
                    "provider": route.provider,
                }
                for route in routes
            ],
        },
        "learned_model_output": {
            "schema_version": "1",
            "router_model_version": "qwen35-08b-sft-<version>",
            "predictions": [
                {
                    "route_id": route.route_id,
                    "success_probability": "<float 0..1>",
                    "expected_input_tokens": "<integer >= 0>",
                    "expected_cache_read_tokens": "<integer >= 0>",
                    "expected_output_tokens": "<integer >= 0>",
                }
                for route in routes
            ],
        },
        "policy_only_fields": [
            "live_price",
            "availability",
            "quality_threshold",
            "route_override",
        ],
    }


def _normalize_repository_auth(value: object) -> dict[str, Any]:
    if value is None:
        return {"strategy": "public"}
    if not isinstance(value, dict):
        raise ValueError("repository auth must be an object")
    strategy = value.get("strategy")
    if strategy == "public":
        return {"strategy": "public"}
    if strategy == "token_env":
        return {
            "strategy": strategy,
            "token_env": _credential_env(value.get("token_env"), "token_env"),
        }
    if strategy == "github_app":
        normalized = {
            "strategy": strategy,
            "app_id_env": _credential_env(value.get("app_id_env"), "app_id_env"),
            "private_key_env": _credential_env(
                value.get("private_key_env"),
                "private_key_env",
            ),
            "installation_id_env": _credential_env(
                value.get("installation_id_env"),
                "installation_id_env",
            ),
        }
        installation_token_env = value.get("installation_token_env")
        if installation_token_env is not None:
            normalized["installation_token_env"] = _credential_env(
                installation_token_env,
                "installation_token_env",
            )
        return normalized
    raise ValueError(
        "repository auth strategy must be public, token_env, or github_app"
    )


def _credential_env(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ENV_REFERENCE.fullmatch(value):
        raise ValueError(
            f"repository auth {field} must be an uppercase environment variable name"
        )
    return value


def _next_steps(workspace_id: str, privacy_mode: str) -> str:
    location = (
        "the customer-controlled runner"
        if privacy_mode == "customer_runner"
        else "the Castform-hosted runner"
    )
    return f"""# Next steps for {workspace_id}

This workspace is configured but not ready to spend model credits.

1. Install the Castform GitHub App or provide a short-lived checkout credential.
2. Run CodeProbe mining in {location}.
3. Convert tasks with Benchmax's PR-era test overlay and leak-guarded checkout.
4. Gate every task: oracle must pass and nop must fail across all runs.
5. Collect the allowed harness/model routes with Harbor.
6. Build the audited temporal dataset and run the existing router baselines.
7. Score against always-route, random, and oracle policies.
8. Format the audited matrix:
   `castform-router format-training-data training_runs/{workspace_id}`.
9. Install the optional training dependencies with `uv sync --extra training`.
10. Run supervised LoRA training:
    `castform-router train-sft training_runs/{workspace_id}`.
11. Serve the checkpoint through an OpenAI-compatible endpoint and run:
    `castform-router evaluate-trained training_runs/{workspace_id}`.
12. Use the emitted `picks_trained.jsonl` in the existing Benchmax scoreboard.
13. Shadow the router before allowing it to control harness selection.

Repository source is not copied by this generator.
"""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
