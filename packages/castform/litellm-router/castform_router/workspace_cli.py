"""Command-line interface for code-first router projects."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from castform_router.benchmax_workflow import (
    STAGES,
    build_benchmax_plan,
    execute_benchmax_plan,
    select_plan_steps,
    write_benchmax_plan,
)
from castform_router.project import (
    create_training_project,
    load_project_spec,
)
from castform_router.pull_requests import materialize_pull_request_tasks
from castform_router.training_data import format_benchmax_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="castform-router",
        description="Validate and create Castform router-training workspaces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate a project JSON file without writing a workspace",
    )
    validate_parser.add_argument("config", type=Path)

    create_parser = subparsers.add_parser(
        "create",
        help="create a Castform + Benchmax training workspace",
    )
    create_parser.add_argument("config", type=Path)
    create_parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_runs"),
        help="workspace parent directory (default: training_runs)",
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="create a workspace and generate the Benchmax workflow plan",
    )
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_runs"),
        help="workspace parent directory (default: training_runs)",
    )
    _add_benchmax_arguments(prepare_parser, include_execute=True)

    benchmax_parser = subparsers.add_parser(
        "benchmax",
        help="print or execute Benchmax's existing model-router workflow",
    )
    benchmax_parser.add_argument("workspace", type=Path)
    _add_benchmax_arguments(benchmax_parser, include_execute=True)

    materialize_parser = subparsers.add_parser(
        "materialize",
        help="legacy: fetch PR metadata without Benchmax conversion or gating",
    )
    materialize_parser.add_argument("workspace", type=Path)

    format_parser = subparsers.add_parser(
        "format-training-data",
        help="convert an audited Benchmax dataset into Qwen SFT JSONL",
    )
    format_parser.add_argument("workspace", type=Path)
    format_parser.add_argument(
        "--dataset",
        type=Path,
        help="override benchmax/model_router/dataset.jsonl",
    )
    format_parser.add_argument(
        "--held-out-repo",
        action="append",
        default=[],
        help="repository to reserve for evaluation (repeatable)",
    )

    scaffold_sft_parser = subparsers.add_parser(
        "scaffold-model-router-sft",
        help="scaffold SFT data directly from castform-ai/model-router",
    )
    scaffold_sft_parser.add_argument(
        "source",
        type=Path,
        help="local checkout of castform-ai/model-router",
    )
    scaffold_sft_parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_runs/model-router-sft"),
        help="new training workspace (default: training_runs/model-router-sft)",
    )
    scaffold_sft_parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="canonical model-router route to include (repeatable)",
    )
    scaffold_sft_parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.5,
        help="newest fraction held out within each repository (default: 0.5)",
    )

    train_parser = subparsers.add_parser(
        "train-sft",
        help="fine-tune the configured Qwen 0.8B router with LoRA",
    )
    train_parser.add_argument("workspace", type=Path)
    train_parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Hugging Face checkpoint (default: Qwen/Qwen3.5-0.8B)",
    )
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--max-sequence-length", type=int, default=8192)

    evaluate_parser = subparsers.add_parser(
        "evaluate-trained",
        help="run the served router on held-out examples and emit Benchmax picks",
    )
    evaluate_parser.add_argument("workspace", type=Path)
    evaluate_parser.add_argument(
        "--router-url",
        default=os.getenv(
            "CASTFORM_ROUTER_MODEL_BASE_URL",
            "http://localhost:4000",
        ),
    )
    evaluate_parser.add_argument(
        "--model",
        default=os.getenv(
            "CASTFORM_ROUTER_MODEL_NAME",
            "castform-router-0.8b",
        ),
    )
    evaluate_parser.add_argument(
        "--api-key",
        default=os.getenv(
            "CASTFORM_ROUTER_MODEL_API_KEY",
            os.getenv("LITELLM_MASTER_KEY", "sk-local-dev"),
        ),
    )
    evaluate_parser.add_argument("--quality-threshold", type=float, default=0.84)

    args = parser.parse_args(argv)
    try:
        if args.command == "scaffold-model-router-sft":
            from castform_router.model_router_scaffold import (
                scaffold_model_router_sft,
                summary_json,
            )

            result = scaffold_model_router_sft(
                args.source,
                output_dir=args.output,
                models=tuple(args.model),
                eval_ratio=args.eval_ratio,
            )
            print(summary_json(result))
            return 0

        if args.command == "evaluate-trained":
            from castform_router.trained_evaluator import (
                evaluate_trained_router,
            )

            result = evaluate_trained_router(
                workspace=args.workspace.resolve(),
                base_url=args.router_url,
                model=args.model,
                api_key=args.api_key,
                quality_threshold=args.quality_threshold,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "format-training-data":
            workspace = args.workspace.resolve()
            dataset = (
                args.dataset.resolve()
                if args.dataset is not None
                else workspace
                / "benchmax"
                / "model_router"
                / "dataset.jsonl"
            )
            manifest = json.loads(
                (workspace / "manifest.json").read_text(encoding="utf-8")
            )
            eval_ratio = float(
                manifest.get("benchmark", {}).get("eval_ratio", 0.2)
            )
            result = format_benchmax_dataset(
                dataset,
                manifest_path=workspace / "manifest.json",
                output_dir=workspace / "router" / "data",
                eval_ratio=eval_ratio,
                held_out_repositories=args.held_out_repo,
            )
            print(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "train-sft":
            from castform_router.train_sft import train_qwen_router

            workspace = args.workspace.resolve()
            result = train_qwen_router(
                train_file=workspace / "router" / "data" / "train.jsonl",
                eval_file=workspace / "router" / "data" / "eval.jsonl",
                output_dir=(
                    workspace
                    / "router"
                    / "checkpoints"
                    / "qwen35-08b-sft-v2"
                ),
                model_name=args.model,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_sequence_length=args.max_sequence_length,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.command == "benchmax":
            workspace = args.workspace.resolve()
            plan = _benchmax_plan_from_args(workspace, args)
            plan_path = write_benchmax_plan(workspace, plan)
            if args.execute:
                result = execute_benchmax_plan(
                    workspace,
                    plan,
                    from_stage=args.from_stage,
                    through_stage=args.through,
                )
                result["plan"] = str(plan_path)
                print(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(
                    json.dumps(
                        {
                            "mode": "dry_run",
                            "plan": str(plan_path),
                            "from_stage": args.from_stage,
                            "through_stage": args.through,
                            "steps": select_plan_steps(
                                plan,
                                from_stage=args.from_stage,
                                through_stage=args.through,
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            return 0

        if args.command == "materialize":
            workspace = args.workspace.resolve()
            project_spec = json.loads(
                (workspace / "project.spec.json").read_text(encoding="utf-8")
            )
            result = materialize_pull_request_tasks(
                workspace,
                settings=project_spec.get("pull_requests", {}),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        config_path = args.config.resolve()
        spec = load_project_spec(config_path)
        if args.command == "validate":
            print(f"valid: {config_path}")
            return 0
        result = create_training_project(
            spec,
            output_root=args.output.resolve(),
        )
        if args.command == "prepare":
            workspace = Path(result["workspace_path"])
            plan = _benchmax_plan_from_args(workspace, args)
            plan_path = write_benchmax_plan(workspace, plan)
            result["benchmax_workflow"] = {
                "implementation": plan["implementation"],
                "plan": str(plan_path),
                "router_rung": plan["router_rung"],
            }
            result["files"] = sorted(
                [
                    *result["files"],
                    str(plan_path.relative_to(workspace)),
                ]
            )
            result["status"] = "ready_for_benchmax_mining"
            result["next_command"] = (
                f"castform-router benchmax {workspace} "
                "--through gate --execute"
            )
            if args.execute:
                result["execution"] = execute_benchmax_plan(
                    workspace,
                    plan,
                    from_stage=args.from_stage,
                    through_stage=args.through,
                )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_benchmax_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_execute: bool,
) -> None:
    parser.add_argument(
        "--workflow-dir",
        type=Path,
        help=(
            "local examples/model_router directory; otherwise use a local "
            "Benchmax checkout or fetch the model-router branch into the workspace"
        ),
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default="setup",
        help="first stage to include (default: setup)",
    )
    parser.add_argument(
        "--through",
        choices=STAGES,
        default="scoreboard",
        help="last stage to include (default: scoreboard)",
    )
    parser.add_argument(
        "--agent-network",
        choices=["allowlist", "public"],
        default="allowlist",
        help="Harbor agent network policy (default: allowlist)",
    )
    parser.add_argument(
        "--gate-k",
        type=int,
        default=3,
        help="oracle and nop runs per task (default: 3)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Harbor concurrency (default: 4)",
    )
    parser.add_argument(
        "--router-rung",
        choices=["knn", "profile", "baseline"],
        default="knn",
        help="existing Benchmax router baseline to score (default: knn)",
    )
    parser.add_argument(
        "--router-model",
        default="claude-sonnet-4-6",
        help="router model for profile or baseline rungs",
    )
    parser.add_argument(
        "--codeprobe-llm",
        action="store_true",
        help=(
            "allow CodeProbe model calls for instruction enrichment; "
            "default uses --no-llm"
        ),
    )
    if include_execute:
        parser.add_argument(
            "--execute",
            action="store_true",
            help=(
                "execute the plan; quality mining, collection, and prompted "
                "router stages may spend model credits"
            ),
        )


def _benchmax_plan_from_args(
    workspace: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    return build_benchmax_plan(
        workspace,
        workflow_dir=args.workflow_dir,
        agent_network=args.agent_network,
        gate_k=args.gate_k,
        concurrency=args.concurrency,
        router_rung=args.router_rung,
        router_model=args.router_model,
        codeprobe_llm=args.codeprobe_llm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
