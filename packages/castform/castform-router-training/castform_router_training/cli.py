"""Command-line entry point for the simple training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from castform_router_training.dataset import build_dataset
from castform_router_training.model_router_scaffold import (
    scaffold_model_router_sft,
    summary_json,
)
from castform_router_training.project import load_project
from castform_router_training.repositories import mine_tasks
from castform_router_training.train import train_qwen


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="castform-router-training")
    commands = value.add_subparsers(dest="command", required=True)

    mine = commands.add_parser("mine", help="clone repos and export recent changes as tasks")
    mine.add_argument("project", type=Path)
    mine.add_argument("--output", type=Path, required=True)
    mine.add_argument("--limit-per-repo", type=int, default=100)

    dataset = commands.add_parser("dataset", help="join tasks and measured rollout outcomes")
    dataset.add_argument("project", type=Path)
    dataset.add_argument("--tasks", type=Path, required=True)
    dataset.add_argument("--outcomes", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)

    scaffold = commands.add_parser(
        "scaffold-model-router",
        help="build a pinned SFT workspace from castform-ai/model-router",
    )
    scaffold.add_argument("source", type=Path)
    scaffold.add_argument("--output", type=Path, required=True)
    scaffold.add_argument("--model", action="append", required=True)
    scaffold.add_argument("--eval-ratio", type=float, default=0.5)

    train = commands.add_parser("train", help="LoRA fine-tune Qwen on the generated JSONL")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=float, default=1.0)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "mine":
        path = mine_tasks(
            load_project(args.project), args.output, limit_per_repo=args.limit_per_repo
        )
        print(path)
    elif args.command == "dataset":
        count = build_dataset(load_project(args.project), args.tasks, args.outcomes, args.output)
        print(f"wrote {count} examples to {args.output}")
    elif args.command == "scaffold-model-router":
        result = scaffold_model_router_sft(
            args.source,
            output_dir=args.output,
            models=tuple(args.model),
            eval_ratio=args.eval_ratio,
        )
        print(summary_json(result))
    else:
        train_qwen(dataset=args.dataset, model=args.model, output=args.output, epochs=args.epochs)
    return 0
