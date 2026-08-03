"""Build, validate, and launch the Chroma RAG example."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from typing import Any

from benchmax.bundle import dump_bundle
from castform.platform import ensure_session, upload_assets
from chroma_rag_env import ChromaRagEnv
from data import (
    DATA_DIR,
    DEFAULT_QUESTION_COUNT,
    build_chunks,
    connection_args,
    ingest_corpus,
    prepare_data,
    require_dataset_files,
)

RUN_NAME = "chroma-rag"
RUNTIME_DEPENDENCIES = ["chromadb>=1.5.9,<2"]
TRAINING_ARGS = {
    "model": "Qwen/Qwen3.5-4B",
    "max_context_tokens": 8_192,
    "num_epochs": 3,
}


def _constructor_args() -> dict[str, Any]:
    from castform import config

    args = connection_args()
    api_key = os.environ.get("CHROMA_API_KEY", "").strip() or None
    if args.get("tenant") and not api_key:
        raise RuntimeError("Chroma Cloud validation and launch require CHROMA_API_KEY")
    return {
        "judge_base_url": config.llm_url(),
        "embedding_base_url": config.llm_url(),
        "api_key": api_key,
        **args,
    }


def validate(env: ChromaRagEnv, uploaded_assets):
    from castform import validate_environment

    report = asyncio.run(
        validate_environment(
            env,
            model="gpt-5.4-mini",
            split="eval",
            base_dir=DATA_DIR,
            remote_assets=uploaded_assets,
            max_context_tokens=4_096,
        )
    )
    _print_validation(report)
    return report


def launch(uploaded_assets, *, assume_yes: bool = False) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes and input("launch Chroma RAG training? [y/N] ").lower() not in {
        "y",
        "yes",
    }:
        print("launch: cancelled")
        return None
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=RUN_NAME,
            launcher_args=TRAINING_ARGS,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("ingest", "data", "validate", "launch"), nargs="?", default="validate"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--question-count",
        type=int,
        default=DEFAULT_QUESTION_COUNT,
        help=f"grounded Q&A examples to generate (default: {DEFAULT_QUESTION_COUNT})",
    )
    args = parser.parse_args(argv)
    if args.question_count < 2:
        parser.error("--question-count must be at least 2")

    if args.action == "ingest":
        from castform.rag.example_data import RagDataModelConfig

        ensure_session()
        ingest_corpus(build_chunks(), RagDataModelConfig.from_env("CHROMA_RAG"))
        return 0
    if args.action == "data":
        ensure_session()
        prepare_data(force=args.force, question_count=args.question_count)
        return 0

    files = require_dataset_files()
    constructor_args = _constructor_args()
    ensure_session()
    bundle = dump_bundle(
        ChromaRagEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    assets = upload_assets(bundle=bundle, dataset_files=files, run_name=RUN_NAME)
    report = validate(ChromaRagEnv(**constructor_args), assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        launch(assets, assume_yes=args.yes)
    return 0


def _print_validation(report: Any) -> None:
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            suffix = f" error={outcome.error}" if outcome.error else ""
            print(f"{location} {rollout_id}: {outcome.termination_reason}{suffix}")
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"{location} {rollout_id}: {error}")
    print("validation passed" if report.ok else "validation failed")


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
