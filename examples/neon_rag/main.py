"""A complete Neon-backed retrieval environment and Castform workflow."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from typing import Any
from urllib.parse import urlsplit

from benchmax.bundle import dump_bundle
from castform.platform import ensure_session, upload_assets
from data import (
    DATA_DIR,
    DEFAULT_QUESTION_COUNT,
    build_chunks,
    ingest_corpus,
    model_config,
    prepare_data,
    require_dataset_files,
)
from neon_rag_env import NeonRagEnv

RUN_NAME = "neon-rag"
VALIDATION_MODEL = "gpt-5.4-mini"
TRAINING_ARGS = {
    "model": "Qwen/Qwen3.5-4B",
    "max_context_tokens": 10_000,
    "num_epochs": 3,
}
RUNTIME_DEPENDENCIES = [
    "pgvector>=0.3.0",
    "psycopg[binary]>=3.2.0",
]


def validate(env: NeonRagEnv, uploaded_assets):
    from castform import validate_environment

    report = asyncio.run(
        validate_environment(
            env,
            model=VALIDATION_MODEL,
            split="eval",
            base_dir=DATA_DIR,
            remote_assets=uploaded_assets,
            max_context_tokens=10_000,
        )
    )
    _print_validation(report)
    return report


def launch(
    uploaded_assets,
    *,
    run_name: str = RUN_NAME,
    assume_yes: bool = False,
) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch Neon RAG training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("launch: cancelled")
            return None

    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=run_name,
            launcher_args=TRAINING_ARGS,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def _constructor_args(search_database_url: str) -> dict[str, str]:
    from castform import config

    base_url = config.llm_url()
    return {
        "judge_base_url": base_url,
        "embedding_base_url": base_url,
        "search_database_url": search_database_url,
    }


def _require_database_url(
    parser: argparse.ArgumentParser,
    *,
    action: str,
    flag: str,
    value: str | None,
) -> str:
    if not value:
        parser.error(f"{action} requires {flag}")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or not parsed.path.lstrip("/")
    ):
        parser.error(f"{flag} must be a valid PostgreSQL database URL")
    return value


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("ingest", "data", "validate", "launch"),
        default="validate",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip launch confirmation")
    parser.add_argument(
        "--question-count",
        type=int,
        default=DEFAULT_QUESTION_COUNT,
        help=f"grounded Q&A examples to generate (default: {DEFAULT_QUESTION_COUNT})",
    )
    parser.add_argument("--neon-data-preparation-database-url")
    parser.add_argument("--neon-search-database-url")
    args = parser.parse_args(argv)
    if args.question_count < 2:
        parser.error("--question-count must be at least 2")

    if args.action == "ingest":
        database_url = _require_database_url(
            parser,
            action="ingest",
            flag="--neon-data-preparation-database-url",
            value=args.neon_data_preparation_database_url,
        )
        ensure_session()
        config = model_config()
        ingest_corpus(
            build_chunks(),
            config,
            data_preparation_database_url=database_url,
        )
        return 0

    if args.action == "data":
        database_url = _require_database_url(
            parser,
            action="data",
            flag="--neon-data-preparation-database-url",
            value=args.neon_data_preparation_database_url,
        )
        ensure_session()
        print("[stage 1/1] ingesting documents and generating grounded questions")
        prepare_data(
            data_preparation_database_url=database_url,
            force=args.force,
            question_count=args.question_count,
        )
        return 0

    search_database_url = _require_database_url(
        parser,
        action=args.action,
        flag="--neon-search-database-url",
        value=args.neon_search_database_url,
    )
    data_preparation_database_url: str | None = None
    if args.action == "launch":
        data_preparation_database_url = _require_database_url(
            parser,
            action="launch",
            flag="--neon-data-preparation-database-url",
            value=args.neon_data_preparation_database_url,
        )

    ensure_session()
    if args.action == "launch":
        total_stages = 6
        print(f"[stage 1/{total_stages}] preparing corpus and grounded questions")
        dataset_files = prepare_data(
            data_preparation_database_url=data_preparation_database_url,
            force=args.force,
            question_count=args.question_count,
        )
        bundle_stage = 2
    else:
        total_stages = 4
        dataset_files = require_dataset_files()
        bundle_stage = 1

    constructor_args = _constructor_args(search_database_url)
    print(f"[stage {bundle_stage}/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        NeonRagEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    upload_stage = bundle_stage + 1
    print(f"[stage {upload_stage}/{total_stages}] uploading environment and dataset")
    uploaded_assets = upload_assets(
        bundle=bundled_environment,
        dataset_files=dataset_files,
        run_name=RUN_NAME,
    )
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    validation_stage = upload_stage + 1
    print(f"[stage {validation_stage}/{total_stages}] validating environment")
    report = validate(NeonRagEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    passed_stage = validation_stage + 1
    print(f"[stage {passed_stage}/{total_stages}] validation passed")
    if args.action == "launch":
        print(f"[stage {passed_stage + 1}/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes)
    return 0


def _print_validation(report: Any) -> None:
    from benchmax.envs.environment import Environment

    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
            else:
                mark = (
                    "✅"
                    if outcome.termination_reason in Environment.scorable_termination_reasons
                    else "❌"
                )
                error_suffix = f" error={outcome.error}" if outcome.error else ""
                print(
                    f"{mark} {location} {rollout_id}: "
                    f"{outcome.termination_reason} {dict(outcome.rewards)}{error_suffix}"
                )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
