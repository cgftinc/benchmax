"""Harbor environment seed written by ``castform setup --template harbor``.

Use this shape when Harbor owns the task package, sandbox, harness, and verifier.
The package dataset resolves at runtime, so Castform uploads only this environment
bundle. Modal and verifier credentials enter through explicit CLI arguments and
are serialized into the same constructor kwargs used locally and remotely.

The stock Mini-SWE agent is intentional. Replace it with a bundled/custom harness
only when the task requires a different loop or hermetic installation.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmax.envs.harbor import (
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
)
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)

from benchmax.bundle import dump_bundle
from benchmax.envs import Environment
from castform import validate_environment
from castform.platform.client import TrainerClient
from castform.platform.environment_assets import upload_assets
from castform.platform.login import ensure_session

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CustomHarborEnv(HarborEnv):
    """A Harbor package on Modal using Harbor's stock Mini-SWE harness."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_ref: str,
        sandbox_credentials: ModalCredentials,
        verifier_env: Mapping[str, str] | None = None,
    ) -> None:
        if not dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")
        if not dataset_ref.strip():
            raise ValueError("dataset_ref must be non-empty")
        super().__init__(
            dataset=DatasetConfig(name=dataset_name, ref=dataset_ref),
            eval_ratio=0.1,
            trial=HarborTrialTemplate(
                # Leave model_name and sampling kwargs unset. HarborEnv injects
                # Castform's tracked model session for each rollout.
                agent=TrialAgentConfig(name="mini-swe-agent"),
                environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
                verifier=TrialVerifierConfig(env=dict(verifier_env or {})),
            ),
            sandbox_credentials=sandbox_credentials,
            # Do not impose a small environment-side rollout bottleneck. Provider
            # and trainer capacity still bound actual concurrency.
            max_concurrent_trials=1000,
        )


VALIDATE_CONFIG = {
    "model": "gpt-5.4-mini",
    "max_context_tokens": 16_384,
    "local_timeout_seconds": 1_800,
}

LAUNCH_CONFIG = {
    "model": "Qwen/Qwen3.5-4B",
    "max_context_tokens": 16_384,
    "num_epochs": 2,
}

RUNTIME_DEPENDENCIES = ["harbor[modal]>=0.18.0,<0.19"]


def _parse_verifier_env(assignments: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not _ENV_NAME.fullmatch(name) or not value:
            raise ValueError("--verifier-env values must use a non-empty NAME=VALUE form")
        if name in environment:
            raise ValueError(f"duplicate verifier environment variable: {name}")
        environment[name] = value
    return environment


def _constructor_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI inputs into the exact kwargs bundled for remote workers."""
    return {
        "dataset_name": args.dataset,
        "dataset_ref": args.dataset_ref,
        "sandbox_credentials": ModalCredentials(
            token_id=args.modal_token_id,
            token_secret=args.modal_token_secret,
        ),
        "verifier_env": _parse_verifier_env(args.verifier_env),
    }


def _run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    dataset = re.sub(r"[^A-Za-z0-9._-]+", "-", args.dataset).strip("-")
    return f"harbor-{dataset}"


def generate_data(*, dataset_name: str, dataset_ref: str) -> None:
    print(
        f"data: {dataset_name}@{dataset_ref} resolves through Harbor at runtime; "
        "no JSONL upload is needed"
    )


def _print_validation(report: Any) -> None:
    for location in ("static", "local", "remote"):
        warnings = getattr(report, f"{location}_warnings", {}) or {}
        for item, messages in warnings.items():
            values = messages if isinstance(messages, list) else [messages]
            for message in values:
                print(f"⚠️ {location} {item}: {message}")
    for item, error in (getattr(report, "static_errors", {}) or {}).items():
        print(f"❌ static {item}: {error}")
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
                continue
            mark = (
                "✅"
                if outcome.termination_reason in Environment.scorable_termination_reasons
                else "❌"
            )
            suffix = f" error={outcome.error}" if outcome.error else ""
            print(
                f"{mark} {location} {rollout_id}: "
                f"{outcome.termination_reason} {dict(outcome.rewards)}{suffix}"
            )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def validate(env: CustomHarborEnv, uploaded_assets: Any) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        report = asyncio.run(
            validate_environment(
                env,
                model=str(VALIDATE_CONFIG["model"]),
                split="train",
                base_dir=Path(tmp),
                remote_assets=uploaded_assets,
                max_context_tokens=int(VALIDATE_CONFIG["max_context_tokens"]),
                local_timeout_seconds=float(VALIDATE_CONFIG["local_timeout_seconds"]),
            )
        )
    _print_validation(report)
    return report


def launch(
    uploaded_assets: Any,
    *,
    run_name: str,
    assume_yes: bool = False,
) -> str | None:
    if not assume_yes:
        reply = input(f"Launch '{run_name}' on GPUs — this spends credits. Continue? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("launch: aborted.")
            return None
    with TrainerClient() as client:
        run_id = client.launch_training_run(
            name=run_name,
            launcher_args=LAUNCH_CONFIG,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started run {run_id}")
    return run_id


def _require_execution_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    missing = [
        flag
        for flag, value in (
            ("--modal-token-id", args.modal_token_id),
            ("--modal-token-secret", args.modal_token_secret),
        )
        if not value
    ]
    if missing:
        parser.error(f"{', '.join(missing)} required for validate and launch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Castform for a Harbor package on Modal.")
    parser.add_argument(
        "action",
        nargs="?",
        default="validate",
        choices=["data", "validate", "launch"],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Harbor package dataset, for example aime/aime.",
    )
    parser.add_argument("--dataset-ref", default="latest")
    parser.add_argument("--run-name")
    parser.add_argument("--modal-token-id")
    parser.add_argument("--modal-token-secret")
    parser.add_argument(
        "--verifier-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Verifier credential/config; repeat for each required variable.",
    )
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] resolving data")
    generate_data(dataset_name=args.dataset, dataset_ref=args.dataset_ref)
    if args.action == "data":
        return 0

    _require_execution_args(parser, args)
    constructor_args = _constructor_args(args)
    run_name = _run_name(args)
    ensure_session()

    print(f"[stage 2/{total_stages}] bundling Harbor environment")
    bundle = dump_bundle(
        CustomHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment")
    uploaded_assets = upload_assets(bundle=bundle, run_name=run_name)
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")

    print(f"[stage 4/{total_stages}] validating Harbor trials")
    report = validate(CustomHarborEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        return (
            0 if launch(uploaded_assets, run_name=run_name, assume_yes=args.yes) is not None else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
