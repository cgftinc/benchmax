"""AIME Harbor environment using the offline-installed Mini-SWE agent.

The dataset (aime/aime@latest) resolves through Harbor at trainer runtime, so
only the environment bundle is uploaded. Validation runs real Modal sandbox
trials. Modal credentials are mandatory CLI arguments; they are bundled into
the environment constructor args so trainer-side trials can reach Modal.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmax.bundle import dump_bundle
from benchmax.envs.harbor import (
    BundledAgentSource,
    BundledHarborAgent,
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
)
from castform.platform import ensure_session, upload_assets
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)
from harness.aime_agent import MINI_SWE_AGENT_VERSION

_HARNESS_SOURCE = BundledAgentSource.from_directory(
    Path(__file__).parent / "harness",
    files=(
        "aime_agent.py",
        "mini_swe_probe.py",
        "castform_model.py",
        "run_mini_castform.py",
    ),
)


def mini_swe_harness(*, max_timeout_secs: float | None = None) -> BundledHarborAgent:
    """The default harness: the upstream mini-swe-agent loop bundled from ``harness/``."""

    return BundledHarborAgent(
        config=TrialAgentConfig(
            import_path="aime_agent:UpstreamMiniSweAgent",
            kwargs={"version": MINI_SWE_AGENT_VERSION},
            max_timeout_sec=max_timeout_secs,
        ),
        source=_HARNESS_SOURCE,
    )


class AimeMiniSweHarborEnv(HarborEnv):
    """AIME latest on Modal; the agent harness defaults to the offline Mini-SWE loop."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        harness: BundledHarborAgent | None = None,
        max_agent_timeout_secs: float | None = None,
    ) -> None:
        if harness is not None and max_agent_timeout_secs is not None:
            raise ValueError(
                "max_agent_timeout_secs applies only to the default harness; "
                "set max_timeout_sec on the custom harness config instead"
            )
        super().__init__(
            dataset=DatasetConfig(name="aime/aime", ref="latest"),
            eval_ratio=0.1,
            trial=HarborTrialTemplate(
                agent=harness
                if harness is not None
                else mini_swe_harness(max_timeout_secs=max_agent_timeout_secs),
                environment=TrialEnvironmentConfig(
                    type=EnvironmentType.MODAL,
                ),
                verifier=TrialVerifierConfig(),
                trials_dir=Path("/tmp/castform-aime-harbor-trials"),
            ),
            sandbox_credentials=sandbox_credentials,
            max_concurrent_trials=1000,
        )


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3.5-35B-A3B"
VALIDATE_MODEL = "gpt-5.4-mini"
RUNTIME_DEPENDENCIES = ["harbor[modal]>=0.18.0,<0.19"]
RUN_NAME = "aime"
TRAINING_ARGS = {"model": MODEL}


def _constructor_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sandbox_credentials": ModalCredentials(
            token_id=args.modal_token_id, token_secret=args.modal_token_secret
        ),
    }


def generate_data(*, force: bool) -> None:
    del force
    print("data: aime/aime@latest resolves through Harbor at runtime — nothing to download")


def validate(env: AimeMiniSweHarborEnv, uploaded_assets: Any) -> Any:
    from castform import validate_environment

    with tempfile.TemporaryDirectory() as tmp:
        report = asyncio.run(
            validate_environment(
                env,
                model=VALIDATE_MODEL,
                split="eval",
                base_dir=Path(tmp),
                remote_assets=uploaded_assets,
            )
        )
    _print_validation(report)
    return report


def launch(uploaded_assets: Any, *, assume_yes: bool) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
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


def _print_validation(report: Any) -> None:
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
            else:
                print(
                    f"✅ {location} {rollout_id}: "
                    f"{outcome.termination_reason} {dict(outcome.rewards)}"
                )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("data", "validate", "launch"),
        default="validate",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the launch confirmation",
    )
    parser.add_argument(
        "--modal-token-id",
        required=True,
        help="Modal token id for sandbox trials (bundled into constructor args).",
    )
    parser.add_argument(
        "--modal-token-secret",
        required=True,
        help="Modal token secret for sandbox trials (bundled into constructor args).",
    )
    args = parser.parse_args(argv)
    constructor_args = _constructor_args(args)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    generate_data(force=args.force)
    if args.action == "data":
        return 0

    ensure_session()
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        AimeMiniSweHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment")
    uploaded_assets = upload_assets(bundle=bundled_environment, run_name=RUN_NAME)
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(AimeMiniSweHarborEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
