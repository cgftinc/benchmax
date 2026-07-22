"""Harvey LAB Harbor environment using Harvey's native harness loop.

`python main.py [data|validate|launch|all]` drives the loop. The dataset
(harveyai/lab@latest) resolves through Harbor at trainer runtime, so the data
stage has nothing to download. Validation runs two real Modal sandbox trials.
Launch uploads the bundle and starts a GPU run (explicit, confirmed — it
spends credits). Credentials: Modal from ~/.modal.toml, the judge key from
HARVEY_JUDGE_API_KEY (falls back to PLATFORM_API_KEY).

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import tomllib
import uuid
from typing import Any


from pathlib import Path

from benchmax.envs.harbor import (
    BundledAgentSource,
    BundledHarborAgent,
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

_AGENT_SOURCE = BundledAgentSource.from_directory(
    Path(__file__).parent,
    files=("harvey_agent.py", "harvey_runtime.py"),
)


class HarveyLabHarborEnv(HarborEnv):
    """Harvey's latest LAB dataset on Modal with the native Harvey harness."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        # A fixed credential carried in bundles, exactly like the Modal pair;
        # the sandboxed verifier only accepts static environment variables.
        judge_api_key: str,
        judge_model: str = "openai/gpt-5.4-nano",
        judge_base_url: str = "https://llm.castform.dev/v1",
        judge_concurrency: int = 1,
        max_agent_timeout_secs: float | None = None,
        max_concurrent_trials: int | None = 1000,
        eval_ratio: float = 0.1,
    ) -> None:
        if not isinstance(judge_api_key, str) or not judge_api_key:
            raise ValueError("judge_api_key must be a non-empty string")
        if judge_concurrency < 1:
            raise ValueError("judge_concurrency must be positive")
        if not 0 < eval_ratio < 1:
            raise ValueError("eval_ratio must be in (0, 1)")

        normalized_judge_base_url = judge_base_url.rstrip("/")
        verifier_env = {
            "REWARDKIT_JUDGE": judge_model,
            "OPENAI_API_KEY": judge_api_key,
            "OPENAI_BASE_URL": normalized_judge_base_url,
            "OPENAI_API_BASE": normalized_judge_base_url,
            # The published tasks declare this placeholder even when RewardKit
            # is overridden to an OpenAI-compatible judge.
            "ANTHROPIC_API_KEY": judge_api_key,
            "JUDGE_CONCURRENCY": str(judge_concurrency),
        }
        super().__init__(
            dataset=DatasetConfig(name="harveyai/lab", ref="latest"),
            reward_keys=("reward", "partial_credit"),
            eval_ratio=eval_ratio,
            trial=HarborTrialTemplate(
                agent=BundledHarborAgent(
                    config=TrialAgentConfig(
                        import_path="harvey_agent:HarveyHarnessAgent",
                        max_timeout_sec=max_agent_timeout_secs,
                    ),
                    source=_AGENT_SOURCE,
                ),
                environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
                verifier=TrialVerifierConfig(env=verifier_env),
                trials_dir=Path("/tmp/castform-harvey-harbor-trials"),
            ),
            sandbox_credentials=sandbox_credentials,
            max_concurrent_trials=max_concurrent_trials,
        )


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3.5-35B-A3B"
VALIDATE_MODEL = "gpt-5.4-mini"
RUNTIME_DEPENDENCIES = ["harbor[modal]>=0.18.0,<0.19"]
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "castform")


def _constructor_args() -> dict[str, Any]:
    data = tomllib.loads((Path.home() / ".modal.toml").read_text())[MODAL_PROFILE]
    judge_api_key = os.environ.get("HARVEY_JUDGE_API_KEY") or os.environ.get(
        "PLATFORM_API_KEY"
    )
    if not judge_api_key:
        raise SystemExit(
            "set HARVEY_JUDGE_API_KEY (or PLATFORM_API_KEY) for the sandbox verifier"
        )
    return {
        "sandbox_credentials": ModalCredentials(
            token_id=data["token_id"], token_secret=data["token_secret"]
        ),
        "judge_api_key": judge_api_key,
    }


def generate_data(*, force: bool) -> None:
    del force
    print(
        "data: harveyai/lab@latest resolves through Harbor at runtime — nothing to download"
    )


def validate() -> Any:
    from castform import validate_environment

    env = HarveyLabHarborEnv(**_constructor_args())
    with tempfile.TemporaryDirectory() as tmp:
        dataset = asyncio.run(env.create_dataset("eval", Path(tmp)))
        example = dataset[0]
        print(f"validating with example {example.id[:16]}... on Modal")
        report = asyncio.run(
            validate_environment(env, example=example, model=VALIDATE_MODEL)
        )
    for rollout_id, outcome in report.local.items():
        print(
            f"  {rollout_id}: termination={outcome.termination_reason} "
            f"rewards={dict(outcome.rewards)}"
        )
    return report


def launch(*, assume_yes: bool) -> str | None:
    from benchmax.bundle import dump_bundle
    from castform import config
    from castform.platform.client import TrainerClient
    from castform.platform.training_run import upload_training_run

    run_name = f"harvey-{uuid.uuid4().hex[:8]}"
    if not assume_yes:
        reply = input(
            f"Launch {run_name!r} on GPUs — this spends credits. Continue? [y/N] "
        )
        if reply.strip().lower() not in ("y", "yes"):
            print("Launch aborted.")
            return None

    # Bundle-only upload: Harbor resolves the dataset at trainer runtime.
    bundle = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args=_constructor_args(),
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    uploaded = upload_training_run(bundle=bundle, run_name=run_name)
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            name=run_name,
            launcher_args={"model": MODEL},
        )
    print(f"✓ Launched run_id={run_id}")
    print(f"  View / cancel at: {config.web_app_url()}/train/{run_id}")
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the castform loop for this env: data → validate → launch.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["data", "validate", "launch", "all"],
        help="Stage to run (default: all = data → validate, then STOP).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate datasets even if present."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the launch confirmation (it spends GPU credits).",
    )
    args = parser.parse_args(argv)

    from castform.platform import ensure_session

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        ensure_session()
        report = validate()
        ok = report is not None and report.ok
    if args.stage == "launch":
        ensure_session()
        ok = launch(assume_yes=args.yes) is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
