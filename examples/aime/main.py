"""AIME Harbor environment using the offline-installed Mini-SWE agent.

`python main.py [data|validate|launch|all]` drives the loop. The dataset
(aime/aime@latest) resolves through Harbor at trainer runtime, so the data
stage has nothing to download. Validation runs two real Modal sandbox trials.
Launch uploads the bundle and starts a GPU run (explicit, confirmed — it
spends credits). Modal credentials come from ~/.modal.toml.

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


from pathlib import Path
from typing import Any

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

from aime_agent import MINI_SWE_AGENT_VERSION

_AGENT_SOURCE = BundledAgentSource.from_directory(
    Path(__file__).parent,
    files=(
        "aime_agent.py",
        "mini_swe_probe.py",
        "castform_model.py",
        "run_mini_castform.py",
    ),
)


class AimeMiniSweHarborEnv(HarborEnv):
    """AIME latest on Modal, solved by the offline-installed Mini-SWE agent."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        max_agent_timeout_secs: float | None = None,
    ) -> None:
        super().__init__(
            dataset=DatasetConfig(name="aime/aime", ref="latest"),
            reward_keys=("reward", "partial_credit"),
            eval_ratio=0.1,
            trial=HarborTrialTemplate(
                agent=BundledHarborAgent(
                    config=TrialAgentConfig(
                        import_path="aime_agent:UpstreamMiniSweAgent",
                        kwargs={"version": MINI_SWE_AGENT_VERSION},
                        max_timeout_sec=max_agent_timeout_secs,
                    ),
                    source=_AGENT_SOURCE,
                ),
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
MODAL_PROFILE = os.environ.get("MODAL_PROFILE", "castform")


def _constructor_args() -> dict[str, Any]:
    data = tomllib.loads((Path.home() / ".modal.toml").read_text())[MODAL_PROFILE]
    return {
        "sandbox_credentials": ModalCredentials(
            token_id=data["token_id"], token_secret=data["token_secret"]
        ),
    }


def generate_data(*, force: bool) -> None:
    del force
    print(
        "data: aime/aime@latest resolves through Harbor at runtime — nothing to download"
    )


def validate() -> Any:
    from castform import validate_environment

    env = AimeMiniSweHarborEnv(**_constructor_args())
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

    run_name = f"aime-{uuid.uuid4().hex[:8]}"
    if not assume_yes:
        reply = input(
            f"Launch {run_name!r} on GPUs — this spends credits. Continue? [y/N] "
        )
        if reply.strip().lower() not in ("y", "yes"):
            print("Launch aborted.")
            return None

    # Bundle-only upload: Harbor resolves the dataset at trainer runtime.
    bundle = dump_bundle(
        AimeMiniSweHarborEnv,
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
