"""Harvey LAB Harbor environment using Harvey's native harness loop.

`python main.py [data|validate|launch|all]` drives the loop. The dataset
(harveyai/lab@latest) resolves through Harbor at trainer runtime, so the data
stage has nothing to download. Validation runs two real Modal sandbox trials.
Launch uploads the bundle and starts a GPU run (explicit, confirmed — it
spends credits). Credentials: Modal from MODAL_TOKEN_ID/MODAL_TOKEN_SECRET; the
verifier provider and model come from --judge-provider and --judge-model.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

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
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_VERIFIER_ENV = frozenset({"JUDGE_CONCURRENCY", "REWARDKIT_JUDGE"})


def _validated_verifier_env(verifier_env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(verifier_env, Mapping):
        raise TypeError("verifier_env must be a mapping")
    environment: dict[str, str] = {}
    for key, value in verifier_env.items():
        if not isinstance(key, str) or not _ENV_NAME_PATTERN.fullmatch(key):
            raise ValueError(f"invalid verifier environment variable name: {key!r}")
        if key in _RESERVED_VERIFIER_ENV:
            raise ValueError(f"{key} cannot be supplied through verifier_env")
        if not isinstance(value, str) or not value:
            raise ValueError(f"verifier environment variable {key!r} must be non-empty")
        environment[key] = value
    if not environment:
        raise ValueError("verifier_env must contain at least one value")
    return environment


JudgeProvider = Literal["anthropic", "openai"]


def _verifier_env_for_provider(provider: JudgeProvider) -> dict[str, str]:
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("set ANTHROPIC_API_KEY for the Anthropic judge")
        return {"ANTHROPIC_API_KEY": api_key}

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("set OPENAI_API_KEY for the OpenAI-compatible judge")
        environment = {
            "OPENAI_API_KEY": api_key,
            # harveyai/lab declares this variable even when RewardKit is
            # explicitly overridden to use an OpenAI-compatible judge.
            "ANTHROPIC_API_KEY": "unused-for-openai-judge",
        }
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if value := os.environ.get(name):
                environment[name] = value
        return environment

    raise ValueError(f"unsupported judge provider: {provider}")


class HarveyLabHarborEnv(HarborEnv):
    """Harvey's latest LAB dataset on Modal with the native Harvey harness."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        verifier_env: Mapping[str, str],
        judge_model: str,
        judge_concurrency: int = 1,
        max_agent_timeout_secs: float | None = None,
        max_concurrent_trials: int | None = 1000,
        eval_ratio: float = 0.1,
    ) -> None:
        if not isinstance(judge_model, str) or not judge_model:
            raise ValueError("judge_model must be a non-empty string")
        validated_verifier_env = _validated_verifier_env(verifier_env)
        if judge_concurrency < 1:
            raise ValueError("judge_concurrency must be positive")
        if not 0 < eval_ratio < 1:
            raise ValueError("eval_ratio must be in (0, 1)")

        verifier_env = {
            **validated_verifier_env,
            "REWARDKIT_JUDGE": judge_model,
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


def _modal_credentials_from_process() -> ModalCredentials:
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    if not token_id or not token_secret:
        raise ValueError("set both MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    return ModalCredentials(token_id=token_id, token_secret=token_secret)


def _constructor_args(
    *,
    judge_provider: JudgeProvider,
    judge_model: str,
    judge_concurrency: int,
) -> dict[str, Any]:
    try:
        verifier_env = _verifier_env_for_provider(judge_provider)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    try:
        sandbox_credentials = _modal_credentials_from_process()
    except ValueError as error:
        raise SystemExit(f"could not load Modal credentials: {error}") from None
    return {
        "sandbox_credentials": sandbox_credentials,
        "verifier_env": verifier_env,
        "judge_model": judge_model,
        "judge_concurrency": judge_concurrency,
    }


def generate_data(*, force: bool) -> None:
    del force
    print(
        "data: harveyai/lab@latest resolves through Harbor at runtime — nothing to download"
    )


def validate(
    *,
    judge_provider: JudgeProvider,
    judge_model: str,
    judge_concurrency: int,
) -> Any:
    from castform import validate_environment

    env = HarveyLabHarborEnv(
        **_constructor_args(
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_concurrency=judge_concurrency,
        )
    )
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


def launch(
    *,
    assume_yes: bool,
    judge_provider: JudgeProvider,
    judge_model: str,
    judge_concurrency: int,
) -> str | None:
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
        constructor_args=_constructor_args(
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_concurrency=judge_concurrency,
        ),
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
    parser.add_argument(
        "--judge-provider",
        choices=["anthropic", "openai"],
        help="Credential convention used by the verifier.",
    )
    parser.add_argument(
        "--judge-model",
        help="RewardKit/LiteLLM model name used by the verifier.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent judge calls (default: 1).",
    )
    args = parser.parse_args(argv)
    if args.stage in ("validate", "launch", "all"):
        if not args.judge_provider:
            parser.error("--judge-provider is required for validate and launch")
        if not args.judge_model:
            parser.error("--judge-model is required for validate and launch")

    from castform.platform import ensure_session

    ok = True
    if args.stage in ("data", "all"):
        generate_data(force=args.force)
    if args.stage in ("validate", "all"):
        ensure_session()
        report = validate(
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_concurrency=args.judge_concurrency,
        )
        ok = report is not None and report.ok
    if args.stage == "launch":
        ensure_session()
        ok = (
            launch(
                assume_yes=args.yes,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                judge_concurrency=args.judge_concurrency,
            )
            is not None
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
