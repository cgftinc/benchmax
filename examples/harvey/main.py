"""Harvey LAB Harbor environment using Harvey's native harness loop.

The dataset (harveyai/lab@latest) resolves through Harbor at trainer runtime,
so only the environment bundle is uploaded. Validation runs real Modal sandbox
trials. All credentials are mandatory CLI arguments
(--modal-token-id / --modal-token-secret / --judge-api-key); they are bundled
into the environment constructor args so trainer-side trials can reach Modal
and the judge.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
from benchmax.envs.environment import Environment
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

from benchmax.bundle import dump_bundle

_HARNESS_SOURCE = BundledAgentSource.from_directory(
    Path(__file__).parent / "harness",
    files=("harvey_agent.py", "harvey_runtime.py"),
)
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def harvey_harness(
    *,
    max_timeout_secs: float | None = None,
    source: BundledAgentSource | None = None,
) -> BundledHarborAgent:
    """The default harness: Harvey's native agent loop bundled from ``harness/``.

    With the two-file default source, the agent sparse-clones Harvey's LAB tree
    on the trial host; pass ``source=_lab_source_bundle()`` to ship the tree in
    the bundle instead. This function must not reference harness.harvey_agent:
    the env class points at it, so any such reference would drag the module
    (and its thread lock) into the bundle pickle.
    """

    return BundledHarborAgent(
        config=TrialAgentConfig(
            import_path="harvey_agent:HarveyHarnessAgent",
            max_timeout_sec=max_timeout_secs,
        ),
        source=source or _HARNESS_SOURCE,
    )


def _lab_source_bundle() -> BundledAgentSource:
    """Clone Harvey's LAB tree here and capture it alongside the harness files.

    Trial hosts (the hosted validation sandbox, the trainer) then need neither
    git nor GitHub egress.
    """

    from harness.harvey_agent import IGNORED_UPLOAD_NAMES, fetch_harvey_source

    lab_root = fetch_harvey_source()
    files = dict(_HARNESS_SOURCE.files)
    for path in sorted(lab_root.rglob("*")):
        relative = path.relative_to(lab_root)
        if not path.is_file() or IGNORED_UPLOAD_NAMES.intersection(relative.parts):
            continue
        files[f"harvey-labs/{relative.as_posix()}"] = path.read_bytes()
    return BundledAgentSource.from_files(files)


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


def _verifier_env_for_provider(
    provider: JudgeProvider,
    *,
    api_key: str,
    base_url: str | None = None,
) -> dict[str, str]:
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("judge api_key must be a non-empty string")

    if provider == "anthropic":
        if base_url:
            raise ValueError("base_url is only supported with the openai judge provider")
        return {"ANTHROPIC_API_KEY": api_key}

    if provider == "openai":
        environment = {
            "OPENAI_API_KEY": api_key,
            # harveyai/lab declares this variable even when RewardKit is
            # explicitly overridden to use an OpenAI-compatible judge.
            "ANTHROPIC_API_KEY": "unused-for-openai-judge",
        }
        if base_url:
            environment["OPENAI_BASE_URL"] = base_url
            environment["OPENAI_API_BASE"] = base_url
        return environment

    raise ValueError(f"unsupported judge provider: {provider}")


class HarveyLabHarborEnv(HarborEnv):
    """Harvey's latest LAB dataset on Modal; the agent harness defaults to Harvey's own."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        verifier_env: Mapping[str, str],
        judge_model: str,
        judge_concurrency: int = 1,
        harness: BundledHarborAgent | None = None,
        max_agent_timeout_secs: float | None = None,
        max_concurrent_trials: int | None = 1000,
        eval_ratio: float = 0.1,
        modal_app_name: str | None = None,
        sandbox_timeout_secs: int | None = None,
        sandbox_idle_timeout_secs: int | None = None,
    ) -> None:
        if harness is not None and max_agent_timeout_secs is not None:
            raise ValueError(
                "max_agent_timeout_secs applies only to the default harness; "
                "set max_timeout_sec on the custom harness config instead"
            )
        if not isinstance(judge_model, str) or not judge_model:
            raise ValueError("judge_model must be a non-empty string")
        validated_verifier_env = _validated_verifier_env(verifier_env)
        if judge_concurrency < 1:
            raise ValueError("judge_concurrency must be positive")
        if modal_app_name is not None and not modal_app_name.strip():
            raise ValueError("modal_app_name must be non-empty when provided")
        for name, value in (
            ("sandbox_timeout_secs", sandbox_timeout_secs),
            ("sandbox_idle_timeout_secs", sandbox_idle_timeout_secs),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when provided")

        verifier_env = {
            **validated_verifier_env,
            "REWARDKIT_JUDGE": judge_model,
            "JUDGE_CONCURRENCY": str(judge_concurrency),
        }
        environment_kwargs = {
            key: value
            for key, value in {
                "app_name": modal_app_name,
                "sandbox_timeout_secs": sandbox_timeout_secs,
                "sandbox_idle_timeout_secs": sandbox_idle_timeout_secs,
            }.items()
            if value is not None
        }
        super().__init__(
            dataset=DatasetConfig(name="harveyai/lab", ref="latest"),
            eval_ratio=eval_ratio,
            trial=HarborTrialTemplate(
                agent=harness
                if harness is not None
                else harvey_harness(max_timeout_secs=max_agent_timeout_secs),
                environment=TrialEnvironmentConfig(
                    type=EnvironmentType.MODAL,
                    kwargs=environment_kwargs,
                ),
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
RUN_NAME = "harvey"
TRAINING_ARGS = {"model": MODEL}


def _constructor_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sandbox_credentials": ModalCredentials(
            token_id=args.modal_token_id, token_secret=args.modal_token_secret
        ),
        "verifier_env": _verifier_env_for_provider(
            args.judge_provider,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
        ),
        "judge_model": args.judge_model,
        "judge_concurrency": args.judge_concurrency,
        # Hermetic harness: the LAB tree rides in the bundle instead of being
        # cloned on the trial host.
        "harness": harvey_harness(source=_lab_source_bundle()),
    }


def _check_judge_credentials(args: argparse.Namespace) -> None:
    """Fail fast on a bad judge key before any Modal trial spends money."""

    model = args.judge_model.removeprefix(f"{args.judge_provider}/")
    if args.judge_provider == "anthropic":
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": args.judge_api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=30,
        )
    else:
        base_url = (args.judge_base_url or "https://api.openai.com/v1").rstrip("/")
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {args.judge_api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_completion_tokens": 16,
            },
            timeout=60,
        )
    if response.status_code in (401, 403):
        raise SystemExit(
            f"judge preflight: API key rejected (HTTP {response.status_code}) — "
            f"check --judge-api-key: {response.text[:200]}"
        )
    if response.is_error:
        raise SystemExit(
            f"judge preflight: {args.judge_provider} judge call failed "
            f"(HTTP {response.status_code}): {response.text[:200]}"
        )
    print("judge preflight: credentials accepted")


def generate_data(*, force: bool) -> None:
    del force
    print("data: harveyai/lab@latest resolves through Harbor at runtime — nothing to download")


def validate(env: HarveyLabHarborEnv, uploaded_assets: Any) -> Any:
    from castform import validate_environment

    print("validate: running real modal sandbox trials — this stage takes a few minutes")
    with tempfile.TemporaryDirectory() as tmp:
        report = asyncio.run(
            validate_environment(
                env,
                model=VALIDATE_MODEL,
                split="eval",
                base_dir=Path(tmp),
                remote_assets=uploaded_assets,
                # Small enough that the budget stop ends trials in minutes
                # instead of the full 30-turn loop; 6144 was measured too
                # small for the harness prompt plus its first useful turn.
                max_context_tokens=8192,
                # Modal sandbox build plus a several-turn trial still exceeds
                # the 120s local default.
                local_timeout_seconds=1800,
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
    parser.add_argument(
        "--judge-provider",
        choices=["anthropic", "openai"],
        required=True,
        help="Credential convention used by the verifier.",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="RewardKit/LiteLLM model name used by the verifier.",
    )
    parser.add_argument(
        "--judge-api-key",
        required=True,
        help="Judge API key for --judge-provider (bundled into constructor args).",
    )
    parser.add_argument(
        "--judge-base-url",
        help="OpenAI-compatible judge base URL (only with --judge-provider openai).",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent judge calls (default: 1).",
    )
    args = parser.parse_args(argv)
    if args.judge_base_url and args.judge_provider != "openai":
        parser.error("--judge-base-url is only supported with --judge-provider openai")
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    # The data stage never talks to the judge, so it skips the preflight.
    if args.action != "data":
        _check_judge_credentials(args)

    print(f"[stage 1/{total_stages}] generating data")
    generate_data(force=args.force)
    if args.action == "data":
        return 0

    # Built after the data early-return: harvey's harness capture clones the
    # LAB tree, which the data stage must not pay for.
    constructor_args = _constructor_args(args)
    ensure_session()
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment")
    uploaded_assets = upload_assets(bundle=bundled_environment, run_name=RUN_NAME)
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(HarveyLabHarborEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
