"""Harvey LAB Harbor environment using Harvey's native harness loop.

The pinned Harvey dataset resolves through Harbor at trainer runtime,
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
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
from benchmax.bundle import dump_bundle
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

_HARNESS_SOURCE = BundledAgentSource.from_directory(
    Path(__file__).parent / "harness",
    files=("autocompact.py", "harvey_agent.py", "harvey_runtime.py"),
)
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HARVEY_LAB_DATASET_REF = "sha256:5cbe32ed0ce44c7244191ff764bda7e54b9e6b106726a1cf438b1009080e1628"


def harvey_harness(
    *,
    max_timeout_secs: float | None = None,
    source: BundledAgentSource | None = None,
    env: Mapping[str, str] | None = None,
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
            env=dict(env or {}),
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
        trials_dir: str | Path = "/tmp/castform-harvey-harbor-trials",
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
            # Pin the 1,251-task snapshot so the content-hash-ordered 90/10
            # split is identical across collection and evaluation runs.
            dataset=DatasetConfig(name="harveyai/lab", ref=HARVEY_LAB_DATASET_REF),
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
                trials_dir=Path(trials_dir),
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
    harness_env: dict[str, str] = {}
    if getattr(args, "autocompact_mode", "off") != "off":
        harness_env = {
            "HARBOR_HARVEY_AUTOCOMPACT_MODE": args.autocompact_mode,
            "HARBOR_HARVEY_MAX_COMPACTIONS": str(args.max_compactions),
        }
    if getattr(args, "autocompact_mode", "off") == "judge":
        harness_env.update(
            {
                "HARBOR_HARVEY_COMPACTION_JUDGE_PROVIDER": args.judge_provider,
                "HARBOR_HARVEY_COMPACTION_JUDGE_MODEL": args.judge_model,
                "HARBOR_HARVEY_COMPACTION_JUDGE_API_KEY": args.judge_api_key,
            }
        )
        if args.judge_base_url:
            harness_env["HARBOR_HARVEY_COMPACTION_JUDGE_BASE_URL"] = args.judge_base_url
    constructor_args = {
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
        "harness": harvey_harness(source=_lab_source_bundle(), env=harness_env),
    }
    if getattr(args, "action", None) == "collect":
        constructor_args.update(
            {
                "max_concurrent_trials": args.max_concurrent_trials,
                "trials_dir": str(Path(args.output_dir).expanduser().resolve() / "trials"),
            }
        )
    return constructor_args


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
                # instead of the full 30-turn loop; 4096 was measured too
                # small for even the first model call.
                max_context_tokens=6144,
                # Modal sandbox build plus a several-turn trial still exceeds
                # the 120s local default.
                local_timeout_seconds=1800,
            )
        )
    _print_validation(report)
    return report


def launch(
    uploaded_assets: Any,
    *,
    assume_yes: bool,
    training_mode: Literal["rl", "sft"] = "rl",
) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    is_sft = training_mode == "sft"
    stage = "launch-sft" if is_sft else "launch"
    if not assume_yes:
        kind = "AutoCompact SFT" if is_sft else "training"
        reply = input(f"launch {kind} on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print(f"{stage}: cancelled")
            return None

    launcher_args = dict(TRAINING_ARGS)
    name = RUN_NAME
    if is_sft:
        name = f"{RUN_NAME}-autocompact-sft"
        launcher_args.update(training_mode="sft", group_size=1)
    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=name,
            launcher_args=launcher_args,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"{stage}: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def collect(args: argparse.Namespace, env: HarveyLabHarborEnv) -> dict[str, Any] | None:
    from castform import config
    from castform.model_auth import model_auth_for_endpoint
    from collection import collect_trajectories

    if not args.yes:
        reply = input("collect judge-guided Modal trajectories? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("collect: cancelled")
            return None
    base_url = args.model_base_url or config.llm_url()
    model_auth = model_auth_for_endpoint(
        api_key=args.model_api_key or "",
        base_url=base_url,
        purpose="Harvey AutoCompact collection",
    )
    manifest = asyncio.run(
        collect_trajectories(
            env,
            output_dir=Path(args.output_dir),
            model=args.model,
            base_url=base_url,
            model_auth=model_auth,
            max_examples=args.max_examples,
            rollouts_per_task=args.rollouts_per_task,
            max_concurrent_tasks=args.max_concurrent_trials,
            max_compactions=args.max_compactions,
            resume=args.resume,
            dataset_ref=HARVEY_LAB_DATASET_REF,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


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
        choices=("data", "validate", "launch", "collect", "launch-sft"),
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
    parser.add_argument("--output-dir", default="harvey-autocompact-data")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-api-key")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--rollouts-per-task", type=int, default=1)
    parser.add_argument("--max-concurrent-trials", type=int, default=4)
    parser.add_argument("--max-compactions", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--autocompact-mode",
        choices=("off", "judge", "autonomous"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    args.autocompact_mode = args.autocompact_mode or (
        "judge" if args.action == "collect" else "off"
    )
    for name in ("rollouts_per_task", "max_concurrent_trials", "max_compactions"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.judge_base_url and args.judge_provider != "openai":
        parser.error("--judge-base-url is only supported with --judge-provider openai")
    total_stages = {
        "data": 1,
        "validate": 4,
        "launch": 5,
        "collect": 2,
        "launch-sft": 4,
    }[args.action]

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
    if args.action == "collect":
        print("[stage 2/2] collecting judge-guided train trajectories")
        collect(args, HarveyLabHarborEnv(**constructor_args))
        return 0
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        HarveyLabHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment")
    upload_kwargs: dict[str, Any] = {
        "bundle": bundled_environment,
        "run_name": RUN_NAME,
    }
    if args.action == "launch-sft":
        output_dir = Path(args.output_dir).expanduser().resolve()
        train_path = output_dir / "train.jsonl"
        eval_path = output_dir / "eval.jsonl"
        if not train_path.is_file() or not eval_path.is_file():
            parser.error("launch-sft requires <output-dir>/train.jsonl and eval.jsonl")
        upload_kwargs["dataset_files"] = {
            "train.jsonl": train_path,
            "eval.jsonl": eval_path,
        }
    uploaded_assets = upload_assets(**upload_kwargs)
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    if args.action == "launch-sft":
        print("[stage 4/4] launching SFT")
        launch(uploaded_assets, assume_yes=args.yes, training_mode="sft")
        return 0
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
