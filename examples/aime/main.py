"""AIME Harbor environment using the offline-installed Mini-SWE agent.

The dataset (aime/aime@latest) resolves through Harbor at trainer runtime, so
only the environment bundle is uploaded. Validation runs real Modal or
Cloudflare sandbox trials. Provider credentials ride in the environment bundle
so trainer-side trials can reach the selected sandbox service.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import cloudflare_environment as cloudflare_adapter
import cloudflare_transport
from benchmax.envs.environment import Environment
from benchmax.envs.harbor import (
    BundledAgentSource,
    BundledHarborAgent,
    CustomSandboxCredentials,
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
    SandboxCredentials,
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

from benchmax.bundle import dump_bundle

_CLOUDFLARE_LOCAL_MODULES = (cloudflare_transport, cloudflare_adapter)

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


SandboxProvider = Literal["modal", "cloudflare"]


def _prepare_cloudflare_modules_for_ray() -> None:
    """Carry the example-local adapter through Trainer's second Ray pickle."""

    for module in _CLOUDFLARE_LOCAL_MODULES:
        if not isinstance(module, ModuleType):
            raise TypeError("Cloudflare local module capture received a non-module")
        sys.modules[module.__name__] = module
    try:
        from ray import cloudpickle as ray_cloudpickle
    except ImportError:
        return
    for module in _CLOUDFLARE_LOCAL_MODULES:
        ray_cloudpickle.register_pickle_by_value(module)


class AimeMiniSweHarborEnv(HarborEnv):
    """AIME latest on Modal or Cloudflare with the bundled Mini-SWE loop."""

    def __init__(
        self,
        *,
        sandbox_credentials: SandboxCredentials,
        sandbox_provider: SandboxProvider = "modal",
        harness: BundledHarborAgent | None = None,
        max_agent_timeout_secs: float | None = None,
        max_concurrent_trials: int | None = 128,
    ) -> None:
        if harness is not None and max_agent_timeout_secs is not None:
            raise ValueError(
                "max_agent_timeout_secs applies only to the default harness; "
                "set max_timeout_sec on the custom harness config instead"
            )
        if sandbox_provider == "modal":
            if not isinstance(sandbox_credentials, ModalCredentials):
                raise TypeError("Modal AIME requires ModalCredentials")
            environment = TrialEnvironmentConfig(type=EnvironmentType.MODAL)
        elif sandbox_provider == "cloudflare":
            if sandbox_credentials.provider != "cloudflare":
                raise ValueError("Cloudflare AIME requires cloudflare credentials")
            _prepare_cloudflare_modules_for_ray()
            environment = TrialEnvironmentConfig(
                import_path="cloudflare_environment:AimeCloudflareEnvironment"
            )
        else:
            raise ValueError(f"unsupported sandbox_provider: {sandbox_provider}")
        super().__init__(
            dataset=DatasetConfig(name="aime/aime", ref="latest"),
            eval_ratio=0.1,
            trial=HarborTrialTemplate(
                agent=harness
                if harness is not None
                else mini_swe_harness(max_timeout_secs=max_agent_timeout_secs),
                environment=environment,
                verifier=TrialVerifierConfig(),
                trials_dir=Path("/tmp/castform-aime-harbor-trials"),
            ),
            sandbox_credentials=sandbox_credentials,
            max_concurrent_trials=max_concurrent_trials,
        )


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3.6-35B-A3B"
VALIDATE_MODEL = "gpt-5.4-mini"
TRAINING_ARGS = {"model": MODEL, "num_epochs": 1}


def _constructor_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.sandbox_provider == "modal":
        if not args.modal_token_id or not args.modal_token_secret:
            raise SystemExit("Modal requires --modal-token-id and --modal-token-secret")
        credentials: SandboxCredentials = ModalCredentials(
            token_id=args.modal_token_id,
            token_secret=args.modal_token_secret,
        )
    else:
        api_url = args.cloudflare_sandbox_api_url or os.environ.get("CLOUDFLARE_SANDBOX_API_URL")
        api_key = args.cloudflare_sandbox_api_key or os.environ.get("CLOUDFLARE_SANDBOX_API_KEY")
        if not api_url or not api_key:
            raise SystemExit(
                "Cloudflare requires CLOUDFLARE_SANDBOX_API_URL and "
                "CLOUDFLARE_SANDBOX_API_KEY (environment variables or CLI flags)"
            )
        credentials = CustomSandboxCredentials(
            provider="cloudflare",
            values={
                "CLOUDFLARE_SANDBOX_API_URL": api_url.rstrip("/"),
                "CLOUDFLARE_SANDBOX_API_KEY": api_key,
            },
        )
    return {
        "sandbox_credentials": credentials,
        "sandbox_provider": args.sandbox_provider,
    }


def _runtime_dependencies(provider: SandboxProvider) -> list[str]:
    if provider == "modal":
        return ["harbor[modal]>=0.18.0,<0.19"]
    return ["harbor>=0.18.0,<0.19", "httpx>=0.27.0"]


def _local_modules(provider: SandboxProvider) -> list[ModuleType]:
    return list(_CLOUDFLARE_LOCAL_MODULES) if provider == "cloudflare" else []


def _preflight_cloudflare(constructor_args: dict[str, Any]) -> None:
    if constructor_args["sandbox_provider"] != "cloudflare":
        return
    import httpx

    values = constructor_args["sandbox_credentials"].host_environment()
    response = httpx.get(
        f"{values['CLOUDFLARE_SANDBOX_API_URL']}/v1/openapi.json",
        headers={"Authorization": f"Bearer {values['CLOUDFLARE_SANDBOX_API_KEY']}"},
        timeout=30,
    )
    if response.is_error:
        raise SystemExit(
            f"Cloudflare Sandbox preflight failed (HTTP {response.status_code}): "
            f"{response.text[:200]}"
        )
    print("Cloudflare Sandbox preflight: bridge credentials accepted")


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
                split="train",
                base_dir=Path(tmp),
                remote_assets=uploaded_assets,
            )
        )
    _print_validation(report)
    return report


def launch(uploaded_assets: Any, *, assume_yes: bool, run_name: str) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
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
        "--sandbox-provider",
        choices=("modal", "cloudflare"),
        default="modal",
    )
    parser.add_argument(
        "--modal-token-id",
        help="Modal token id for sandbox trials (bundled into constructor args).",
    )
    parser.add_argument(
        "--modal-token-secret",
        help="Modal token secret for sandbox trials (bundled into constructor args).",
    )
    parser.add_argument(
        "--cloudflare-sandbox-api-url",
        help="deployed AIME Sandbox bridge URL (defaults to environment)",
    )
    parser.add_argument(
        "--cloudflare-sandbox-api-key",
        help="AIME Sandbox bridge bearer key (defaults to environment)",
    )
    args = parser.parse_args(argv)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    generate_data(force=args.force)
    if args.action == "data":
        return 0

    # Built after the data early-return: harvey's harness capture clones the
    # LAB tree, which the data stage must not pay for.
    constructor_args = _constructor_args(args)
    _preflight_cloudflare(constructor_args)
    ensure_session()
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        AimeMiniSweHarborEnv,
        constructor_args=constructor_args,
        pip_dependencies=_runtime_dependencies(args.sandbox_provider),
        local_modules=_local_modules(args.sandbox_provider),
    )
    run_name = f"aime-{args.sandbox_provider}"
    print(f"[stage 3/{total_stages}] uploading environment")
    uploaded_assets = upload_assets(bundle=bundled_environment, run_name=run_name)
    print(f"  env_cls_path: {uploaded_assets.env_cls_path}")
    print(f"  env_metadata_path: {uploaded_assets.env_metadata_path}")
    print(f"  dataset_path: {uploaded_assets.dataset_path}")
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(AimeMiniSweHarborEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes, run_name=run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
