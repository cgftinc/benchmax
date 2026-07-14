from __future__ import annotations

import asyncio
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)

from benchmax.envs import RolloutRequest
from benchmax.envs.harbor import (
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_harbor_env_runs_miniswe_agent_in_modal(tmp_path: Path) -> None:
    """Exercise Benchmax -> Harbor -> Modal -> model -> verifier."""

    settings = _live_settings()
    source = tmp_path / "source"
    _write_live_task(source / "write-answer")
    env = HarborEnv(
        dataset=DatasetConfig(path=source),
        eval_ratio=0,
        trial=HarborTrialTemplate(
            agent=TrialAgentConfig(
                name="mini-swe-agent",
                kwargs={"max_tokens": 2048},
            ),
            environment=TrialEnvironmentConfig(
                type=EnvironmentType.MODAL,
                kwargs={
                    "app_name": "benchmax-harbor-e2e",
                    "sandbox_timeout_secs": 900,
                },
            ),
            verifier=TrialVerifierConfig(),
            trials_dir=tmp_path / "trials",
        ),
        sandbox_credentials=ModalCredentials(
            token_id=settings.modal_token_id,
            token_secret=settings.modal_token_secret,
        ),
    )
    dataset = await env.create_dataset("train", tmp_path / "dataset")
    assert len(dataset) == 1
    request = RolloutRequest(
        rollout_id="harbor-modal-live",
        example=dataset[0],
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
    )

    async with asyncio.timeout(900):
        outcomes = await env.run_group([request])

    assert outcomes[request.rollout_id].rewards == {"reward": 1.0}
    assert outcomes[request.rollout_id].termination_reason == "finished"


@dataclass(frozen=True, slots=True)
class _LiveSettings:
    modal_token_id: str
    modal_token_secret: str
    base_url: str
    api_key: str
    model: str


def _live_settings() -> _LiveSettings:
    """Load explicit live-test inputs without making production config implicit."""

    if os.environ.get("BENCHMAX_RUN_HARBOR_MODAL_LIVE") != "1":
        pytest.skip("set BENCHMAX_RUN_HARBOR_MODAL_LIVE=1 to run the live test")
    if importlib.util.find_spec("modal") is None:
        pytest.skip("install the Harbor Modal extra: harbor[modal]>=0.18,<0.19")

    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    api_key = os.environ.get("BENCHMAX_HARBOR_API_KEY") or os.environ.get(
        "CASTFORM_API_KEY"
    )
    missing = [
        name
        for name, value in (
            ("MODAL_TOKEN_ID", modal_token_id),
            ("MODAL_TOKEN_SECRET", modal_token_secret),
            ("BENCHMAX_HARBOR_API_KEY or CASTFORM_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        pytest.skip("missing live credentials: " + ", ".join(missing))
    assert modal_token_id is not None
    assert modal_token_secret is not None
    assert api_key is not None

    return _LiveSettings(
        modal_token_id=modal_token_id,
        modal_token_secret=modal_token_secret,
        base_url=os.environ.get(
            "BENCHMAX_HARBOR_BASE_URL",
            "https://llm.castform.dev/v1",
        ),
        api_key=api_key,
        model=os.environ.get("BENCHMAX_HARBOR_MODEL", "qwen3.5-4b"),
    )


def _write_live_task(task_dir: Path) -> None:
    """Create a tiny real Harbor task with an observable sandbox side effect."""

    task_dir.joinpath("environment").mkdir(parents=True)
    task_dir.joinpath("tests").mkdir()
    task_dir.joinpath("task.toml").write_text(
        """\
version = "1.0"

[metadata]
name = "benchmax/harbor-modal-live"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
gpus = 0
"""
    )
    task_dir.joinpath("instruction.md").write_text(
        "Use the terminal to create /tmp/benchmax-harbor-answer.txt. "
        "The file must contain exactly benchmax-harbor-ok followed by a newline."
    )
    task_dir.joinpath("environment", "Dockerfile").write_text("FROM ubuntu:24.04\n")
    verifier = task_dir.joinpath("tests", "test.sh")
    verifier.write_text(
        """\
#!/bin/bash
set -u
if [[ "$(cat /tmp/benchmax-harbor-answer.txt 2>/dev/null)" == "benchmax-harbor-ok" ]]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
    )
    verifier.chmod(0o755)
