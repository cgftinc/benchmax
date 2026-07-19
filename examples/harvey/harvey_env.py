"""Harvey LAB Harbor environment using Harvey's native harness loop."""

from __future__ import annotations

import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchmax.envs.harbor import HarborEnv, HarborTrialTemplate, ModalCredentials
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)

# Harbor imports custom agents by module path after the BenchMax environment has
# been unpickled. Capture both files by value and materialize them on the remote
# trainer so `import harvey_agent` and the probe upload work without this checkout.
_AGENT_SOURCE_NAMES = ("harvey_agent.py", "harvey_castform_probe.py")
_AGENT_SOURCES: dict[str, str] = {
    name: Path(__file__).with_name(name).read_text() for name in _AGENT_SOURCE_NAMES
}


def _ensure_agent_module() -> None:
    """Make the custom Harbor agent importable from a by-value bundle."""

    try:
        import harvey_agent  # noqa: F401

        return
    except ImportError:
        pass
    digest = sha256(
        "".join(_AGENT_SOURCES[name] for name in _AGENT_SOURCE_NAMES).encode()
    ).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"castform-harvey-agent-{digest}"
    target.mkdir(exist_ok=True)
    for name, source in _AGENT_SOURCES.items():
        path = target / name
        if not path.exists():
            path.write_text(source)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


class HarveyLabHarborEnv(HarborEnv):
    """Harvey's latest LAB dataset on Modal with the native Harvey harness."""

    PIP_DEPENDENCIES = ["harbor[modal]>=0.18.0,<0.19"]

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        judge_api_key: str,
        judge_model: str = "openai/gpt-5.4-nano",
        judge_base_url: str = "https://llm.castform.dev/v1",
        judge_concurrency: int = 2,
        max_agent_timeout_secs: float | None = None,
        max_concurrent_trials: int | None = 1000,
    ) -> None:
        _ensure_agent_module()
        super().__init__(
            **harvey_harbor_constructor_args(
                sandbox_credentials,
                judge_api_key=judge_api_key,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                judge_concurrency=judge_concurrency,
                max_agent_timeout_secs=max_agent_timeout_secs,
                max_concurrent_trials=max_concurrent_trials,
            )
        )


def harvey_harbor_constructor_args(
    sandbox_credentials: ModalCredentials,
    *,
    judge_api_key: str,
    judge_model: str = "openai/gpt-5.4-nano",
    judge_base_url: str = "https://llm.castform.dev/v1",
    judge_concurrency: int = 2,
    max_agent_timeout_secs: float | None = None,
    max_concurrent_trials: int | None = 1000,
) -> dict[str, Any]:
    """Return the current generic Harbor configuration for a portable bundle."""

    if not judge_api_key:
        raise ValueError("judge_api_key must be non-empty")
    if judge_concurrency < 1:
        raise ValueError("judge_concurrency must be positive")

    normalized_judge_base_url = judge_base_url.rstrip("/")
    verifier_env = {
        "REWARDKIT_JUDGE": judge_model,
        "OPENAI_API_KEY": judge_api_key,
        "OPENAI_BASE_URL": normalized_judge_base_url,
        "OPENAI_API_BASE": normalized_judge_base_url,
        # The published tasks declare this placeholder even when RewardKit is
        # overridden to an OpenAI-compatible judge.
        "ANTHROPIC_API_KEY": judge_api_key,
        "JUDGE_CONCURRENCY": str(judge_concurrency),
    }
    return {
        "dataset": DatasetConfig(name="harveyai/lab", ref="latest"),
        "eval_ratio": 0.1,
        "trial": HarborTrialTemplate(
            agent=TrialAgentConfig(
                import_path="harvey_agent:HarveyHarnessAgent",
                max_timeout_sec=max_agent_timeout_secs,
            ),
            environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
            verifier=TrialVerifierConfig(env=verifier_env),
            trials_dir=Path("/tmp/castform-harvey-harbor-trials"),
        ),
        "sandbox_credentials": sandbox_credentials,
        "max_concurrent_trials": max_concurrent_trials,
    }
