"""Harvey LAB Harbor environment using Harvey's native harness loop."""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        if not isinstance(judge_api_key, str) or not judge_api_key:
            raise ValueError("judge_api_key must be a non-empty string")
        if judge_concurrency < 1:
            raise ValueError("judge_concurrency must be positive")

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
            eval_ratio=0.1,
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
