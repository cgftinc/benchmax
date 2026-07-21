"""AIME Harbor environment using the offline-installed Mini-SWE agent."""

from __future__ import annotations

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
