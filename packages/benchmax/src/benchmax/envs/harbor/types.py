from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmax.envs.harbor.bundled_agent import BundledHarborAgent
from benchmax.envs.harbor.runtime_verifier import RuntimeOnlyHarborVerifier

if TYPE_CHECKING:
    from harbor.models.task.config import ArtifactConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        VerifierConfig,
    )

__all__ = ["HarborTrialTemplate"]


@dataclass(frozen=True, slots=True, repr=False)
class HarborTrialTemplate:
    """Task-independent Harbor configuration shared by every rollout.

    Harbor's ``TrialConfig`` cannot itself be the template because it requires
    a resolved task, trial name, and output directory. ``HarborEnv`` supplies
    those per rollout and passes every field below through to ``TrialConfig``.
    """

    agent: AgentConfig | BundledHarborAgent
    environment: EnvironmentConfig
    verifier: VerifierConfig | RuntimeOnlyHarborVerifier
    trials_dir: Path = Path("trials")
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    verifier_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    artifacts: tuple[str | ArtifactConfig, ...] = ()
    extra_instruction_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.agent is None:
            raise ValueError("Harbor trial agent config is required")
        if self.environment is None:
            raise ValueError("Harbor trial environment config is required")
        if self.verifier is None:
            raise ValueError("Harbor trial verifier config is required")

        trials_dir = Path(self.trials_dir).expanduser()
        if not str(trials_dir):
            raise ValueError("Harbor trials_dir must be non-empty")
        object.__setattr__(self, "trials_dir", trials_dir)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(
            self,
            "extra_instruction_paths",
            tuple(Path(path).expanduser() for path in self.extra_instruction_paths),
        )

        multipliers = {
            "timeout_multiplier": self.timeout_multiplier,
            "agent_timeout_multiplier": self.agent_timeout_multiplier,
            "verifier_timeout_multiplier": self.verifier_timeout_multiplier,
            "agent_setup_timeout_multiplier": self.agent_setup_timeout_multiplier,
            "environment_build_timeout_multiplier": (
                self.environment_build_timeout_multiplier
            ),
        }
        for name, value in multipliers.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def __repr__(self) -> str:
        return (
            "HarborTrialTemplate("
            f"trials_dir={self.trials_dir!r}, "
            f"timeout_multiplier={self.timeout_multiplier!r}, "
            f"artifacts={len(self.artifacts)})"
        )
