"""AIME Harbor environment using the offline-installed Mini-SWE agent."""

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

from aime_agent import MINI_SWE_AGENT_VERSION

# Agent module + the files it uploads into sandboxes, captured at import time.
# Bundles pickle this env by value; harbor later resolves the agent with a real
# `import aime_agent`, so on hosts without these files on disk the sources are
# materialized into a temp dir that joins sys.path (see _ensure_agent_module).
_AGENT_SOURCE_NAMES = (
    "aime_agent.py",
    "mini_swe_probe.py",
    "castform_model.py",
    "run_mini_castform.py",
)
_AGENT_SOURCES: dict[str, str] = {
    name: Path(__file__).with_name(name).read_text() for name in _AGENT_SOURCE_NAMES
}


def _ensure_agent_module() -> None:
    """Make `import aime_agent` work on hosts that only have this bundle."""

    try:
        import aime_agent  # noqa: F401

        return
    except ImportError:
        pass
    digest = sha256(
        "".join(_AGENT_SOURCES[name] for name in _AGENT_SOURCE_NAMES).encode()
    ).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"castform-aime-agent-{digest}"
    target.mkdir(exist_ok=True)
    for name, source in _AGENT_SOURCES.items():
        path = target / name
        if not path.exists():
            path.write_text(source)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


class AimeMiniSweHarborEnv(HarborEnv):
    """AIME latest on Modal, solved by the offline-installed Mini-SWE agent."""

    def __init__(
        self,
        *,
        sandbox_credentials: ModalCredentials,
        max_agent_timeout_secs: float | None = None,
    ) -> None:
        _ensure_agent_module()
        super().__init__(
            **aime_harbor_constructor_args(
                sandbox_credentials,
                max_agent_timeout_secs=max_agent_timeout_secs,
            )
        )


def aime_harbor_constructor_args(
    sandbox_credentials: ModalCredentials,
    *,
    max_agent_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Return the complete generic Harbor configuration for a portable bundle."""

    return {
        "dataset": DatasetConfig(name="aime/aime", ref="latest"),
        "eval_ratio": 0.1,
        "trial": HarborTrialTemplate(
            agent=TrialAgentConfig(
                import_path="aime_agent:UpstreamMiniSweAgent",
                kwargs={"version": MINI_SWE_AGENT_VERSION},
                max_timeout_sec=max_agent_timeout_secs,
            ),
            environment=TrialEnvironmentConfig(
                type=EnvironmentType.MODAL,
            ),
            verifier=TrialVerifierConfig(),
            trials_dir=Path("/tmp/castform-aime-harbor-trials"),
        ),
        "sandbox_credentials": sandbox_credentials,
        "max_concurrent_trials": 1000,
    }
