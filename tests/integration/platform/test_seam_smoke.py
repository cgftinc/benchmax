"""Integration smoke for per-request credential resolution (device-auth Phase 1).

Constructs a ``RolloutClient`` with **no** ``api_key``, so the platform-service
bearer — and the rollout's own platform-LLM leg key — both resolve via the
credential seam (``PLATFORM_API_KEY``). This exercises the path mocks can't:
platform-service validates the seam-resolved key, mints the act_as JWT, reaches
rollout-service, and the rollout's LLM completion succeeds.

Hits staging. Requires ``PLATFORM_API_KEY`` (from env / ``.env.test``); targets
``castform.dev`` unless ``CASTFORM_BASE_DOMAIN`` is already set.

Run: uv run pytest tests/integration/platform/test_seam_smoke.py -v
"""

import os

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.platform.client import RolloutClient

pytestmark = pytest.mark.integration

_API_KEY = os.environ.get("PLATFORM_API_KEY", "")
_AUTH_MARKERS = ("401", "403", "authentication", "Authentication", "Unauthorized")


def _make_echo_env():
    """Minimal no-tool env (one LLM turn, constant reward), defined in a local
    scope so cloudpickle pickles it by value — no local-module ref for
    dump_bundle to reject (mirrors the unit-test smoke env)."""

    class _EchoEnv(BaseEnv):
        system_prompt = "Reply with one short word."

        async def list_tools(self):
            return []

        async def run_tool(self, *a, **k):
            raise NotImplementedError

        async def compute_reward(self, *a, **k):
            return {"reward": 1.0}

    return _EchoEnv


@pytest.mark.skipif(not _API_KEY, reason="PLATFORM_API_KEY not set (the seam source)")
def test_rollout_client_resolves_bearer_via_seam(monkeypatch):
    """A client built with no api_key authenticates and rolls out via the seam.

    Two assertions, split so an infra flake can't masquerade as an auth regression:
      1. No example fails with an auth error — a broken seam bearer is rejected
         by platform-service (401/403) *before* any rollout starts, so this is
         the deterministic guard for the credential change.
      2. At least one rollout completes — proves the platform-LLM leg received a
         real key (not an empty bearer; the Phase 1 LLM-leg fix). Tolerates the
         occasional transient sandbox-runtime worker flake on a single example.
    """
    monkeypatch.setenv(
        "CASTFORM_BASE_DOMAIN", os.environ.get("CASTFORM_BASE_DOMAIN", "castform.dev")
    )

    client = RolloutClient()  # no api_key → bearer + LLM-leg key resolve via the seam
    result = client.validate_examples(
        [{"prompt": "hi"}, {"prompt": "yo"}],
        env_class=_make_echo_env(),
        n=2,
        verbose=True,
    )

    auth_failures = [
        e.error
        for e in result.examples
        if not e.ok and e.error and any(m in e.error for m in _AUTH_MARKERS)
    ]
    assert not auth_failures, f"seam-resolved bearer rejected: {auth_failures}"
    assert any(e.ok for e in result.examples), [e.error for e in result.examples]
