"""Integration smoke for per-request credential resolution (device-auth Phase 1).

Constructs a ``RolloutClient`` with **no** control-plane ``api_key``, so the
platform-service bearer resolves via the ``PLATFORM_API_KEY`` seam. The rollout
model receives a separate, explicitly provided ``CASTFORM_API_KEY``.

Hits staging. Requires ``PLATFORM_API_KEY`` and ``CASTFORM_API_KEY`` (from env /
``.env.test``); targets ``castform.dev`` unless ``CASTFORM_BASE_DOMAIN`` is
already set.

Run: uv run pytest tests/integration/platform/test_seam_smoke.py -v
"""

import os

import pytest

from benchmax.bundle import dump_bundle
from benchmax.envs import BaseEnv, Example, JsonlDataset, canonical_example_id
from castform.platform.client import RolloutClient
from castform.platform.exceptions import RolloutError

pytestmark = pytest.mark.integration

_PLATFORM_API_KEY = os.environ.get("PLATFORM_API_KEY", "")
_MODEL_API_KEY = os.environ.get("CASTFORM_API_KEY", "")
_AUTH_MARKERS = ("401", "403", "authentication", "Authentication", "Unauthorized")


def _make_echo_env():
    """Minimal no-tool env (one LLM turn, constant reward), defined in a local
    scope so cloudpickle pickles it by value — no local-module ref for
    dump_bundle to reject (mirrors the unit-test smoke env)."""

    class _EchoEnv(BaseEnv):
        max_turns = 1

        async def create_dataset(self, split, base_dir):
            def make_example(row):
                payload = {
                    "prompt_messages": [
                        {"role": "system", "content": "Reply with one short word."},
                        {"role": "user", "content": str(row["prompt"])},
                    ]
                }
                return Example(id=canonical_example_id(payload), payload=payload)

            return JsonlDataset(
                base_dir / f"{split}.jsonl", row_to_example=make_example
            )

        async def compute_reward(self, *a, **k):
            return {"reward": 1.0}

    return _EchoEnv


@pytest.mark.skipif(
    not (_PLATFORM_API_KEY and _MODEL_API_KEY),
    reason="PLATFORM_API_KEY and CASTFORM_API_KEY are required",
)
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

    bundle = dump_bundle(_make_echo_env())
    client = RolloutClient()  # no api_key → platform bearer resolves via the seam
    outcomes: list[dict] = []
    errors: list[str] = []
    for index, example in enumerate(({"prompt": "hi"}, {"prompt": "yo"})):
        try:
            outcomes.append(
                client.stream_rollout(
                    raw_example=example,
                    env_cls_bytes=bundle.pickled,
                    env_metadata_bytes=bundle.metadata.to_json_bytes(),
                    example_index=index,
                    llm_api_key=_MODEL_API_KEY,
                )
            )
        except (RolloutError, RuntimeError) as exc:
            errors.append(str(exc))

    auth_failures = [
        error for error in errors if any(marker in error for marker in _AUTH_MARKERS)
    ]
    assert not auth_failures, f"seam-resolved bearer rejected: {auth_failures}"
    assert any(outcome.get("success") for outcome in outcomes), errors
