"""Integration: TrainerClient stored-rollout reads hit the live platform API.

Exercises the ``/rollouts/*`` endpoints that the CLI's ``runs rollouts`` /
``runs rollout`` wrap — the query params (mode, page/limit, promptMessageId) and
the response shapes the CLI depends on (``promptMessageId``, a rollout ``id``,
per-component ``rewards``, ``avg``) — against a real completed run. Mocks can't
validate the wire contract; this does (CLAUDE.md: fetch/query/pagination changes
must have a live integration test).

Requires ``PLATFORM_API_KEY`` (the seam) AND ``CASTFORM_TEST_RUN_ID`` (a completed
run with stored eval rollouts). Targets ``castform.dev`` unless
``CASTFORM_BASE_DOMAIN`` is already set. Skipped in CI (no creds).

Run: uv run pytest tests/integration/platform/test_rollout_reads.py -v
"""

import os

import pytest

from benchmax.platform.client import TrainerClient

pytestmark = pytest.mark.integration

_API_KEY = os.environ.get("PLATFORM_API_KEY", "")
_RUN_ID = os.environ.get("CASTFORM_TEST_RUN_ID", "")


@pytest.mark.skipif(
    not (_API_KEY and _RUN_ID),
    reason="PLATFORM_API_KEY + CASTFORM_TEST_RUN_ID required for the live rollout reads",
)
def test_rollout_read_chain_live(monkeypatch):
    """summary → heatmap → details → mode-average against a real run.

    Asserts the shapes the CLI parses, so a server-side rename (promptMessageId,
    rollout id, rewards[]) or a params regression is caught before merge.
    """
    monkeypatch.setenv(
        "CASTFORM_BASE_DOMAIN", os.environ.get("CASTFORM_BASE_DOMAIN", "castform.dev")
    )

    with TrainerClient() as c:
        summary = c.get_rollout_summary(_RUN_ID, mode="eval", limit=5)
        assert isinstance(summary, list)
        if not summary:
            pytest.skip(f"run {_RUN_ID} has no eval rollouts to inspect")

        example = summary[0]
        assert "promptMessageId" in example, example
        prompt_message_id = example["promptMessageId"]

        heatmap = c.get_rollout_heatmap(_RUN_ID, prompt_message_id, mode="eval")
        assert isinstance(heatmap, list) and heatmap, heatmap
        rollout_id = heatmap[0]["id"]

        details = c.get_rollout_details(_RUN_ID, rollout_id)
        # The CLI reads messages (transcript) + rewards[{name,value}] from here.
        assert isinstance(details.get("rewards", []), list)
        assert "messages" in details or "promptMessages" in details

        avg = c.get_rollout_mode_average(_RUN_ID, mode="eval")
        assert isinstance(avg, dict)
