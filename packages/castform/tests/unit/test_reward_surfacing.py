"""Slice 1.4 SDK plumbing: reward values now ride on ExampleValidation.

Offline — exercises the pure helpers + the group-event mean. The end-to-end
"real rollout produces these numbers" check is the staging gate.
"""

from __future__ import annotations

from castform.platform.client import ExampleValidation, RolloutClient, _mean_rewards


def test_mean_rewards_basic():
    assert _mean_rewards([{"a": 1.0}, {"a": 0.0}]) == {"a": 0.5}


def test_mean_rewards_per_key_counts():
    # 'b' only appears once — mean over the rollouts that have it.
    assert _mean_rewards([{"a": 1.0, "b": 4.0}, {"a": 3.0}]) == {"a": 2.0, "b": 4.0}


def test_mean_rewards_excludes_bools_and_nondicts():
    assert _mean_rewards([{"ok": True}, "junk", None, {"x": 2}]) == {"x": 2.0}


def test_mean_rewards_none_when_empty():
    assert _mean_rewards([]) is None
    assert _mean_rewards([{}, {"flag": True}]) is None


def test_example_validation_carries_rewards():
    ev = ExampleValidation(index=0, ok=True, rewards={"r": 1.0})
    assert ev.rewards == {"r": 1.0}
    # default stays None so existing index/ok/error constructors are unaffected
    assert ExampleValidation(index=-1, ok=False, error="x").rewards is None


def test_assess_group_events_returns_mean_on_success():
    client = RolloutClient()
    events = [
        {"success": True, "rewards": {"q": 1.0}},
        {"success": True, "rewards": {"q": 0.0}},
    ]
    ev = client._assess_group_events(events, samples=2, verbose=False)
    assert ev.ok is True
    assert ev.index == -1
    assert ev.rewards == {"q": 0.5}


def test_assess_group_events_error_still_fails_without_rewards():
    client = RolloutClient()
    events = [{"success": True, "group_reward_error": "judge key invalid"}]
    ev = client._assess_group_events(events, samples=1, verbose=False)
    assert ev.ok is False
    assert "judge key invalid" in (ev.error or "")
    assert ev.rewards is None
