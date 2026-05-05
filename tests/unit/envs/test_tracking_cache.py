"""Test LRU bounds on the tracker cache (L4)."""

from __future__ import annotations

import pytest

from benchmax.envs import tracking
from benchmax.envs.tracking import TrackingConfig


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset cache between tests so they don't leak state."""
    tracking._TRACKER_CACHE.clear()
    yield
    tracking._TRACKER_CACHE.clear()


def test_tracker_cache_is_bounded(monkeypatch):
    """L4 regression: cache must evict old entries past the max."""
    # Stub the builder to avoid pulling in job_telemetry; we only care about
    # cache mechanics here.
    monkeypatch.setattr(tracking, "_build_tracker", lambda config: object())
    monkeypatch.setattr(tracking, "_TRACKER_CACHE_MAX", 4)

    for i in range(10):
        tracking.get_tracker(TrackingConfig(run_id=f"run-{i}", api_key="k"))

    # Cache should never exceed the max size.
    assert len(tracking._TRACKER_CACHE) == 4
    # Most-recent run-9 should still be present; oldest run-0 evicted.
    assert ("run-9", "k") in tracking._TRACKER_CACHE
    assert ("run-0", "k") not in tracking._TRACKER_CACHE


def test_tracker_cache_lru_promotes_recent(monkeypatch):
    """Accessing an existing entry should mark it as recently used."""
    monkeypatch.setattr(tracking, "_build_tracker", lambda config: object())
    monkeypatch.setattr(tracking, "_TRACKER_CACHE_MAX", 3)

    # Fill the cache.
    for i in range(3):
        tracking.get_tracker(TrackingConfig(run_id=f"run-{i}", api_key="k"))

    # Touch run-0 — should move it to most-recent end.
    tracking.get_tracker(TrackingConfig(run_id="run-0", api_key="k"))

    # Add a new entry; run-1 (now oldest) should be evicted, not run-0.
    tracking.get_tracker(TrackingConfig(run_id="run-3", api_key="k"))

    assert ("run-0", "k") in tracking._TRACKER_CACHE
    assert ("run-1", "k") not in tracking._TRACKER_CACHE
    assert ("run-3", "k") in tracking._TRACKER_CACHE
