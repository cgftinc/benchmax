"""Unit tests for the unified benchmax.platform.validate_env entry point.

Covers the wiring between the public function and the remote smoke pass
(RolloutClient.validate_examples) — the local contract checks themselves are
exercised end-to-end by tests/unit/rewards/test_diversity_env.py.
"""

from __future__ import annotations

from typing import Any

from benchmax.platform.client import ExampleValidation, ValidationResult
from benchmax.platform.validation import ValidationReport, validate_env


class _DummyEnv:
    """Placeholder env class — never instantiated in these tests (remote path
    is faked, local path is skipped via local=False)."""


def _install_fake_rollout(monkeypatch) -> dict[str, Any]:
    """Patch RolloutClient so validate_env's remote pass records its wiring
    instead of hitting the network. Returns a dict the test inspects."""
    seen: dict[str, Any] = {}

    class _FakeRolloutClient:
        def __init__(self, api_key, server_url=None, timeout=300.0):
            seen["api_key"] = api_key
            seen["server_url"] = server_url

        def validate_examples(self, examples, **kwargs):
            seen["examples"] = examples
            seen["kwargs"] = kwargs
            return ValidationResult(examples=[ExampleValidation(0, True)])

    # validate_env does `from .client import RolloutClient` lazily, so patch
    # the attribute on the client module it resolves against.
    monkeypatch.setattr("benchmax.platform.client.RolloutClient", _FakeRolloutClient)
    return seen


# ---------------------------------------------------------------------------
# Report semantics
# ---------------------------------------------------------------------------


def test_report_bool_cast():
    assert bool(ValidationReport(3, 0, None, True, False)) is True
    assert bool(ValidationReport(2, 1, None, True, False)) is False
    # Nothing ran → not a pass (fail loudly).
    assert bool(ValidationReport(0, 0, None, False, False)) is False


def test_report_remote_failure_fails_overall():
    remote = ValidationResult(examples=[ExampleValidation(0, False, "boom")])
    report = ValidationReport(5, 0, remote, True, True)
    assert report.local_ok is True
    assert report.remote_ok is False
    assert bool(report) is False


# ---------------------------------------------------------------------------
# Remote pass wiring
# ---------------------------------------------------------------------------


def test_local_false_runs_remote_via_seam(monkeypatch):
    """A keyless launch (local=False, no api_key) runs the remote smoke with
    RolloutClient(api_key=None), so it resolves through the credential seam /
    cached session. (Interactive login is the script's up-front ensure_session,
    not validate_env's job.)"""
    seen = _install_fake_rollout(monkeypatch)

    report = validate_env(
        env_class=_DummyEnv,
        env_args={},
        train_dataset=[{"prompt": "hi"}],
        local=False,
        verbose=False,
    )

    assert report.remote_ran is True
    assert report.remote_ok is True
    assert seen["api_key"] is None  # resolves via the seam, not an explicit key


def test_api_key_runs_remote_and_threads_urls(monkeypatch):
    """With an api_key, base_url maps to RolloutClient.server_url and the LLM
    URL/key + example count are forwarded to validate_examples."""
    seen = _install_fake_rollout(monkeypatch)

    report = validate_env(
        env_class=_DummyEnv,
        env_args={"foo": 1},
        train_dataset=[{"prompt": "a"}, {"prompt": "b"}],
        local=False,
        api_key="sk_test",
        base_url="https://api.castform.dev",
        llm_base_url="https://llm.castform.dev/v1",
        llm_api_key="llm-key",
        remote_examples=2,
        pip_dependencies=["openai"],
        verbose=False,
    )

    assert report.remote_ran is True
    assert report.remote_ok is True
    assert seen["api_key"] == "sk_test"
    assert seen["server_url"] == "https://api.castform.dev"
    kw = seen["kwargs"]
    assert kw["env_class"] is _DummyEnv
    assert kw["constructor_args"] == {"foo": 1}
    assert kw["llm_base_url"] == "https://llm.castform.dev/v1"
    assert kw["llm_api_key"] == "llm-key"
    assert kw["n"] == 2
    assert kw["pip_dependencies"] == ["openai"]


def test_local_only_skips_remote(monkeypatch):
    """Offline dev (local=True default, no api_key) runs no remote smoke."""
    seen = _install_fake_rollout(monkeypatch)
    # Skip the real local contract checks (they'd instantiate the placeholder env).
    monkeypatch.setattr(
        "benchmax.platform.validation._run_local_checks", lambda *a, **k: (1, 0)
    )
    monkeypatch.setattr(
        "benchmax.platform.validation._shutdown_shared_loop", lambda: None
    )

    report = validate_env(
        env_class=_DummyEnv,
        env_args={},
        train_dataset=[{"prompt": "hi"}],
        local=True,
        verbose=False,
    )

    assert report.remote_ran is False
    assert report.remote is None
    assert "api_key" not in seen  # RolloutClient never constructed
