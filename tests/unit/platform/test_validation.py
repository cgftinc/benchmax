"""Unit tests for the unified benchmax.platform.validate_env entry point.

Covers the wiring between the public function and the remote smoke pass
(RolloutClient.validate_examples) — the local contract checks themselves are
exercised end-to-end by tests/unit/rewards/test_diversity_env.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmax.envs import (
    BaseEnv,
    Example,
    JsonlDataset,
    canonical_example_id,
)
from benchmax.platform.client import ExampleValidation, ValidationResult
from benchmax.platform.validation import (
    ValidationReport,
    assert_group_reward_contract,
    overrides_compute_group_reward,
    validate_env,
)


class _DummyEnv:
    """Placeholder env class — never instantiated in these tests (remote path
    is faked, local path is skipped via local=False)."""


# ---------------------------------------------------------------------------
# Concrete envs for the local-layer group-reward checks. Module-level so
# cloudpickle round-trips them by value (tests pass local_modules=[this]).
# ---------------------------------------------------------------------------


class _PlainEnv(BaseEnv):
    """Passes every local check; does NOT override compute_group_reward."""

    max_turns = 1

    async def create_dataset(self, split, base_dir):
        def make_example(row):
            payload = {
                "prompt_messages": [
                    {"role": "user", "content": str(row.get("prompt", ""))}
                ],
                **{key: value for key, value in row.items() if key != "prompt"},
            }
            return Example(id=canonical_example_id(payload), payload=payload)

        return JsonlDataset(base_dir / f"{split}.jsonl", row_to_example=make_example)

    async def compute_reward(
        self, rollout_id, messages, example_args, *, termination_reason
    ):
        return {"r": 1.0}


class _GoodGroupEnv(_PlainEnv):
    async def compute_group_reward(
        self, rollout_ids, messages_list, example_args_list, termination_reasons
    ):
        return [{"r": 1.0} for _ in rollout_ids]


class _NotListGroupEnv(_PlainEnv):
    async def compute_group_reward(
        self, rollout_ids, messages_list, example_args_list, termination_reasons
    ):
        return {"r": 1.0}  # not a list[dict]


class _ShortGroupEnv(_PlainEnv):
    async def compute_group_reward(
        self, rollout_ids, messages_list, example_args_list, termination_reasons
    ):
        return [{"r": 1.0}]  # always length 1 → breaks 1:1 pairing


class _NonFiniteGroupEnv(_PlainEnv):
    async def compute_group_reward(
        self, rollout_ids, messages_list, example_args_list, termination_reasons
    ):
        return [{"r": float("nan")} for _ in rollout_ids]


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


# ---------------------------------------------------------------------------
# Group-reward helpers
# ---------------------------------------------------------------------------


def test_overrides_compute_group_reward_predicate():
    assert overrides_compute_group_reward(_GoodGroupEnv) is True
    assert overrides_compute_group_reward(_GoodGroupEnv()) is True  # instance too
    assert overrides_compute_group_reward(_PlainEnv) is False


def test_contract_ok_returns_summary():
    summary = assert_group_reward_contract(
        _GoodGroupEnv(), ["a", "b"], [[], []], [{}, {}]
    )
    assert "2 dict" in summary


def test_contract_raises_on_non_list():
    with pytest.raises(ValueError, match="expected list"):
        assert_group_reward_contract(_NotListGroupEnv(), ["a"], [[]], [{}])


def test_contract_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="one per rollout_id"):
        assert_group_reward_contract(_ShortGroupEnv(), ["a", "b"], [[], []], [{}, {}])


def test_contract_raises_on_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        assert_group_reward_contract(_NonFiniteGroupEnv(), ["a"], [[]], [{}])


# ---------------------------------------------------------------------------
# Local layer (5c) end-to-end through validate_env
# ---------------------------------------------------------------------------

_LOCAL_DATASET = [{"prompt": "hi"}, {"prompt": "yo"}]


def _this_module():
    import tests.unit.platform.test_validation as mod

    return mod


def test_local_group_reward_check_passes():
    report = validate_env(
        env_class=_GoodGroupEnv,
        env_args={},
        train_dataset=_LOCAL_DATASET,
        local_modules=[_this_module()],
        verbose=False,
    )
    assert report.local_ran is True
    assert report.local_failed == 0
    assert report.remote is None


def test_local_group_reward_check_flags_bad_env(capsys):
    report = validate_env(
        env_class=_NotListGroupEnv,
        env_args={},
        train_dataset=_LOCAL_DATASET,
        local_modules=[_this_module()],
        verbose=False,
    )
    assert report.local_failed >= 1
    assert "compute_group_reward failed" in capsys.readouterr().out


def test_local_group_reward_skipped_when_not_overridden(capsys):
    report = validate_env(
        env_class=_PlainEnv,
        env_args={},
        train_dataset=_LOCAL_DATASET,
        local_modules=[_this_module()],
        verbose=False,
    )
    assert report.local_failed == 0
    assert "compute_group_reward: skipped" in capsys.readouterr().out
