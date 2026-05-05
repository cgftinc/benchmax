"""Unit tests for benchmax.platform.client wire format and URL resolution."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from benchmax.platform.client import (
    ExampleValidation,
    RolloutClient,
    TrainerClient,
    ValidationResult,
)
from benchmax.platform.exceptions import (
    AuthenticationError,
    RolloutError,
    RolloutNotFound,
    RolloutServerError,
    RolloutStreamError,
)


# ---------------------------------------------------------------------------
# Wire format: launch_training_run hits the right path and reads runId
# ---------------------------------------------------------------------------


def _make_trainer_with_transport(handler) -> TrainerClient:
    """Construct a TrainerClient whose HTTP client uses a MockTransport."""
    client = TrainerClient(api_key="test-key", base_url="https://example.invalid")
    client._http_client = httpx.Client(
        base_url="https://example.invalid",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def test_launch_training_run_posts_to_train_runs_launch():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"runId": "run-abc"})

    trainer = _make_trainer_with_transport(handler)
    run_id = trainer.launch_training_run(
        training_run_type="simple",
        env_cls_path="x/env-cls.bmxp",
        env_metadata_path="x/env-metadata.json",
        train_dataset_path="x/train.jsonl",
        eval_dataset_path="x/eval.jsonl",
        name="test-run",
    )

    assert run_id == "run-abc"
    assert "/train/runs/launch" in captured["url"]


def test_launch_training_run_sends_training_run_type_in_body():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "r1"})

    trainer = _make_trainer_with_transport(handler)
    trainer.launch_training_run(
        training_run_type="simple-r5",
        env_cls_path="a", env_metadata_path="b",
        train_dataset_path="c", eval_dataset_path="d",
    )

    assert captured["body"]["type"] == "simple-r5"
    assert captured["body"]["args"]["env_cls_path"] == "a"


def test_launch_training_run_reads_run_id_not_experiment_id():
    """Regression guard: server returns {runId}, not {experimentId}."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"runId": "the-id"})

    trainer = _make_trainer_with_transport(handler)
    assert trainer.launch_training_run(
        training_run_type="simple",
        env_cls_path="a", env_metadata_path="b",
        train_dataset_path="c", eval_dataset_path="d",
    ) == "the-id"


# ---------------------------------------------------------------------------
# RolloutClient: URL resolution at construction time, not import time
# ---------------------------------------------------------------------------


def test_rollout_client_picks_up_env_var_changes_after_import(monkeypatch):
    """S1 regression: setting CASTFORM_BASE_DOMAIN before constructing
    RolloutClient must take effect (was frozen at import time)."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "staging.castform.com")
    # Ensure the override env var doesn't pre-empt the base domain test.
    monkeypatch.delenv("CASTFORM_ROLLOUT_URL", raising=False)

    client = RolloutClient(api_key="k")
    assert "staging.castform.com" in client._server_url


def test_rollout_client_explicit_server_url_wins(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "staging.castform.com")
    client = RolloutClient(api_key="k", server_url="https://override.example/")
    assert client._server_url == "https://override.example"


# ---------------------------------------------------------------------------
# S2: stream_rollout refuses to forward platform key to a third-party LLM host
# ---------------------------------------------------------------------------


def test_stream_rollout_refuses_to_forward_platform_key_to_third_party_llm(monkeypatch):
    """S2 regression: when llm_base_url points outside the platform LLM endpoint,
    an explicit llm_api_key is required."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")

    client = RolloutClient(api_key="platform-key")
    with pytest.raises(ValueError, match="third-party host"):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
            llm_base_url="https://api.openai.com/v1",  # third-party
            llm_api_key="",  # missing — should raise
        )


def test_stream_rollout_allows_platform_key_for_platform_llm_endpoint(monkeypatch):
    """When llm_base_url is None (uses platform default), the platform key
    is auto-forwarded — should not raise."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")

    client = RolloutClient(api_key="platform-key", server_url="https://rollout.example")

    # Stub httpx.stream so we don't hit the network. We only care that the
    # pre-flight key-forwarding check passes; we raise inside the context to
    # avoid exercising the SSE loop in this test.
    import httpx as httpx_mod
    captured: dict[str, Any] = {}

    class _StubCM:
        def __init__(self, payload):
            captured["payload"] = payload
        def __enter__(self):
            raise RuntimeError("stub: skipping SSE loop")
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        httpx_mod, "stream",
        lambda method, url, json=None, **kw: _StubCM(json),
    )

    with pytest.raises(RuntimeError, match="stub"):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
            # llm_base_url=None → resolves to platform default → key forwarding allowed
        )

    assert captured["payload"]["llm"]["api_key"] == "platform-key"
    assert captured["payload"]["llm"]["base_url"].endswith("/v1")


# ---------------------------------------------------------------------------
# A1: typed errors for HTTP status codes
# ---------------------------------------------------------------------------


def _stream_with_status(monkeypatch, status: int, body: bytes = b""):
    """Make httpx.stream return a Response with the given status."""
    import httpx as httpx_mod

    class _CM:
        def __enter__(self):
            return httpx_mod.Response(status, content=body)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx_mod, "stream", lambda *a, **kw: _CM())


def test_stream_rollout_raises_authentication_error_on_401(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 401, b"bad token")

    client = RolloutClient(api_key="bad")
    with pytest.raises(AuthenticationError) as exc_info:
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
        )
    assert exc_info.value.status_code == 401


def test_stream_rollout_raises_rollout_not_found_on_404(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 404, b"no such endpoint")

    client = RolloutClient(api_key="k")
    with pytest.raises(RolloutNotFound):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
        )


def test_stream_rollout_raises_rollout_server_error_on_5xx(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 503, b"down for maintenance")

    client = RolloutClient(api_key="k")
    with pytest.raises(RolloutServerError):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
        )


# ---------------------------------------------------------------------------
# A2: ValidationResult bool-castable + per-example detail
# ---------------------------------------------------------------------------


def test_validation_result_is_bool_castable():
    assert bool(ValidationResult(examples=[ExampleValidation(0, True)])) is True
    assert bool(ValidationResult(examples=[ExampleValidation(0, False, "err")])) is False
    assert bool(ValidationResult(examples=[])) is True  # vacuously true


def test_validation_result_ok_property():
    r = ValidationResult(examples=[
        ExampleValidation(0, True),
        ExampleValidation(1, False, "boom"),
    ])
    assert r.ok is False
    assert r.examples[1].error == "boom"
