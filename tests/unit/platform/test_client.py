"""Unit tests for benchmax.platform.client wire format and URL resolution."""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from benchmax.platform.client import (
    ExampleValidation,
    LaunchArgSpec,
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
        env_cls_path="x/env-cls.pkl",
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
# list_launch_args / print_launch_args
# ---------------------------------------------------------------------------


_SAMPLE_LAUNCH_ARGS = [
    {
        "name": "learning_rate",
        "label": "learning rate",
        "type": "number",
        "required": False,
        "description": "Adam learning rate. Slime flag: --lr.",
        "default": 1e-5,
        "min": 0,
    },
    {
        "name": "max_response_len",
        "label": "max response length",
        "type": "integer",
        "required": False,
        "description": "Cap on generated tokens per rollout.",
        "warnAbove": 16384,
    },
    {
        "name": "model",
        "label": "model",
        "type": "string",
        "required": False,
        "description": "HuggingFace model id. Selects the trainer YAML.",
        "enum": ["Qwen/Qwen3-4B-Instruct-2507"],
    },
]


def test_list_launch_args_hits_endpoint_and_parses_response():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"args": _SAMPLE_LAUNCH_ARGS})

    trainer = _make_trainer_with_transport(handler)
    specs = trainer.list_launch_args()

    assert "/train/launch-args" in captured["url"]
    assert len(specs) == 3
    assert all(isinstance(s, LaunchArgSpec) for s in specs)

    lr = specs[0]
    assert lr.name == "learning_rate"
    assert lr.default == 1e-5
    assert lr.min == 0

    # warnAbove (camelCase from the API) maps to warn_above (snake_case in Python).
    msl = specs[1]
    assert msl.warn_above == 16384

    model = specs[2]
    assert model.enum == ("Qwen/Qwen3-4B-Instruct-2507",)


def test_list_launch_args_raises_authentication_error_on_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "missing api key"})

    trainer = _make_trainer_with_transport(handler)
    with pytest.raises(AuthenticationError):
        trainer.list_launch_args()


def test_print_launch_args_prints_each_spec(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"args": _SAMPLE_LAUNCH_ARGS})

    trainer = _make_trainer_with_transport(handler)
    trainer.print_launch_args()

    out = capsys.readouterr().out
    assert "learning_rate" in out
    assert "max_response_len" in out
    assert "warn_above=16384" in out
    assert "Qwen/Qwen3-4B-Instruct-2507" in out


# ---------------------------------------------------------------------------
# RolloutClient: URL resolution at construction time, not import time
# ---------------------------------------------------------------------------


def test_rollout_client_picks_up_env_var_changes_after_import(monkeypatch):
    """S1 regression: setting CASTFORM_BASE_DOMAIN before constructing
    RolloutClient must take effect (was frozen at import time). Rollouts route
    through platform-service, so the base derives from platform_url()."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "staging.castform.com")
    # Ensure the override env var doesn't pre-empt the base domain test.
    monkeypatch.delenv("CASTFORM_PLATFORM_URL", raising=False)

    client = RolloutClient(api_key="k")
    assert "staging.castform.com" in client._server_url


def test_rollout_client_explicit_server_url_wins(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "staging.castform.com")
    client = RolloutClient(api_key="k", server_url="https://override.example/")
    assert client._server_url == "https://override.example"


def test_rollout_client_targets_platform_service_v1(monkeypatch):
    """Rollouts route through platform-service (the API-key gate): it validates
    the sk_ key and mints an act_as JWT for rollout-service. The request path is
    /v1/rollout/stream (platform mounts the proxy at /v1)."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    monkeypatch.delenv("CASTFORM_PLATFORM_URL", raising=False)

    import httpx as httpx_mod

    captured: dict[str, Any] = {}

    class _CM:
        def __enter__(self):
            return httpx_mod.Response(503, content=b"stop before SSE loop")

        def __exit__(self, *a):
            return False

    def _fake_stream(method, url, **kw):
        captured["url"] = url
        return _CM()

    monkeypatch.setattr(httpx_mod, "stream", _fake_stream)

    client = RolloutClient(api_key="k")
    with pytest.raises(RolloutServerError):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
        )

    assert captured["url"] == "https://api.castform.com/v1/rollout/stream"


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


def test_stream_rollout_raises_authentication_error_on_403(monkeypatch):
    """platform-service's optionalAuth gate rejects a bad/expired key as 403
    ('sign in to run rollouts'), not 401 — surface it as an auth error too."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 403, b"Demo mode is disabled")

    client = RolloutClient(api_key="bad")
    with pytest.raises(AuthenticationError) as exc_info:
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a", env_metadata_path="b",
        )
    assert exc_info.value.status_code == 403


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


# ---------------------------------------------------------------------------
# validate_examples(env_class=...) — bundle locally to bytes, no upload needed
# ---------------------------------------------------------------------------


def _make_smoke_env():
    """A minimal concrete BaseEnv defined in a local scope so cloudpickle
    pickles it by value (no local-module ref for dump_bundle to reject)."""
    from benchmax.envs.base_env import BaseEnv

    class _SmokeEnv(BaseEnv):
        async def list_tools(self):
            return []

        async def run_tool(self, rollout_id, tool_name, **tool_args):
            raise NotImplementedError

        async def compute_reward(self, rollout_id, messages, task, **kwargs):
            return {"reward": 1.0}

    return _SmokeEnv


def test_validate_examples_env_class_bundles_to_bytes(monkeypatch):
    """env_class is bundled to bytes in-process (no upload) and forwarded as
    env_cls_bytes/env_metadata_bytes — never as blob paths — to each rollout."""
    client = RolloutClient(api_key="k")

    captured: list[dict[str, Any]] = []

    def _fake_stream_rollout(**kwargs):
        captured.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}, {"prompt": "yo"}],
        env_class=_make_smoke_env(),
        n=2,
        verbose=False,
    )

    assert result.ok
    assert len(captured) == 2
    for kw in captured:
        assert kw["env_cls_bytes"] is not None
        assert kw["env_metadata_bytes"] is not None
        assert kw["env_cls_path"] is None
        assert kw["env_metadata_path"] is None


def test_validate_examples_env_class_conflicts_with_explicit_env(monkeypatch):
    """env_class is mutually exclusive with explicit paths/bytes."""
    client = RolloutClient(api_key="k")
    # Stub so a missing-guard regression can't accidentally hit the network.
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})

    with pytest.raises(ValueError, match="env_class OR"):
        client.validate_examples(
            [{"prompt": "hi"}],
            env_class=_make_smoke_env(),
            env_cls_path="a",
            env_metadata_path="b",
            verbose=False,
        )
