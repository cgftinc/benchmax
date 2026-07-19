"""Unit tests for castform.platform.client wire format and URL resolution."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from castform.platform.client import (
    ExampleValidation,
    LaunchArgSpec,
    RolloutClient,
    StorageClient,
    TrainerClient,
    ValidationResult,
    _BearerAuth,
)
from castform.platform.exceptions import (
    AuthenticationError,
    RolloutNotFound,
    RolloutServerError,
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
        env_cls_path="x/env-cls.pkl",
        env_metadata_path="x/env-metadata.json",
        train_dataset_path="x/train.jsonl",
        eval_dataset_path="x/eval.jsonl",
        name="test-run",
    )

    assert run_id == "run-abc"
    assert "/train/runs/launch" in captured["url"]


def test_launch_training_run_surfaces_server_warnings():
    """Soft-cap / OOM-risk warnings come back in the response and are raised
    as Python warnings so they're visible in notebooks/REPL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "runId": "run-warn",
                "warnings": [
                    '"max_rollout_len" = 32000 exceeds soft cap of 16384; proceed with caution.'
                ],
            },
        )

    trainer = _make_trainer_with_transport(handler)
    with pytest.warns(UserWarning, match=r"max_rollout_len.*16384"):
        run_id = trainer.launch_training_run(
            env_cls_path="x/env-cls.pkl",
            env_metadata_path="x/env-metadata.json",
            train_dataset_path="x/train.jsonl",
            eval_dataset_path="x/eval.jsonl",
        )
    assert run_id == "run-warn"


def test_launch_training_run_omits_training_run_type_from_body():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "r1"})

    trainer = _make_trainer_with_transport(handler)
    trainer.launch_training_run(
        env_cls_path="a",
        env_metadata_path="b",
        train_dataset_path="c",
        eval_dataset_path="d",
    )

    assert "type" not in captured["body"]
    assert captured["body"]["args"]["env_cls_path"] == "a"


def test_launch_training_run_rejects_training_run_type_kwarg():
    trainer = _make_trainer_with_transport(
        lambda request: httpx.Response(200, json={"runId": "unused"})
    )

    with pytest.raises(TypeError, match="training_run_type"):
        trainer.launch_training_run(
            training_run_type="simple-cpu",
            env_cls_path="a",
            env_metadata_path="b",
            train_dataset_path="c",
            eval_dataset_path="d",
        )


def test_launch_training_run_rejects_trainer_ref_kwarg():
    trainer = _make_trainer_with_transport(
        lambda request: httpx.Response(200, json={"runId": "unused"})
    )

    with pytest.raises(TypeError, match="trainer_ref"):
        trainer.launch_training_run(
            trainer_ref="main",
            env_cls_path="a",
            env_metadata_path="b",
            train_dataset_path="c",
            eval_dataset_path="d",
        )


def test_launch_training_run_reads_run_id_not_experiment_id():
    """Regression guard: server returns {runId}, not {experimentId}."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"runId": "the-id"})

    trainer = _make_trainer_with_transport(handler)
    assert (
        trainer.launch_training_run(
            env_cls_path="a",
            env_metadata_path="b",
            train_dataset_path="c",
            eval_dataset_path="d",
        )
        == "the-id"
    )


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
        "name": "max_rollout_len",
        "label": "max rollout length",
        "type": "integer",
        "required": False,
        "description": "Total tokens generated across the whole rollout.",
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
    assert "max_rollout_len" in out
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
            env_cls_path="a",
            env_metadata_path="b",
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
            env_cls_path="a",
            env_metadata_path="b",
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
        httpx_mod,
        "stream",
        lambda method, url, json=None, **kw: _StubCM(json),
    )

    with pytest.raises(RuntimeError, match="stub"):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a",
            env_metadata_path="b",
            # llm_base_url=None → resolves to platform default → key forwarding allowed
        )

    assert captured["payload"]["llm"]["api_key"] == "platform-key"
    assert captured["payload"]["llm"]["base_url"].endswith("/v1")


# ---------------------------------------------------------------------------
# Per-request credential resolution (api_key optional → resolves via the seam)
# ---------------------------------------------------------------------------


def test_bearer_auth_resolves_per_request():
    """_BearerAuth calls token_provider on every request — never frozen."""
    tokens = iter(["tok-1", "tok-2"])
    auth = _BearerAuth(lambda: next(tokens))

    req1 = httpx.Request("GET", "https://x/")
    list(auth.auth_flow(req1))
    assert req1.headers["authorization"] == "Bearer tok-1"

    req2 = httpx.Request("GET", "https://x/")
    list(auth.auth_flow(req2))
    assert req2.headers["authorization"] == "Bearer tok-2"


def test_storage_client_optional_api_key_resolves_via_seam(monkeypatch):
    """No api_key → StorageClient resolves the bearer from PLATFORM_API_KEY."""
    monkeypatch.delenv("ACT_AS_TOKEN_PATH", raising=False)
    monkeypatch.setenv("PLATFORM_API_KEY", "sk_seam")
    client = StorageClient(base_url="https://example.invalid")
    assert client._token_provider() == "sk_seam"


def test_trainer_client_resolves_bearer_per_request():
    """A rotating token_provider is re-resolved on each request, not frozen at
    construction (regression guard for the StorageClient/TrainerClient bake)."""
    seen: list[str] = []
    tokens = iter(["tok-1", "tok-2"])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"args": []})

    client = TrainerClient(
        base_url="https://example.invalid",
        token_provider=lambda: next(tokens),
    )
    # Keep the real auth flow; only swap in a MockTransport.
    client._http_client = httpx.Client(
        base_url="https://example.invalid",
        auth=_BearerAuth(client._token_provider),
        transport=httpx.MockTransport(handler),
    )

    client.list_launch_args()
    client.list_launch_args()
    assert seen == ["Bearer tok-1", "Bearer tok-2"]


def test_stream_rollout_resolves_bearer_and_llm_key_via_seam(monkeypatch):
    """api_key unset → BOTH the platform-service header and the platform-LLM
    leg key resolve via the seam (PLATFORM_API_KEY here). Guards the LLM-leg
    fix: the rollout's own completion call must not go out with an empty key."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    monkeypatch.delenv("ACT_AS_TOKEN_PATH", raising=False)
    monkeypatch.setenv("PLATFORM_API_KEY", "sk_seam")

    import httpx as httpx_mod

    captured: dict[str, Any] = {}

    class _StubCM:
        def __init__(self, payload, headers):
            captured["payload"] = payload
            captured["headers"] = headers

        def __enter__(self):
            raise RuntimeError("stub: skipping SSE loop")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        httpx_mod,
        "stream",
        lambda method, url, json=None, headers=None, **kw: _StubCM(json, headers),
    )

    client = RolloutClient(server_url="https://rollout.example")
    with pytest.raises(RuntimeError, match="stub"):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a",
            env_metadata_path="b",
        )

    assert captured["headers"]["Authorization"] == "Bearer sk_seam"
    assert captured["payload"]["llm"]["api_key"] == "sk_seam"


def test_stream_rollout_raises_without_any_credential(monkeypatch, tmp_path):
    """No explicit key and no seam credential → fail loudly before the network."""
    monkeypatch.delenv("ACT_AS_TOKEN_PATH", raising=False)
    monkeypatch.delenv("PLATFORM_API_KEY", raising=False)
    # Isolate from a logged-in dev's ~/.castform/credentials.json fallback — else the
    # resolver mints a real token and hits the network instead of failing loudly.
    monkeypatch.setenv("CASTFORM_CREDENTIALS_PATH", str(tmp_path / "none.json"))

    client = RolloutClient(server_url="https://rollout.example")
    with pytest.raises(RuntimeError, match="No Castform platform credential"):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a",
            env_metadata_path="b",
        )


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
            env_cls_path="a",
            env_metadata_path="b",
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
            env_cls_path="a",
            env_metadata_path="b",
        )
    assert exc_info.value.status_code == 403


def test_stream_rollout_raises_rollout_not_found_on_404(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 404, b"no such endpoint")

    client = RolloutClient(api_key="k")
    with pytest.raises(RolloutNotFound):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a",
            env_metadata_path="b",
        )


def test_stream_rollout_raises_rollout_server_error_on_5xx(monkeypatch):
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    _stream_with_status(monkeypatch, 503, b"down for maintenance")

    client = RolloutClient(api_key="k")
    with pytest.raises(RolloutServerError):
        client.stream_rollout(
            raw_example={"prompt": "hi"},
            env_cls_path="a",
            env_metadata_path="b",
        )


# ---------------------------------------------------------------------------
# A2: ValidationResult bool-castable + per-example detail
# ---------------------------------------------------------------------------


def test_validation_result_is_bool_castable():
    assert bool(ValidationResult(examples=[ExampleValidation(0, True)])) is True
    assert (
        bool(ValidationResult(examples=[ExampleValidation(0, False, "err")])) is False
    )
    assert bool(ValidationResult(examples=[])) is True  # vacuously true


def test_validation_result_ok_property():
    r = ValidationResult(
        examples=[
            ExampleValidation(0, True),
            ExampleValidation(1, False, "boom"),
        ]
    )
    assert r.ok is False
    assert r.examples[1].error == "boom"


# ---------------------------------------------------------------------------
# validate_examples(env_class=...) — bundle locally to bytes, no upload needed
# ---------------------------------------------------------------------------


def _make_smoke_env():
    """A minimal concrete BaseEnv defined in a local scope so cloudpickle
    pickles it by value (no local-module ref for dump_bundle to reject)."""
    from benchmax.envs import BaseEnv

    class _SmokeEnv(BaseEnv):
        async def create_dataset(self, split, base_dir):
            raise NotImplementedError

        async def compute_reward(
            self, rollout_id, messages, example_args, *, termination_reason
        ):
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


def test_validate_examples_full_messages_surfaces_transcript(monkeypatch):
    """full_messages=True asks stream_rollout to capture_messages and surfaces
    the streamed transcript on each ExampleValidation (so `python main.py validate
    --json` can carry real completions for a reward audit)."""
    client = RolloutClient(api_key="k")

    captured: list[dict[str, Any]] = []
    transcript = [{"role": "assistant", "content": "the answer"}]

    def _fake_stream_rollout(**kwargs):
        captured.append(kwargs)
        return {"success": True, "rewards": {"r": 1.0}, "messages": transcript}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_smoke_env(),
        n=1,
        full_messages=True,
        verbose=False,
    )

    assert result.ok
    assert captured[0]["capture_messages"] is True
    assert result.examples[0].messages == transcript


def test_validate_examples_omits_messages_without_full_messages(monkeypatch):
    """Default (full_messages=False) → capture_messages off, messages stays None."""
    client = RolloutClient(api_key="k")

    captured: list[dict[str, Any]] = []

    def _fake_stream_rollout(**kwargs):
        captured.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_smoke_env(),
        n=1,
        verbose=False,
    )

    assert captured[0]["capture_messages"] is False
    assert result.examples[0].messages is None


def test_validate_examples_retries_transient_worker_error(monkeypatch):
    """A one-off worker_error (infra flake) is retried once and then succeeds —
    it must not fail the example."""
    client = RolloutClient(api_key="k")

    calls = {"n": 0}

    def _fake_stream_rollout(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"event": "worker_error", "error": "worker_args.json not found"}
        return {"event": "rollout_completed", "success": True, "rewards": {"r": 1.0}}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}], env_class=_make_smoke_env(), n=1, verbose=False
    )

    assert calls["n"] == 2  # first (flaked) + one retry
    assert result.ok
    assert result.examples[0].ok


def test_validate_examples_persistent_worker_error_fails_after_retry(monkeypatch):
    """A worker_error that persists through the retry is recorded as a failure —
    the retry is bounded (one extra attempt), not infinite."""
    client = RolloutClient(api_key="k")

    calls = {"n": 0}

    def _fake_stream_rollout(**kwargs):
        calls["n"] += 1
        return {"event": "worker_error", "error": "sandbox setup failed"}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}], env_class=_make_smoke_env(), n=1, verbose=False
    )

    assert calls["n"] == 2  # original + exactly one retry, then give up
    assert not result.ok
    assert result.examples[0].error


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


def test_validate_examples_forwards_llm_base_url_and_key(monkeypatch):
    """Regression for the URL-wiring bug: validate_examples must thread
    llm_base_url/llm_api_key into each stream_rollout call, otherwise the
    rollout's LLM leg silently falls back to the default-domain host."""
    client = RolloutClient(api_key="k")

    captured: list[dict[str, Any]] = []

    def _fake_stream_rollout(**kwargs):
        captured.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(client, "stream_rollout", _fake_stream_rollout)

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_smoke_env(),
        n=1,
        llm_base_url="https://llm.castform.dev/v1",
        llm_api_key="dev-key",
        verbose=False,
    )

    assert result.ok
    assert len(captured) == 1
    assert captured[0]["llm_base_url"] == "https://llm.castform.dev/v1"
    assert captured[0]["llm_api_key"] == "dev-key"


# ---------------------------------------------------------------------------
# validate_examples — faithful server-side compute_group_reward check
#
# The group reward runs SERVER-SIDE: validate_examples submits a one-example
# batch with samples_per_example=N to rollout-service (run_group), which forms a
# real co-located sibling group and runs compute_group_reward in the trainer
# image. A failure comes back as group_reward_error per rollout. These tests
# fake stream_rollout (per-example smoke) and run_group (the group batch).
# ---------------------------------------------------------------------------


def _make_group_env():
    """A concrete BaseEnv that OVERRIDES compute_group_reward so the group check
    fires. The body never runs here — the group reward executes server-side,
    which run_group mocks away."""
    from benchmax.envs import BaseEnv

    class _GroupEnv(BaseEnv):
        async def create_dataset(self, split, base_dir):
            raise NotImplementedError

        async def compute_reward(
            self, rollout_id, messages, example_args, *, termination_reason
        ):
            return {"r": 1.0}

        async def compute_group_reward(
            self,
            rollout_ids,
            messages_list,
            example_args_list,
            termination_reasons,
        ):
            return [{"r": 1.0} for _ in rollout_ids]

    return _GroupEnv


def _completed(success=True, group_reward_error=None):
    """A rollout_completed event in the shape run_group returns."""
    e: dict[str, Any] = {"event": "rollout_completed", "success": success}
    if group_reward_error is not None:
        e["group_reward_error"] = group_reward_error
    return e


def test_validate_examples_runs_server_side_group(monkeypatch):
    """Override + clean group → group reward validated server-side, result ok.
    The group is examples[0] run as samples_per_example=N via run_group."""
    client = RolloutClient(api_key="k")
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})
    seen: dict[str, Any] = {}

    def fake_run_group(example, *, samples, **kw):
        seen["example"] = example
        seen["samples"] = samples
        return [_completed(), _completed()]

    monkeypatch.setattr(client, "run_group", fake_run_group)

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_group_env(),
        n=1,
        group_reward_samples=2,
        verbose=False,
    )

    assert result.ok
    assert result.group_reward is not None and result.group_reward.ok
    assert result.group_reward.index == -1
    assert seen["example"] == {"prompt": "hi"}
    assert seen["samples"] == 2


def test_validate_examples_server_side_group_error_fails(monkeypatch):
    """A server-reported group_reward_error fails the whole result."""
    client = RolloutClient(api_key="k")
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})
    monkeypatch.setattr(
        client,
        "run_group",
        lambda example, *, samples, **kw: [
            _completed(),
            _completed(group_reward_error="ValueError: boom"),
        ],
    )

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_group_env(),
        n=1,
        verbose=False,
    )

    assert result.group_reward is not None
    assert result.group_reward.ok is False
    assert "boom" in (result.group_reward.error or "")
    assert result.ok is False


def test_validate_examples_skips_group_when_proxy_missing(monkeypatch):
    """If the batch proxy isn't deployed yet (404), the group check is SKIPPED,
    not failed — so the SDK can land ahead of platform-service's proxy."""
    client = RolloutClient(api_key="k")
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})

    def _not_found(*a, **k):
        raise RolloutNotFound("no such endpoint", 404)

    monkeypatch.setattr(client, "run_group", _not_found)

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_group_env(),
        n=1,
        verbose=False,
    )

    assert result.group_reward is None
    assert result.ok is True


def _counting_run_group(counter: dict[str, int]):
    def _stub(*a, **k):
        counter["n"] += 1
        return []

    return _stub


def test_validate_examples_check_group_reward_false_skips_group(monkeypatch):
    """check_group_reward=False → run_group is never called."""
    client = RolloutClient(api_key="k")
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})
    called = {"n": 0}
    monkeypatch.setattr(client, "run_group", _counting_run_group(called))

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_group_env(),
        n=1,
        check_group_reward=False,
        verbose=False,
    )

    assert result.group_reward is None
    assert result.ok is True
    assert called["n"] == 0


def test_validate_examples_no_override_skips_group(monkeypatch):
    """An env that doesn't override compute_group_reward → run_group not called."""
    client = RolloutClient(api_key="k")
    monkeypatch.setattr(client, "stream_rollout", lambda **kw: {"success": True})
    called = {"n": 0}
    monkeypatch.setattr(client, "run_group", _counting_run_group(called))

    result = client.validate_examples(
        [{"prompt": "hi"}],
        env_class=_make_smoke_env(),  # no compute_group_reward override
        n=1,
        verbose=False,
    )

    assert result.group_reward is None
    assert result.ok is True
    assert called["n"] == 0


# _assess_group_events — verdict logic over a group's rollout_completed events


def test_assess_group_events_ok():
    client = RolloutClient(api_key="k")
    v = client._assess_group_events([_completed(), _completed()], 2, verbose=False)
    assert v.ok is True and v.index == -1


def test_assess_group_events_surfaces_error():
    client = RolloutClient(api_key="k")
    v = client._assess_group_events(
        [_completed(), _completed(group_reward_error="TypeError: x")], 2, verbose=False
    )
    assert v.ok is False and "TypeError" in (v.error or "")


def test_assess_group_events_all_failed():
    client = RolloutClient(api_key="k")
    events = [_completed(success=False), _completed(success=False)]
    events[0]["error"] = "rollout blew up"
    v = client._assess_group_events(events, 2, verbose=False)
    assert v.ok is False and "blew up" in (v.error or "")


# run_group — one-example batch + batch-SSE consumption


def test_run_group_parses_batch_sse(monkeypatch):
    """run_group POSTs a one-example batch (samples_per_example=N) to
    /v1/rollout/batch/stream and collects the rollout_completed events."""
    monkeypatch.setenv("CASTFORM_PLATFORM_URL", "https://api.castform.com")
    import httpx as httpx_mod

    captured: dict[str, Any] = {}
    lines = [
        'data: {"event": "batch_started", "total": 2}',
        "",
        'data: {"event": "rollout_completed", "success": true}',
        "",
        'data: {"event": "rollout_completed", "success": true, '
        '"group_reward_error": "ValueError: boom"}',
        "",
        'data: {"event": "batch_completed", "total": 2, "succeeded": 2, "failed": 0}',
        "",
    ]

    class _FakeResp:
        status_code = 200

        def iter_lines(self):
            return iter(lines)

        def read(self):
            return b""

    class _CM:
        def __enter__(self):
            return _FakeResp()

        def __exit__(self, *a):
            return False

    def _fake_stream(method, url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        return _CM()

    monkeypatch.setattr(httpx_mod, "stream", _fake_stream)

    client = RolloutClient(api_key="k")
    events = client.run_group(
        {"prompt": "hi"},
        samples=2,
        env_cls_bytes=b"x",
        env_metadata_bytes=b"y",
        verbose=False,
    )

    assert captured["url"] == "https://api.castform.com/v1/rollout/batch/stream"
    assert captured["json"]["options"]["samples_per_example"] == 2
    # One group → one worker (siblings co-locate); pinning this avoids an empty
    # second worker crashing on a no-example partition.
    assert captured["json"]["concurrent_workers"] == 1
    assert len(events) == 2
    assert events[1]["group_reward_error"] == "ValueError: boom"


def test_run_group_ignores_worker_error(monkeypatch):
    """A worker_error event (e.g. a stray empty-partition worker) is non-fatal:
    run_group keeps the rollout_completed events instead of raising."""
    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.com")
    import httpx as httpx_mod

    lines = [
        'data: {"event": "batch_started", "total": 2}',
        "",
        'data: {"event": "worker_error", "exit_code": 1, "error": "empty partition"}',
        "",
        'data: {"event": "rollout_completed", "success": true}',
        "",
        'data: {"event": "rollout_completed", "success": true}',
        "",
        'data: {"event": "batch_completed", "total": 2, "succeeded": 2, "failed": 0}',
        "",
    ]

    class _FakeResp:
        status_code = 200

        def iter_lines(self):
            return iter(lines)

        def read(self):
            return b""

    class _CM:
        def __enter__(self):
            return _FakeResp()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx_mod, "stream", lambda *a, **k: _CM())

    client = RolloutClient(api_key="k")
    events = client.run_group(
        {"prompt": "hi"},
        samples=2,
        env_cls_bytes=b"x",
        env_metadata_bytes=b"y",
        verbose=False,
    )
    # worker_error didn't raise; both real rollouts came through.
    assert len(events) == 2
    assert all(e["success"] for e in events)


def test_validation_result_group_reward_folds_into_ok():
    """A failed group_reward makes the aggregate result falsey even when every
    per-example rollout passed."""
    passing = [ExampleValidation(0, True), ExampleValidation(1, True)]
    assert ValidationResult(examples=passing).ok is True
    good = ValidationResult(examples=passing, group_reward=ExampleValidation(-1, True))
    assert good.ok is True
    bad = ValidationResult(
        examples=passing, group_reward=ExampleValidation(-1, False, "boom")
    )
    assert bad.ok is False
