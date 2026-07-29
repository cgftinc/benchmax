"""Unit tests for castform.platform.client wire format and URL resolution."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from castform.platform.client import (
    LaunchArgSpec,
    RolloutClient,
    StorageClient,
    TrainerClient,
    _BearerAuth,
)
from castform.platform.exceptions import (
    AuthenticationError,
    RolloutError,
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
        import json

        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "run-abc"})

    trainer = _make_trainer_with_transport(handler)
    run_id = trainer.launch_training_run(
        env_cls_path="x/env-cls.pkl",
        env_metadata_path="x/env-metadata.json",
        dataset_path="x/data",
        name="test-run",
    )

    assert run_id == "run-abc"
    assert "/train/runs/launch" in captured["url"]
    assert captured["body"]["args"]["dataset_path"] == "x/data"


def test_launch_training_run_omits_dataset_path_when_not_uploaded():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "run-with-runtime-data"})

    trainer = _make_trainer_with_transport(handler)
    run_id = trainer.launch_training_run(
        env_cls_path="x/env-cls.pkl",
        env_metadata_path="x/env-metadata.json",
    )

    assert run_id == "run-with-runtime-data"
    assert captured["body"]["args"] == {
        "env_cls_path": "x/env-cls.pkl",
        "env_metadata_path": "x/env-metadata.json",
    }


def test_launch_training_run_surfaces_server_warnings():
    """Soft-cap / OOM-risk warnings come back in the response and are raised
    as Python warnings so they're visible in notebooks/REPL."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "runId": "run-warn",
                "warnings": [
                    '"max_context_len" = 40000 exceeds soft cap of 32768; proceed with caution.'
                ],
            },
        )

    trainer = _make_trainer_with_transport(handler)
    with pytest.warns(UserWarning, match=r"max_context_len.*32768"):
        run_id = trainer.launch_training_run(
            env_cls_path="x/env-cls.pkl",
            env_metadata_path="x/env-metadata.json",
            dataset_path="x/data",
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
        dataset_path="c",
    )

    assert "type" not in captured["body"]
    assert captured["body"]["args"]["env_cls_path"] == "a"


def test_launch_training_run_filters_reserved_paths_from_launcher_args():
    """launcher_args cannot smuggle in or override the reserved path keys."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"runId": "r1"})

    trainer = _make_trainer_with_transport(handler)
    trainer.launch_training_run(
        env_cls_path="a",
        env_metadata_path="b",
        launcher_args={
            "env_cls_path": "sneaky",
            "dataset_path": "sneaky",
            "max_context_len": 4000,
        },
    )

    # dataset_path was not supplied as a kwarg, so it must not appear at all.
    assert "dataset_path" not in captured["body"]["args"]
    assert captured["body"]["args"]["env_cls_path"] == "a"
    assert captured["body"]["args"]["max_context_len"] == 4000


def test_launch_training_run_rejects_training_run_type_kwarg():
    trainer = _make_trainer_with_transport(
        lambda request: httpx.Response(200, json={"runId": "unused"})
    )

    with pytest.raises(TypeError, match="training_run_type"):
        trainer.launch_training_run(
            training_run_type="simple-cpu",
            env_cls_path="a",
            env_metadata_path="b",
            dataset_path="c",
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
            dataset_path="c",
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
            dataset_path="c",
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
        "name": "max_context_len",
        "label": "max context length",
        "type": "integer",
        "required": False,
        "description": "Total prompt and response tokens across the whole rollout.",
        "warnAbove": 32768,
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
    assert msl.warn_above == 32768

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
    assert "max_context_len" in out
    assert "warn_above=32768" in out
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
    """No api_key → StorageClient resolves the bearer from CASTFORM_API_KEY."""
    monkeypatch.delenv("ACT_AS_TOKEN_PATH", raising=False)
    monkeypatch.setenv("CASTFORM_API_KEY", "sk_seam")
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


# ---------------------------------------------------------------------------
# run_group — one-example batch + batch-SSE consumption


def test_run_group_posts_group_native_dataset_contract(monkeypatch):
    """run_group pins the worker-created Dataset to its first item."""
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
        samples=2,
        env_cls_path="env/cls.pkl",
        env_metadata_path="env/meta.json",
        dataset_path="datasets/frozen",
        model="test-model",
        verbose=False,
    )

    assert captured["url"] == "https://api.castform.com/v1/rollout/batch/stream"
    assert captured["json"]["group_size"] == 2
    assert captured["json"]["max_examples"] == 1
    assert captured["json"]["model"] == {"name": "test-model"}
    assert captured["json"]["dataset_path"] == "datasets/frozen"
    assert "sampling" not in captured["json"]
    assert "llm" not in captured["json"]
    assert "options" not in captured["json"]
    assert len(events) == 2
    assert events[1]["group_reward_error"] == "ValueError: boom"


def test_run_group_rejects_worker_error(monkeypatch):
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
    with pytest.raises(RolloutError, match="empty partition"):
        client.run_group(
            samples=2,
            env_cls_path="env/cls.pkl",
            env_metadata_path="env/meta.json",
            verbose=False,
        )
