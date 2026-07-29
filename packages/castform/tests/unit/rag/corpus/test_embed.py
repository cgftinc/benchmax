"""Tests for the platform-backed ``embed_fn`` helper."""

from __future__ import annotations

import logging
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import pytest

from castform.rag.corpus.embed import DEFAULT_EMBED_MODEL, platform_embed_fn
from castform.rag.corpus.neon.search import NeonSearch


class _FakeOpenAI:
    """Records construction args + embeddings calls; returns a vector per input."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, *, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[dict] = []
        self.closed = False
        self.embeddings = SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self._create)
        )
        _FakeOpenAI.instances.append(self)

    def _create(self, *, model, input):
        self.calls.append({"model": model, "input": list(input)})
        # one vector per input row, length = len(text) so we can assert ordering
        data = [SimpleNamespace(embedding=[float(len(t))]) for t in input]
        parsed = SimpleNamespace(data=data)
        return SimpleNamespace(status_code=200, parse=lambda: parsed)

    def close(self):
        self.closed = True


def _patch(monkeypatch, *, key="tok-seam"):
    _FakeOpenAI.instances.clear()
    monkeypatch.delenv("PLATFORM_API_KEY", raising=False)
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    # bypass the credential seam (no network / no real token needed)
    monkeypatch.setattr(
        "castform.rag.corpus.embed.resolve_judge_key_with_source",
        lambda api_key, base_url, *, explicit_source="constructor_arg": (
            api_key or key,
            explicit_source if api_key else "fallback",
        ),
    )


def test_returns_one_vector_per_input_in_order(monkeypatch):
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)

    fn = platform_embed_fn()
    out = fn(["a", "bbb"])

    assert out == [[1.0], [3.0]]
    client = _FakeOpenAI.instances[0]
    assert client.base_url == "https://llm.test.example/v1"
    assert client.api_key == "tok-seam"
    assert client.calls[0]["model"] == DEFAULT_EMBED_MODEL
    assert client.calls[0]["input"] == ["a", "bbb"]


def test_client_built_lazily_and_once(monkeypatch):
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)
    monkeypatch.setenv("PLATFORM_API_KEY", "stable-platform-key")

    fn = platform_embed_fn()
    assert _FakeOpenAI.instances == []  # nothing constructed until first call

    fn(["x"])
    fn(["y"])
    assert len(_FakeOpenAI.instances) == 1  # reused across calls


def test_base_url_resolves_at_call_time_from_config(monkeypatch):
    # Cross-env story: a sandbox sets CASTFORM_BASE_DOMAIN, which must be read when the
    # fn runs, not when it was authored.
    _patch(monkeypatch)
    fn = platform_embed_fn()  # authored with no base_url

    monkeypatch.setenv("CASTFORM_BASE_DOMAIN", "castform.dev")
    monkeypatch.delenv("CASTFORM_LLM_URL", raising=False)
    fn(["q"])

    assert _FakeOpenAI.instances[0].base_url == "https://llm.castform.dev/v1"


def test_explicit_overrides_win(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setenv("PLATFORM_API_KEY", "sk-platform")
    fn = platform_embed_fn(
        model="custom-embed", base_url="https://override/v1", api_key="sk-explicit"
    )
    fn(["q"])

    client = _FakeOpenAI.instances[0]
    assert client.base_url == "https://override/v1"
    assert client.api_key == "sk-explicit"
    assert client.calls[0]["model"] == "custom-embed"


def test_neon_search_caller_path_is_reported(monkeypatch, caplog):
    _patch(monkeypatch)
    search = NeonSearch(
        "corpus",
        embed_fn=platform_embed_fn(
            base_url="https://llm.test.example/v1",
            api_key="constructor-secret",
        ),
    )

    with caplog.at_level(logging.INFO, logger="castform.rag.corpus.embed"):
        assert search.embed("query-secret") == [12.0]

    messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_bearer" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "caller_path=castform.rag.corpus.neon.search.NeonSearch.embed" in messages[0]
    assert "constructor-secret" not in messages[0]
    assert "query-secret" not in messages[0]


def test_platform_api_key_bypasses_general_credential_seam(
    monkeypatch, tmp_path, caplog
):
    _patch(monkeypatch)
    credential_seam = Mock()
    monkeypatch.setattr(
        "castform.rag.corpus.embed.resolve_judge_key_with_source",
        credential_seam,
    )
    token_path = tmp_path / "act-as.jwt"
    token_path.write_text("stale-act-as")
    monkeypatch.setenv("ACT_AS_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("PLATFORM_API_KEY", "sk-platform")

    fn = platform_embed_fn(base_url="https://llm.test.example/v1")
    with caplog.at_level(logging.INFO, logger="castform.rag.corpus.embed"):
        fn(["first"])
        fn(["second"])

    assert len(_FakeOpenAI.instances) == 1
    assert _FakeOpenAI.instances[0].api_key == "sk-platform"
    credential_seam.assert_not_called()
    request_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_request" in record.getMessage()
    ]
    assert len(request_messages) == 1
    assert "bearer_source_class=platform_api_key" in request_messages[0]
    assert "bearer_source=PLATFORM_API_KEY" in request_messages[0]
    assert "token_state=stable_cached" in request_messages[0]
    bearer_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_bearer" in record.getMessage()
    ]
    assert len(bearer_messages) == 2
    assert "bearer_resolution=fresh_resolved" in bearer_messages[0]
    assert "client_cache_state=miss_stored" in bearer_messages[0]
    assert "bearer_resolution=reused_cached" in bearer_messages[1]
    assert "client_cache_state=hit_reused" in bearer_messages[1]


def test_general_credential_seam_remains_fallback(monkeypatch):
    _patch(monkeypatch, key="fallback-act-as")

    fn = platform_embed_fn(base_url="https://llm.test.example/v1")
    fn(["q"])

    assert _FakeOpenAI.instances[0].api_key == "fallback-act-as"


def test_credential_seam_is_resolved_into_a_fresh_client_per_call(
    monkeypatch, tmp_path, caplog
):
    _FakeOpenAI.instances.clear()
    monkeypatch.delenv("PLATFORM_API_KEY", raising=False)
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    token_path = tmp_path / "act-as.jwt"
    token_path.write_text("act-as-one")
    monkeypatch.setenv("ACT_AS_TOKEN_PATH", str(token_path))
    fn = platform_embed_fn(base_url="https://llm.test.example/v1")

    with caplog.at_level(logging.INFO, logger="castform.rag.corpus.embed"):
        fn(["first"])
        token_path.write_text("act-as-two")
        fn(["second"])

    assert [client.api_key for client in _FakeOpenAI.instances] == [
        "act-as-one",
        "act-as-two",
    ]
    assert all(client.closed for client in _FakeOpenAI.instances)
    messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_request" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "bearer_source_class=credential_seam" in messages[0]
    assert "bearer_source=ACT_AS_TOKEN_PATH" in messages[0]
    assert "token_state=fresh_resolved" in messages[0]
    assert "token_state=reused" not in messages[0]
    bearer_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_bearer" in record.getMessage()
    ]
    assert len(bearer_messages) == 1
    assert "bearer_resolution=fresh_resolved" in bearer_messages[0]
    assert "client_cache_state=request_scoped" in bearer_messages[0]


def test_request_trace_is_sanitized_and_bounded(monkeypatch, caplog):
    _patch(monkeypatch)
    fn = platform_embed_fn(
        base_url=(
            "https://url-user:url-secret@llm.test.example/v1"
            "?api_key=query-secret#fragment-secret"
        ),
        api_key="explicit-token-secret",
    )

    with caplog.at_level(logging.INFO, logger="castform.rag.corpus.embed"):
        fn(["payload-secret-one"])
        fn(["payload-secret-two"])
        fn(["payload-secret-three"])

    request_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_request" in record.getMessage()
    ]
    assert len(request_messages) == 1
    assert "endpoint=https://llm.test.example/v1/embeddings" in request_messages[0]
    assert "status_code=200" in request_messages[0]
    assert "bearer_source_class=explicit_api_key" in request_messages[0]
    assert "bearer_source=constructor_arg" in request_messages[0]
    assert "token_state=stable_cached" in request_messages[0]
    bearer_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_bearer" in record.getMessage()
    ]
    assert len(bearer_messages) == 2
    assert "bearer_resolution=fresh_resolved" in bearer_messages[0]
    assert "client_cache_state=miss_stored" in bearer_messages[0]
    assert (
        "bearer_set_by=castform.rag.corpus.embed.PlatformEmbedFn.__call__"
        in bearer_messages[0]
    )
    assert (
        "caller_path=tests.unit.rag.corpus.test_embed."
        "test_request_trace_is_sanitized_and_bounded" in bearer_messages[0]
    )
    assert "bearer_resolution=reused_cached" in bearer_messages[1]
    assert "client_cache_state=hit_reused" in bearer_messages[1]
    joined = "\n".join(request_messages + bearer_messages)
    for secret in (
        "url-user",
        "url-secret",
        "query-secret",
        "fragment-secret",
        "explicit-token-secret",
        "payload-secret",
    ):
        assert secret not in joined


def test_repeated_request_failures_log_once_without_error_details(monkeypatch, caplog):
    class StatusError(RuntimeError):
        status_code = 401

    class FailingOpenAI:
        def __init__(self, *, base_url, api_key):
            del base_url, api_key
            self.embeddings = SimpleNamespace(
                with_raw_response=SimpleNamespace(create=self._create)
            )

        @staticmethod
        def _create(*, model, input):
            del model, input
            raise StatusError("response-body-secret")

    _patch(monkeypatch)
    monkeypatch.setattr("openai.OpenAI", FailingOpenAI)
    fn = platform_embed_fn(
        base_url="https://llm.test.example/v1",
        api_key="explicit-token-secret",
    )

    with caplog.at_level(logging.INFO, logger="castform.rag.corpus.embed"):
        for _ in range(3):
            with pytest.raises(StatusError):
                fn(["request-payload-secret"])

    messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_request" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "status_code=401" in messages[0]
    assert "bearer_source_class=explicit_api_key" in messages[0]
    assert "bearer_source=constructor_arg" in messages[0]
    assert "token_state=stable_cached" in messages[0]
    bearer_messages = [
        record.getMessage()
        for record in caplog.records
        if "embeddings_bearer" in record.getMessage()
    ]
    assert len(bearer_messages) == 2
    assert "bearer_resolution=fresh_resolved" in bearer_messages[0]
    assert "client_cache_state=miss_stored" in bearer_messages[0]
    assert "bearer_resolution=reused_cached" in bearer_messages[1]
    assert "client_cache_state=hit_reused" in bearer_messages[1]
    for secret in (
        "response-body-secret",
        "request-payload-secret",
        "explicit-token-secret",
    ):
        assert secret not in messages[0]
        assert secret not in "\n".join(bearer_messages)


def test_cloudpickle_roundtrip_before_first_call(monkeypatch):
    # The env bundle is cloudpickled before it ever runs; the fn must survive with no
    # live client captured.
    fn = platform_embed_fn(api_key="tok-seam")
    restored = pickle.loads(cloudpickle.dumps(fn))

    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)
    assert restored(["zz"]) == [[2.0]]


def test_cloudpickle_roundtrip_after_warming(monkeypatch):
    # B2: a WARMED fn (a live client cached in the instance) must still pickle safely —
    # the client is dropped by __getstate__, not serialized, and rebuilt on the far side.
    monkeypatch.setenv("CASTFORM_LLM_URL", "https://llm.test.example/v1")
    _patch(monkeypatch)

    fn = platform_embed_fn(api_key="tok-seam")
    assert fn(["a"]) == [[1.0]]  # warm: builds + caches the live client
    assert fn.__getstate__()["_client"] is None  # never serialized

    data = cloudpickle.dumps(fn)  # a bare warmed closure would drag the live client in
    _FakeOpenAI.instances.clear()
    restored = pickle.loads(data)

    assert restored(["bb"]) == [[2.0]]  # rebuilt lazily post-unpickle
    assert len(_FakeOpenAI.instances) == 1
