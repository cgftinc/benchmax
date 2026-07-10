"""Shared fixtures: stub `AsyncOpenAI` so no test hits the network."""

from __future__ import annotations

from typing import Callable, List
from unittest.mock import AsyncMock, MagicMock

import pytest


def _mk_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _stub_client_factory(responses: List[str] | Callable[..., str]):
    """Return a callable that mimics AsyncOpenAI(...).

    `responses` is either an iterable of canned content strings (returned in
    order) or a callable taking the captured kwargs and returning a string.
    """
    calls: list[dict] = []
    iterator = iter(responses) if isinstance(responses, list) else None

    def factory(*_, **__):
        client = MagicMock()

        async def create(**kwargs):
            calls.append(kwargs)
            if iterator is not None:
                return _mk_response(next(iterator))
            assert callable(responses)
            return _mk_response(responses(kwargs))

        client.chat.completions.create = AsyncMock(side_effect=create)
        # _judge_call_with_retry closes the client after every attempt.
        client.close = AsyncMock()
        return client

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


@pytest.fixture
def stub_openai(monkeypatch):
    """Patch `AsyncOpenAI` in modules that import it; return a builder."""

    def install(responses):
        factory = _stub_client_factory(responses)
        # Construction now lives in _utils._judge_call_with_retry, so patch there.
        monkeypatch.setattr("benchmax.rubrics._utils.AsyncOpenAI", factory)
        # The judge clients resolve their bearer through resolve_judge_key ->
        # platform_bearer. With no platform env that raises by design; stub it
        # so these logic tests don't need credentials.
        monkeypatch.setattr(
            "benchmax.platform.credentials.platform_bearer", lambda: "test-token"
        )
        return factory

    return install


@pytest.fixture
def tmp_cache_file(tmp_path, monkeypatch):
    """Isolate the file-based rubric cache to a tmp directory."""
    path = str(tmp_path / "rubric_cache.json")
    monkeypatch.setattr("benchmax.rubrics.cache.RUBRIC_CACHE_FILE", path)
    return path
