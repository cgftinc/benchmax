"""Shared reward test fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmax.auth import StaticBearerAuth
from benchmax.rewards import Judge


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.fixture
def judge_factory(monkeypatch):
    """Install a deterministic OpenAI client and return a Judge builder."""

    def install(responses: Sequence[str] | Callable[[dict], str]) -> StubJudge:
        calls: list[dict] = []
        clients: list[MagicMock] = []
        iterator = iter(responses) if not callable(responses) else None

        def client_factory(*args, **kwargs):
            client = MagicMock()

            async def create(**request):
                calls.append(request)
                content = responses(request) if callable(responses) else next(iterator)
                return _response(content)

            client.chat.completions.create = AsyncMock(side_effect=create)
            client.close = AsyncMock()
            clients.append(client)
            return client

        monkeypatch.setattr("benchmax.rewards.judge.AsyncOpenAI", client_factory)
        return StubJudge(
            judge=Judge(
                model="test-judge",
                base_url="https://judge.test/v1",
                auth=StaticBearerAuth("test-token"),
            ),
            calls=calls,
            clients=clients,
        )

    return install
@dataclass
class StubJudge:
    judge: Judge
    calls: list[dict]
    clients: list[MagicMock]

