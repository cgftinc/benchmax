from __future__ import annotations

import asyncio

from castform_router.policy import HeuristicRoutePolicy
from castform_router.session_router import SessionRouter
from castform_router.types import RouteRequest


def request(*, session_id: str | None, text: str) -> RouteRequest:
    return RouteRequest(
        request_id=None,
        session_id=session_id,
        task_text=text,
        candidate_models=("cheap-model", "frontier-model"),
    )


def test_first_decision_is_pinned_for_session() -> None:
    router = SessionRouter(
        policy=HeuristicRoutePolicy("cheap-model", "frontier-model"),
    )

    first = asyncio.run(
        router.route(session_id_request("session-1", "Rename this variable."))
    )
    second = asyncio.run(
        router.route(session_id_request("session-1", "Fix a distributed deadlock."))
    )

    assert first.selected_model == "cheap-model"
    assert first.cache_hit is False
    assert second.selected_model == "cheap-model"
    assert second.cache_hit is True


def test_requests_without_session_are_reclassified() -> None:
    router = SessionRouter(
        policy=HeuristicRoutePolicy("cheap-model", "frontier-model"),
    )

    first = asyncio.run(router.route(request(session_id=None, text="Rename a variable.")))
    second = asyncio.run(
        router.route(request(session_id=None, text="Fix a distributed deadlock."))
    )

    assert first.selected_model == "cheap-model"
    assert second.selected_model == "frontier-model"
    assert second.cache_hit is False


def test_expired_session_is_reclassified() -> None:
    now = [10.0]
    router = SessionRouter(
        policy=HeuristicRoutePolicy("cheap-model", "frontier-model"),
        ttl_seconds=5,
        clock=lambda: now[0],
    )

    first = asyncio.run(
        router.route(request(session_id="session-1", text="Rename a variable."))
    )
    now[0] = 16.0
    second = asyncio.run(
        router.route(request(session_id="session-1", text="Fix a distributed deadlock."))
    )

    assert first.selected_model == "cheap-model"
    assert second.selected_model == "frontier-model"
    assert second.cache_hit is False


def session_id_request(session_id: str, text: str) -> RouteRequest:
    return request(session_id=session_id, text=text)
