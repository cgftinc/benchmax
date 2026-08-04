from __future__ import annotations

from castform_router.job_router import ROUTES, HeuristicJobRouter
from castform_router.types import HarnessRouteRequest


def request(
    text: str,
    *,
    session_id: str | None = None,
    override: str | None = None,
) -> HarnessRouteRequest:
    return HarnessRouteRequest(
        request_id="test-request",
        session_id=session_id,
        task_text=text,
        task_domain="software_engineering",
        user_context={"declared_role": "developer"},
        workspace_context={"repository_type": "typescript"},
        candidate_routes=ROUTES,
        route_override=override,
    )


def test_simple_task_selects_low_cost_glm_route() -> None:
    router = HeuristicJobRouter()

    decision = router.route(request("Rename this local variable."))

    assert decision.selected_route.route_id == "claude-code/glm-5.1@zai"
    assert len(decision.predictions) == 3
    assert decision.reason == "cheapest_above_quality_threshold"


def test_complex_task_selects_codex_above_quality_threshold() -> None:
    router = HeuristicJobRouter()

    decision = router.route(request("Debug a distributed race condition."))

    assert decision.selected_route.route_id == "codex/openai-codex@openai"
    assert decision.reason == "cheapest_above_quality_threshold"


def test_override_selects_complete_route() -> None:
    router = HeuristicJobRouter()

    decision = router.route(
        request(
            "Rename this variable.",
            override="claude-code/claude-sonnet@anthropic",
        )
    )

    assert (
        decision.selected_route.route_id
        == "claude-code/claude-sonnet@anthropic"
    )
    assert decision.reason == "request_override"


def test_session_pin_reuses_complete_route() -> None:
    router = HeuristicJobRouter()

    first = router.route(
        request("Debug a distributed race condition.", session_id="session-1")
    )
    second = router.route(
        request("Rename this variable.", session_id="session-1")
    )

    assert first.selected_route.route_id == "codex/openai-codex@openai"
    assert second.selected_route.route_id == first.selected_route.route_id
    assert second.cache_hit is True
