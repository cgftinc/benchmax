from __future__ import annotations

from typing import Any

from castform_router import task_cli


def _result() -> dict[str, Any]:
    return {
        "trace_id": "trace-123",
        "session_id": "terminal-demo",
        "router_duration_ms": 42.5,
        "router_model_label": "Qwen 0.8B",
        "router_output": {
            "predictions": [
                {"route_id": "codex/openai-codex@openai"},
                {"route_id": "claude-code/glm-5.1@zai"},
            ]
        },
        "selected_route": {
            "route_id": "codex/openai-codex@openai",
            "harness": "codex",
            "model": "openai-codex",
            "provider": "openai",
            "gateway_model": "codex-route",
        },
        "decision": {
            "reason": "cheapest_above_quality_threshold",
            "policy_version": "policy-v1",
            "cache_hit": False,
        },
        "response": {
            "choices": [{"message": {"content": "Mock response"}}],
        },
        "events": [
            {
                "timestamp": "2026-07-31T00:00:00+00:00",
                "actor": "Trace UI",
                "stage": "client.task_submitted",
                "summary": "Submitted the task.",
                "input": {"task_text": "write a script"},
            },
            {
                "timestamp": "2026-07-31T00:00:01+00:00",
                "actor": "Decision policy",
                "stage": "policy.route_selected",
                "summary": "Selected a route.",
                "output": {"route_id": "codex/openai-codex@openai"},
            },
        ],
    }


def test_task_command_submits_workspace_context_and_prints_main_flow(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured.update(url=url, payload=payload, timeout=timeout)
        return _result()

    monkeypatch.setattr(task_cli, "_post_json", fake_post)

    exit_code = task_cli.main(
        ["task", "write", "a", "script", "--session", "terminal-demo"]
    )

    assert exit_code == 0
    assert captured["url"] == "http://localhost:3000/api/ask"
    assert captured["payload"]["question"] == "write a script"
    assert captured["payload"]["user_context"]["client"] == "terminal"
    assert captured["payload"]["workspace_context"]["repository_path"]
    output = capsys.readouterr().out
    assert "Main flow" in output
    assert "[1/5] TASK" in output
    assert "[2/5] QWEN ROUTE SCORING" in output
    assert "LiteLLM → Qwen 0.8B → 2 route scores" in output
    assert "[3/5] CASTFORM ROUTE SELECTION" in output
    assert "codex/openai-codex@openai" in output
    assert "[4/5] APPROVAL" in output
    assert "Not implemented yet" in output
    assert "[5/5] CODING HARNESS (SIMULATED)" in output
    assert "codex → LiteLLM (codex-route) → openai" in output
    assert "[01] client.task_submitted" not in output


def test_task_command_verbose_prints_low_level_trace(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(task_cli, "_post_json", lambda *args: _result())

    assert task_cli.main(["task", "write a script", "--verbose"]) == 0

    output = capsys.readouterr().out
    assert "Observability trace (2 events)" in output
    assert "[01] client.task_submitted" in output
    assert '"task_text": "write a script"' in output
    assert "[02] policy.route_selected" in output


def test_task_command_reports_connection_failure(monkeypatch: Any, capsys: Any) -> None:
    def fail(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        del url, payload, timeout
        raise RuntimeError("server unavailable")

    monkeypatch.setattr(task_cli, "_post_json", fail)

    assert task_cli.main(["task", "test the repo"]) == 1
    assert "castform: server unavailable" in capsys.readouterr().err
