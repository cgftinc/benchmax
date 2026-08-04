from __future__ import annotations

from pathlib import Path

WEB_ROOT = Path(__file__).parents[1] / "web"


def test_ui_separates_800m_training_from_upstream_rungs() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="train-router"' in html
    assert "The Qwen 0.8B SFT path is wired." in html
    assert "picks_trained.jsonl" in html
    assert "Waiting for P3 dataset" in html
    assert "Code ready · data pending" in html


def test_ui_names_training_contract_and_commands() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    assert "Qwen/Qwen3.5-0.8B + LoRA" in html
    assert "expected_cache_read_tokens" in html
    assert "format-training-data" in html
    assert "train-sft" in html
    assert "evaluate-trained" in html
    assert "SFT first · RL only if the router later uses tools" in html


def test_live_playground_exposes_router_runtime_and_prompt() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="router-playground"' in html
    assert '<details class="prompt-inspector" open>' in html
    assert 'id="router-runtime-label"' in html
    assert 'id="router-system-prompt"' in html
    assert 'id="router-model-version"' in html
    assert 'id="router-duration"' in html
    assert 'id="decision-reason"' in html
    assert "Castform routes. Coding agents execute." in html
    assert "Router + orchestrator" in html
    assert "Coding-agent harnesses" in html
    assert "Model-call transport" in html
    assert 'id="advanced-trace"' in html
    assert 'id="trace-event-count"' in html
    assert "These are telemetry spans—not additional decisions." in html
    assert 'fetch("/api/router/status")' in javascript
    assert "low-level events recorded" in javascript
