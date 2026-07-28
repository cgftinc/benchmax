from __future__ import annotations

from benchmax.auth import StaticBearerAuth
from castform.rag.qa_generation.filters import env_rollout
from castform.rag.qa_generation.pipeline_config import LLMEnvFilterConfig


def test_disabled_env_rollout_does_not_resolve_external_judge_auth(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_openai_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        env_rollout,
        "create_openai_client",
        fake_create_openai_client,
    )
    config = LLMEnvFilterConfig(
        enabled=False,
        judge_api_key="",
        judge_base_url="https://api.openai.com/v1",
    )

    env_rollout.EnvRolloutFilter(
        rollout_client=object(),  # type: ignore[arg-type]
        cfg=config,
    )

    assert captured["auth"] == StaticBearerAuth("disabled")
