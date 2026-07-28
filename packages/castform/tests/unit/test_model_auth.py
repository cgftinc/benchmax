from __future__ import annotations

import pytest
from benchmax.auth import StaticBearerAuth
from castform.model_auth import CastformModelAuth, model_auth_for_endpoint
from castform.platform.config import PlatformConfig
from castform.rag.qa_generation.pipeline_config import PipelineConfig


def test_explicit_key_builds_static_auth_for_external_endpoint() -> None:
    auth = model_auth_for_endpoint(
        api_key="provider-key",
        base_url="https://api.openai.com/v1",
        purpose="test model",
    )

    assert auth == StaticBearerAuth("provider-key")


def test_external_endpoint_never_falls_back_to_platform_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CASTFORM_AUTH_TOKEN", "forbidden-auth-token")
    monkeypatch.setenv("CASTFORM_API_KEY", "forbidden-platform-key")
    monkeypatch.setenv("OPENAI_API_KEY", "implicit-openai-key")

    with pytest.raises(ValueError, match="requires an explicit API key"):
        model_auth_for_endpoint(
            api_key="",
            base_url="https://api.openai.com/v1",
            purpose="test model",
        )


def test_configured_castform_endpoint_uses_local_model_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        "castform.model_auth.config.llm_url",
        lambda: "https://llm.castform.dev/v1",
    )

    auth = model_auth_for_endpoint(
        api_key="",
        base_url="https://llm.castform.dev/v1",
        purpose="test model",
    )

    assert isinstance(auth, CastformModelAuth)


def test_pipeline_control_plane_key_is_not_inferred_as_model_key() -> None:
    cfg = PipelineConfig(platform=PlatformConfig(api_key="control-plane-key"))

    cfg.resolve_api_keys()

    assert cfg.generation.llm_direct.api_key == ""
    assert cfg.filtering.grounding_llm.judge_api_key == ""
    assert cfg.filtering.retrieval_llm.judge_api_key == ""
    assert cfg.wiki_preprocessing.api_key == ""


def test_pipeline_explicit_shared_llm_key_populates_model_components() -> None:
    cfg = PipelineConfig(
        platform=PlatformConfig(
            api_key="control-plane-key",
            llm_api_key="model-key",
        )
    )

    cfg.resolve_api_keys()

    assert cfg.generation.llm_direct.api_key == "model-key"
    assert cfg.filtering.grounding_llm.judge_api_key == "model-key"
    assert cfg.filtering.retrieval_llm.judge_api_key == "model-key"
    assert cfg.wiki_preprocessing.api_key == "model-key"
