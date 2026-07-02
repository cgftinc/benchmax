"""Public-API guards for the keyless device-auth surface.

Two contracts the wizard codegen relies on (generated run.py scripts):
  - ``platform_bearer`` is importable from ``benchmax.platform`` — generated
    scripts hand it to a raw OpenAI client (e.g. the traces pivot).
  - ``PlatformConfig()`` constructs with no key — an empty key resolves through
    the credential seam, like every sibling config and the keyless clients.
"""

from __future__ import annotations

import pytest

import benchmax.platform as platform


def test_platform_bearer_is_public() -> None:
    # Re-exported at the package top level, not only in
    # benchmax.platform.credentials, so `from benchmax.platform import
    # platform_bearer` resolves.
    from benchmax.platform import platform_bearer

    assert callable(platform_bearer)
    assert "platform_bearer" in platform.__all__


def test_platform_config_is_keyless_by_default() -> None:
    PlatformConfig = pytest.importorskip(
        "benchmax.rag.qa_generation.pipeline_config"
    ).PlatformConfig

    cfg = PlatformConfig()  # no api_key — must not raise
    assert cfg.api_key == ""
    # URLs still resolve from config (session/env-derived), never None.
    assert cfg.base_url
    assert cfg.llm_base_url
