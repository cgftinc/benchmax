"""Public-API guards for the keyless device-auth surface.

Two public behaviors the wizard codegen relies on (generated run.py scripts):
  - ``platform_bearer`` is importable from ``castform.platform`` — generated
    scripts hand it to a raw OpenAI client (e.g. the traces pivot).
  - ``PlatformConfig()`` constructs with no key — an empty key resolves through
    the credential seam, like every sibling config and the keyless clients.
"""

from __future__ import annotations

import castform.platform as platform
from castform.platform import PlatformConfig


def test_platform_bearer_is_public() -> None:
    # Re-exported at the package top level, not only in
    # castform.platform.credentials, so `from castform.platform import
    # platform_bearer` resolves.
    from castform.platform import platform_bearer

    assert callable(platform_bearer)
    assert "platform_bearer" in platform.__all__


def test_platform_config_is_keyless_by_default() -> None:
    cfg = PlatformConfig()  # no api_key — must not raise
    assert cfg.api_key == ""
    # URLs still resolve from config (session/env-derived), never None.
    assert cfg.base_url
    assert cfg.llm_base_url
    assert "PlatformConfig" in platform.__all__
