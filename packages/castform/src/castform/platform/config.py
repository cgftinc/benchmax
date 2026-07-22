from dataclasses import dataclass, field

from castform import config as benchmax_config

__all__ = ["PlatformConfig"]


@dataclass
class PlatformConfig:
    """Platform and model-endpoint settings used by generated pipelines."""

    api_key: str = ""
    base_url: str = field(default_factory=benchmax_config.platform_url)
    llm_api_key: str = ""
    llm_base_url: str = field(default_factory=benchmax_config.llm_url)
