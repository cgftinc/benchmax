"""Client registry for managing API clients and model configurations."""

from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AnthropicFoundry
from openai import OpenAI

ClientType = Literal["anthropic", "openai-chat", "openai-responses"]


@dataclass
class ModelConfig:
    """Configuration for a model."""

    client_type: ClientType
    deployment_name: str


# Default model configurations
DEFAULT_MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-opus-4-5": ModelConfig("anthropic", "claude-opus-4-5"),
    "claude-sonnet-4-5": ModelConfig("anthropic", "claude-sonnet-4-5"),
    "claude-haiku-4-5": ModelConfig("anthropic", "claude-haiku-4-5"),
    # OpenAI (Responses API supported)
    "gpt-5.2": ModelConfig("openai-responses", "gpt-5.2"),
    "gpt-5.2-chat": ModelConfig("openai-responses", "gpt-5.2-chat"),
    "gpt-5-mini": ModelConfig("openai-responses", "gpt-5-mini"),
    "gpt-5-nano": ModelConfig("openai-responses", "gpt-5-nano"),
    # DeepSeek (Chat API only)
    "DeepSeek-V3.2-Speciale": ModelConfig("openai-chat", "DeepSeek-V3.2-Speciale"),
    # Meta Llama (Chat API only)
    "Llama-4-Maverick-17B-128E-Instruct-FP8": ModelConfig(
        "openai-chat", "Llama-4-Maverick-17B-128E-Instruct-FP8"
    ),
    # Grok (Chat API only)
    "grok-4-fast-reasoning": ModelConfig("openai-chat", "grok-4-fast-reasoning"),
    "grok-4-fast-non-reasoning": ModelConfig("openai-chat", "grok-4-fast-non-reasoning"),
    # Kimi (Chat API only)
    "Kimi-K2-Thinking": ModelConfig("openai-chat", "Kimi-K2-Thinking"),
    # Microsoft Phi (Chat API only)
    "Phi-4-reasoning": ModelConfig("openai-chat", "Phi-4-reasoning"),
    "Phi-4-mini-reasoning": ModelConfig("openai-chat", "Phi-4-mini-reasoning"),
    "Phi-4": ModelConfig("openai-chat", "Phi-4"),
    "Phi-4-mini-instruct": ModelConfig("openai-chat", "Phi-4-mini-instruct"),
}


@dataclass
class Endpoints:
    """API endpoints configuration."""

    anthropic: str
    openai: str


class ClientRegistry:
    """Manages API clients and model configurations."""

    def __init__(
        self,
        api_key: str,
        endpoints: Endpoints | dict[str, str],
        model_configs: dict[str, ModelConfig] | None = None,
    ):
        """Initialize the registry with API credentials and endpoints.

        Args:
            api_key: API key for all services (assumes shared key)
            endpoints: Endpoints configuration (Endpoints object or dict with keys:
                       'anthropic', 'openai')
            model_configs: Optional custom model configurations. If None, uses defaults.
        """
        if isinstance(endpoints, dict):
            endpoints = Endpoints(**endpoints)

        self._api_key = api_key
        self._endpoints = endpoints
        self._model_configs = model_configs or DEFAULT_MODEL_CONFIGS

        # Initialize clients
        # For AnthropicFoundry on Azure, extract resource name from URL
        # Format: https://{resource}.services.ai.azure.com/anthropic/
        anthropic_url = endpoints.anthropic
        if ".services.ai.azure.com" in anthropic_url:
            # Extract resource name from Azure URL
            resource = anthropic_url.split("https://")[1].split(".services.ai.azure.com")[0]
            self._anthropic_client = AnthropicFoundry(
                api_key=api_key,
                resource=resource,
            )
        else:
            # Non-Azure endpoint, use base_url
            self._anthropic_client = AnthropicFoundry(
                api_key=api_key,
                base_url=anthropic_url,
            )

        self._openai_client = OpenAI(
            base_url=endpoints.openai,
            api_key=api_key,
        )

    def get_client(self, model: str, use_responses_api: bool = False) -> tuple[Any, ModelConfig]:
        """Get the appropriate client and config for a model.

        Args:
            model: Model name
            use_responses_api: Deprecated parameter, kept for compatibility

        Returns:
            Tuple of (client, model_config)

        Raises:
            ValueError: If model is not configured
        """
        if model not in self._model_configs:
            raise ValueError(
                f"Unknown model: {model}. Available models: {list(self._model_configs.keys())}"
            )

        config = self._model_configs[model]

        if config.client_type == "anthropic":
            return self._anthropic_client, config
        elif config.client_type in ("openai-chat", "openai-responses"):
            return self._openai_client, config
        else:
            raise ValueError(f"Unknown client type: {config.client_type}")

    @property
    def available_models(self) -> list[str]:
        """List of all available model names."""
        return list(self._model_configs.keys())

    def add_model(self, name: str, config: ModelConfig) -> None:
        """Add a custom model configuration.

        Args:
            name: Model name
            config: Model configuration
        """
        self._model_configs[name] = config
