"""Model pricing definitions and preset model lists."""

from typing import Literal

# Pricing per 1000 tokens (input, output) - Global Standard deployments
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic (per 1000 tokens)
    "claude-opus-4-5": (0.005, 0.025),
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-haiku-4-5": (0.001, 0.005),
    # OpenAI (per 1M tokens -> convert to per 1K)
    "gpt-5.2": (1.75 / 1000, 14.00 / 1000),
    "gpt-5.2-chat": (1.75 / 1000, 14.00 / 1000),
    "gpt-5-mini": (0.25 / 1000, 2.00 / 1000),
    "gpt-5-nano": (0.05 / 1000, 0.40 / 1000),
    # DeepSeek (per 1000 tokens)
    "DeepSeek-V3.2-Speciale": (0.00058, 0.00168),
    # Meta Llama (per 1000 tokens)
    "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.00025, 0.001),
    # Grok (per 1000 tokens)
    "grok-4-fast-reasoning": (0.0002, 0.0005),
    "grok-4-fast-non-reasoning": (0.0002, 0.0005),
    # Kimi (per 1000 tokens)
    "Kimi-K2-Thinking": (0.0006, 0.0025),
    # Microsoft Phi (per 1000 tokens)
    "Phi-4-reasoning": (0.000125, 0.0005),
    "Phi-4-mini-reasoning": (0.000075, 0.0003),
    "Phi-4": (0.000125, 0.0005),
    "Phi-4-mini-instruct": (0.000075, 0.0003),
}


def get_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model. Returns (0, 0) if model not found."""
    return MODEL_PRICING.get(model, (0.0, 0.0))


# Preset model lists for convenience
# Budget: weighted avg cost <= $0.000750 (up to and including Llama-4)
BUDGET_MODELS: list[str] = [
    "Phi-4-mini-instruct",
    "gpt-5-nano",
    "Phi-4",
    "grok-4-fast-non-reasoning",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
]

# Mid-tier: weighted avg cost > $0.000750 and < $0.003667 (before claude-haiku-4-5)
MID_TIER_MODELS: list[str] = [
    "DeepSeek-V3.2-Speciale",
    "gpt-5-mini",
    "Kimi-K2-Thinking",
]

# Premium: weighted avg cost >= $0.003667
PREMIUM_MODELS: list[str] = [
    "claude-haiku-4-5",
    "gpt-5.2",
    "gpt-5.2-chat",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

ALL_MODELS: list[str] = list(MODEL_PRICING.keys())

# Type for sort options
SortBy = Literal["cost", "latency"]
