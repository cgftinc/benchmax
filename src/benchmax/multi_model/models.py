"""Data models for multi-model benchmarking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict


class OAIReasoningArgs(TypedDict, total=False):
    """OpenAI reasoning configuration for gpt-5 and o-series models."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    summary: Literal["auto", "concise", "detailed"] | None


class OAIArgs(TypedDict, total=False):
    """OpenAI-specific arguments for the Responses API."""

    verbosity: Literal["low", "medium", "high"] | None
    reasoning: OAIReasoningArgs | None


@dataclass
class ModelResponse:
    """Single response from a model."""

    answer: str
    thinking: str | None  # Only from response structure, not parsed from text
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_response: Any = field(repr=False)  # Original response object
    reasoning_summary: str | None = None  # OAI reasoning summary when requested

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResult:
    """Aggregated results for a single model across N responses."""

    model: str
    responses: list[ModelResponse]
    input_price_per_1k: float
    output_price_per_1k: float

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.responses)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.responses)

    @property
    def total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost

    @property
    def avg_latency_ms(self) -> float:
        if not self.responses:
            return 0.0
        return sum(r.latency_ms for r in self.responses) / len(self.responses)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.responses)

    @property
    def num_responses(self) -> int:
        return len(self.responses)


@dataclass
class BenchmarkResult:
    """Complete benchmark results across all models."""

    prompt: str
    system_prompt: str | None
    results: dict[str, ModelResult]  # model_name -> ModelResult
    timestamp: datetime = field(default_factory=datetime.now)

    def sorted_by_cost(self, ascending: bool = True) -> list[ModelResult]:
        """Return results sorted by total cost."""
        return sorted(
            self.results.values(),
            key=lambda r: r.total_cost,
            reverse=not ascending,
        )

    def sorted_by_latency(self, ascending: bool = True) -> list[ModelResult]:
        """Return results sorted by average latency."""
        return sorted(
            self.results.values(),
            key=lambda r: r.avg_latency_ms,
            reverse=not ascending,
        )

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.results.values())

    @property
    def models(self) -> list[str]:
        return list(self.results.keys())
