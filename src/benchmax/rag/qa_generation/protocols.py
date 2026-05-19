"""Protocol definitions for QA generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from benchmax.rag.qa_generation.anchor_selector import AnchorBundle
    from benchmax.rag.qa_generation.pipeline_config import PipelineContext, GenerationTask
    from benchmax.rag.qa_generation.generated_qa import GeneratedQA


@runtime_checkable
class ChunkLinker(Protocol):
    """Selects related chunks and returns an anchor bundle for one task."""

    def link(
        self,
        primary_chunk: Any,
        *,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
        reasoning_mode: str = "",
    ) -> AnchorBundle: ...


@runtime_checkable
class QuestionGenerator(Protocol):
    """Generates initial QA candidates from task intents."""

    def generate(self, tasks: list[GenerationTask], context: PipelineContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMSupportedGenerator(QuestionGenerator, Protocol):
    """Direct-LLM generator specialization."""

    def generate(self, tasks: list[GenerationTask], context: PipelineContext) -> list[GeneratedQA]: ...


@runtime_checkable
class EvaluatorFilter(Protocol):
    """Evaluates QA candidates and annotates each with a FilterVerdict."""

    def evaluate(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]: ...


@runtime_checkable
class LLMBasedFilter(EvaluatorFilter, Protocol):
    """Direct-LLM evaluation filter specialization."""

    def evaluate(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]: ...


@runtime_checkable
class QuestionTransformer(Protocol):
    """Transforms question text without changing answers or reference chunks."""

    def transform(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]: ...


@runtime_checkable
class Formatter(Protocol):
    """Formats final pipeline output artifacts."""

    def format(self, items: list[GeneratedQA], context: PipelineContext) -> dict[str, Any]: ...
