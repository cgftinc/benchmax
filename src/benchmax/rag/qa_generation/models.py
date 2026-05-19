"""Data models for QA generation datasets."""

from typing import Any, NotRequired, TypedDict


class ReferenceChunk(TypedDict):
    """A reference chunk that contains the answer to a question."""

    id: str
    metadata: dict[str, Any]
    content: str


class QADataPoint(TypedDict):
    """A single question-answer data point with reference chunks."""

    question: str
    task_id: NotRequired[str]
    user_question: NotRequired[str]
    retrieval_query: NotRequired[str]
    style_target: NotRequired[str]
    style_observed: NotRequired[str]
    answer: str
    reference_chunks: list[ReferenceChunk]
    qa_type: str  # "single_hop" or "multi_hop"
    min_hop_count: int | None
    is_co_located: bool | None

    # Populated by pipeline steps
    filter_status: str | None
    filter_reasoning: str | None
    no_context_answer: str | None
    eval_scores: dict

    # Chunks removed by filters (grounding_llm, hop_count_validity) with reason and filter name
    removed_reference_chunks: NotRequired[list[dict]]
