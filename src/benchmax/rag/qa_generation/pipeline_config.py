"""Typed models and config loaders for the QA pipeline."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmax import config
from benchmax.rag.qa_generation.retrieval_query import QueryRewriteConfig
from benchmax.rag.qa_generation.scoring import ScoringConfig

CORPUS_SYSTEM_PROMPT = (
    "You are a technical analyst specializing in document corpus analysis.\n"
    "Your goal is to understand the overall themes, content type, and typical query that a user might search for.\n"
    "\n"
    "Based on the context and thoughts, form an understanding of the corpus, and then provide a short summary and "
    "example search queries. Weigh the user's input if provided.\n"
    "\n"
    "Guideline:\n"
    "- Try to generalize and identify the overall theme of the corpus.\n"
    "- Use your context of the theme, domain, etc. to guess the persona and purpose of searching this corpus\n"
    "- i.e.\n"
    "    - student finding their learning notes\n"
    "    - developer looking up API documentation\n"
    "    - journalist researching a topic\n"
    "    - business analyst gathering market research\n"
    "- Use that persona to guess the types of queries they might perform.\n"
    "\n"
    "For the summary:\n"
    "- First line - corpus themes (documentation, tutorials, reference, etc.)\n"
    "- Second line - content domain (technical, business, scientific, etc.)\n"
    "- Third line - user persona and purpose (likely developer looking up API documentation)\n"
    "- Do NOT cite specific chunk content in the summary.\n"
    "\n"
    "For the example queries:\n"
    "- Provide 5-10 realistic example queries a user might search for in this corpus.\n"
    "- Use the inferred user persona and purpose to guide the query style.\n"
    "- Queries can have incomplete information, as often users do not remember full context.\n"
    "\n"
    "Return JSON with:\n"
    "- thoughts: Your analysis and reasoning here\n"
    "- summary: our summary here (3 lines as described above)\n"
    '- example_queries: List of example queries in the form of ["query1", "query2", ...]'
)

CORPUS_USER_TEMPLATE = (
    "Analyze the following document corpus:\n"
    "\n"
    "<user_context>\n"
    "{user_context}\n"
    "</user_context>\n"
    "\n"
    "<top_level_chunks>\n"
    "{top_level_content}\n"
    "</top_level_chunks>\n"
    "\n"
    "<random_sampled_chunks>\n"
    "{random_content}\n"
    "</random_sampled_chunks>\n"
    "\n"
    "Return your analysis as JSON with keys: thoughts, summary, example_queries"
)

DEFAULT_PRIMARY_TYPE_DISTRIBUTION: dict[str, float] = {
    "lookup": 0.30,
    "multi_hop": 0.70,
}

DEFAULT_REASONING_MODE_DISTRIBUTION: dict[str, float] = {
    "factual": 0.45,
    "temporal": 0.15,
    "inference": 0.30,
    "sequential": 0.10,
}

DEFAULT_HOP_DISTRIBUTION: dict[int, float] = {
    2: 0.50,
    3: 0.35,
    4: 0.15,
}

DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating retrieval difficulty. Given a question, its gold \
answer, and chunks retrieved via BM25 keyword search, determine whether the retrieval \
was trivially easy or required semantic understanding.

Rules:
- Focus ONLY on retrieval difficulty, not on whether the answer is correct or grounded.
- Do not use your own knowledge — only consider what is explicitly in the chunks.
- Classify the result using reason_tag:
  - "too_easy_lexical": the question shares obvious keywords/phrases with the retrieved \
chunks, so naive BM25 keyword search would trivially retrieve them.
  - "challenging_retrieval_pass": retrieving the right chunks required semantic \
understanding beyond simple keyword matching.
- If the retrieved chunks do not appear relevant to the question, that means retrieval \
was challenging — classify as "challenging_retrieval_pass".

Respond with JSON only:
{"confidence": <float 0-1>, \
"reason_tag": "<too_easy_lexical|challenging_retrieval_pass>", \
"reasoning": "<brief explanation>", \
"lexical_anchor_evidence": "<when reason_tag is too_easy_lexical, list the specific \
keywords/phrases shared between question and chunks that make retrieval trivial; \
otherwise omit or set to null>"}"""

DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Retrieved chunks:
{chunks_text}

Is the retrieval trivially easy via keyword matching (too_easy_lexical) \
or does it require semantic understanding (challenging_retrieval_pass)?"""

DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT = """\
You are an expert grounding judge evaluating whether a gold answer is fully supported by provided evidence chunks.

Rules:
- Use only the provided evidence chunks.
- The answer must be FULLY supported. Partial support is not enough.
- For multi-hop answers, all intermediate facts must be present in the evidence.
- Do not use external knowledge.
- Identify which specific chunk IDs provide the supporting evidence.
- CRITICAL: Reject meta-questions about the QA generation process itself. If the question asks \
about "chunks", "multi-hop questions", "generation", "retrieval", or why a question "cannot be \
created/generated", return answerable=false. These are invalid QA pairs that describe generation \
failures rather than real domain questions about the corpus content.

Respond with JSON only:
{"answerable": <bool>, "confidence": <float 0-1>, "reasoning": "<brief explanation>", "supporting_chunk_ids": ["<id1>", "<id2>", ...]}"""

DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {answer}

Evidence chunks:
{chunks_text}

Can the gold answer be fully grounded using ONLY the evidence chunks above?"""


@dataclass
class ModelConfig:
    """Model endpoint settings."""

    model: str = "gpt-5.4"
    api_key: str = ""
    base_url: str = field(default_factory=config.llm_url)


@dataclass
class RolloutLimits:
    """Rollout execution limits."""

    max_turns: int = 16
    max_tool_calls: int = 24
    max_completion_tokens: int = 2048
    timeout: float = 120.0


@dataclass
class EnvBundleConfig:
    """Environment bundle selectors for rollout components."""

    env_cls_path: str = ""
    env_metadata_path: str = ""
    env_cls_file: str = ""
    env_metadata_file: str = ""

    def as_bytes_bundle(self) -> tuple[bytes | None, bytes | None]:
        cls_bytes: bytes | None = None
        meta_bytes: bytes | None = None
        if self.env_cls_file:
            cls_bytes = Path(self.env_cls_file).read_bytes()
        if self.env_metadata_file:
            meta_bytes = Path(self.env_metadata_file).read_bytes()
        return cls_bytes, meta_bytes

    def has_paths(self) -> bool:
        return bool(self.env_cls_path and self.env_metadata_path)

    def has_files(self) -> bool:
        return bool(self.env_cls_file and self.env_metadata_file)


@dataclass
class PlatformConfig:
    """Top-level credentials."""

    api_key: str
    base_url: str = config.platform_url()
    llm_api_key: str = ""
    llm_base_url: str = field(default_factory=config.llm_url)


@dataclass
class CorpusConfig:
    """Corpus source selection."""

    docs_path: str = ""
    corpus_id: str = ""
    corpus_name: str = "default-corpus"
    show_summary: bool = True
    min_chunk_chars: int = 400


@dataclass
class EntityExtractionConfig:
    """LLM-generated patterns for extracting linkable entities from chunk content."""

    entity_names: list[str] = field(default_factory=list)
    code_patterns: dict[str, str] = field(default_factory=dict)
    domain_terms: list[str] = field(default_factory=list)
    query_templates: list[str] = field(default_factory=list)
    confidence: str = "low"  # "low" | "medium" | "high"


@dataclass
class EntityExtractionLLMConfig:
    """LLM settings for the entity extraction call."""

    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class CorpusContextConfig:
    """Optional corpus-intent and auto-profile controls for prompt conditioning."""

    enabled: bool = True
    description: str = ""
    example_queries: list[str] = field(default_factory=list)
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    num_top_level_samples: int = 4
    num_random_samples: int = 4
    min_chunk_chars: int = 400
    system_prompt: str = CORPUS_SYSTEM_PROMPT
    user_template: str = CORPUS_USER_TEMPLATE
    generate_entity_patterns: bool = True
    entity_extraction: EntityExtractionConfig | None = None
    entity_extraction_llm: EntityExtractionLLMConfig = field(
        default_factory=EntityExtractionLLMConfig
    )
    language: str = ""  # e.g. "Korean", "Japanese". Empty = no constraint.


@dataclass
class TargetsConfig:
    """Target matrix settings for QA generation."""

    total_samples: int = 200
    primary_type_distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PRIMARY_TYPE_DISTRIBUTION)
    )
    reasoning_mode_distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REASONING_MODE_DISTRIBUTION)
    )
    hop_distribution: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_HOP_DISTRIBUTION)
    )


@dataclass
class MetadataLinkerCfg:
    """Config for ``MetadataChunkLinker``."""

    max_candidates: int = 10
    max_secondaries: int = 3
    min_chunk_chars: int = 400
    filter_same_file: bool = True
    min_coherence: float = 0.15
    max_secondary_similarity: float = 0.55
    max_primary_similarity: float = 0.55
    retry_confidence: float = 0.5
    header_keys: tuple[str, ...] = ("h2", "h3", "section_header", "title", "h1")


@dataclass
class SearchAgentLinkerCfg:
    """Config for ``SearchAgentLinker`` (rollout-based)."""

    max_turns: int = 4
    max_tool_calls: int = 4
    max_completion_tokens: int = 3072
    fallback_to_metadata: bool = True
    auto_scale_turns: bool = True
    env_bundle: EnvBundleConfig = field(default_factory=EnvBundleConfig)


@dataclass
class LinkerConfig:
    """Chunk-linking mode selection."""

    type: str = "metadata"
    metadata: MetadataLinkerCfg = field(default_factory=MetadataLinkerCfg)
    search_agent: SearchAgentLinkerCfg = field(default_factory=SearchAgentLinkerCfg)
    search_agent_pct: float = 0.0


@dataclass
class LLMDirectGenerationConfig:
    """Direct LLM generation controls."""

    model: str = "gpt-5.4"
    api_key: str = ""
    base_url: str = ""
    max_completion_tokens: int = 4096
    timeout: float = 120.0
    system_prompt: str = "You are an expert QA dataset author."
    prompt_templates_by_qa_type: dict[str, str] = field(default_factory=dict)
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class GenerationConfig:
    """Generator mode selection."""

    mode: str = "llm_direct"
    llm_direct: LLMDirectGenerationConfig = field(default_factory=LLMDirectGenerationConfig)


@dataclass
class TransformationConfig:
    """Post-generation transformation controls (style annotation only)."""


@dataclass
class DeterministicGuardsConfig:
    """Deterministic quality gates."""

    enabled: bool = True
    min_question_chars: int = 12
    min_answer_chars: int = 50
    min_answer_chars_multi_hop: int = 100
    min_reference_chunks: int = 1
    # Jaccard similarity threshold for chunk-to-chunk overlap.
    # If any pair of reference chunks exceeds this, the multi-hop label is
    # suspect because both chunks contain largely the same information.
    chunk_overlap_threshold: float = 0.75


@dataclass
class QualityGateConfig:
    """Heuristic quality gate — runs before LLM filters to save cost."""

    enabled: bool = True
    reject_fragments: bool = True
    reject_structural: bool = True
    reject_guide_pointers: bool = True
    refine_thin_answers: bool = True


@dataclass
class RetrievalLLMFilterConfig:
    """Retrieval + judge filter settings."""

    enabled: bool = True
    judge_model: str = "gpt-5.4-mini"
    judge_api_key: str = ""
    judge_base_url: str = field(default_factory=config.llm_url)
    judge_system_prompt: str = DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
    judge_user_template: str = DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
    top_k: int = 5
    overlap_threshold: float = 0.5
    too_easy_confidence_threshold: float = 0.85
    too_easy_overlap_threshold: float = 0.65
    stats_key: str = "retrieval_too_easy_filter_stats"
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class GroundingLLMFilterConfig:
    """Grounding/answerability judge filter settings."""

    enabled: bool = True
    judge_model: str = "gpt-5.4-mini"
    judge_api_key: str = ""
    judge_base_url: str = field(default_factory=config.llm_url)
    judge_system_prompt: str = DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
    judge_user_template: str = DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE
    top_k: int = 5
    stats_key: str = "grounding_filter_stats"
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True


@dataclass
class LLMEnvFilterConfig:
    """Rollout-backed filter settings."""

    enabled: bool = True
    model: str = "gpt-5.4"
    api_key: str = ""
    base_url: str = field(default_factory=config.llm_url)
    judge_model: str = "gpt-5.4-mini"
    judge_api_key: str = ""
    judge_base_url: str = field(default_factory=config.llm_url)
    env_bundle: EnvBundleConfig = field(default_factory=EnvBundleConfig)
    rollout_limits: RolloutLimits = field(default_factory=RolloutLimits)


@dataclass
class HopCountValidityCfg:
    """Config for the hop-count validity filter."""

    enabled: bool = True
    mode: str = "leave_one_out"
    max_judge_calls: int = 3
    judge_model: str = "gpt-5.4-mini"
    judge_api_key: str = ""
    judge_base_url: str = field(default_factory=config.llm_url)
    max_concurrent: int = 8
    batch_enabled: bool = True
    show_batch_progress: bool = True
    stats_key: str = "hop_count_validity_stats"
    lopsided_high_threshold: float = 0.60
    lopsided_low_threshold: float = 0.15


@dataclass
class FilteringConfig:
    """Filter chain mode selection."""

    deterministic_guards: DeterministicGuardsConfig = field(
        default_factory=DeterministicGuardsConfig
    )
    filters: list[str] = field(
        default_factory=lambda: [
            "quality_gate",
            "retrieval_too_easy_llm",
            "grounding_llm",
            "hop_count_validity",
        ]
    )
    query_rewrite: QueryRewriteConfig = field(default_factory=QueryRewriteConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    retrieval_llm: RetrievalLLMFilterConfig = field(default_factory=RetrievalLLMFilterConfig)
    grounding_llm: GroundingLLMFilterConfig = field(default_factory=GroundingLLMFilterConfig)
    hop_count_validity: HopCountValidityCfg = field(default_factory=HopCountValidityCfg)
    env_rollout: LLMEnvFilterConfig = field(default_factory=LLMEnvFilterConfig)


@dataclass
class RefinementConfig:
    """Refinement controls for retrying failed filter candidates."""

    enabled: bool = True
    model: str = "gpt-5.4"
    api_key: str = ""
    base_url: str = ""
    max_refinements_per_item: int = 2
    max_same_seed_attempts_before_reanchor: int = 2
    max_rounds: int = 4
    prompt_template: str = (
        "Refine this QA pair while preserving answer correctness and anchor intent.\n"
        "qa_type={qa_type}\nfeedback={feedback}\n\n"
        "Current question: {question}\nCurrent answer: {answer}\n\n"
        "Return JSON with keys question and answer."
    )


@dataclass
class DedupConfig:
    """Question deduplication settings."""

    enabled: bool = True
    similarity_threshold: float = 0.70
    ngram_size: int = 2
    stats_key: str = "dedup_stats"


@dataclass
class MicroBatchConfig:
    """Micro-batch processing and checkpointing controls."""

    batch_size: int = 0  # 0 = auto-computed from total_samples
    checkpoint_dir: str = ""  # "" = auto: {output.dir}/.checkpoints/
    resume: bool = True
    max_iterations: int = 50
    max_parallel_batches: int = 0  # 0 = auto-computed from total_samples
    keep_checkpoints: bool = False


@dataclass
class SplitConfig:
    """Train/eval split settings."""

    train_ratio: float = 0.8
    stratify_by: list[str] = field(default_factory=lambda: ["qa_type", "style"])
    seed: int = 42


@dataclass
class OutputConfig:
    """Output artifact settings."""

    dir: str = "outputs/pipeline"
    train_jsonl: str = "train.jsonl"
    eval_jsonl: str = "eval.jsonl"


@dataclass
class WikiPreprocessingConfig:
    """Configuration for the optional wiki preprocessing stage."""

    enabled: bool = False
    min_chunks_per_page: int = 3
    max_chunks_per_page: int = 20
    max_entity_chunks: int = 100
    max_page_tokens: int = 1500
    max_context_tokens: int = 2000
    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class PipelineConfig:
    """Unified configuration for the pipeline."""

    platform: PlatformConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    corpus_context: CorpusContextConfig = field(default_factory=CorpusContextConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    wiki_preprocessing: WikiPreprocessingConfig = field(default_factory=WikiPreprocessingConfig)
    micro_batch: MicroBatchConfig = field(default_factory=MicroBatchConfig)
    random_seed: int = 42
    verbose: bool = True

    def resolve_api_keys(self) -> None:
        """Fill unset component API keys/base URLs with shared platform LLM settings."""
        shared_llm_key = self.platform.llm_api_key or self.platform.api_key
        shared_llm_base_url = self.platform.llm_base_url.strip() or config.llm_url()

        if not self.generation.llm_direct.api_key:
            self.generation.llm_direct.api_key = shared_llm_key
        if not self.generation.llm_direct.base_url:
            self.generation.llm_direct.base_url = shared_llm_base_url

        if not self.corpus_context.api_key:
            self.corpus_context.api_key = shared_llm_key
        if not self.corpus_context.base_url:
            self.corpus_context.base_url = shared_llm_base_url
        if not self.corpus_context.model:
            self.corpus_context.model = self.generation.llm_direct.model

        if not self.corpus_context.entity_extraction_llm.api_key:
            self.corpus_context.entity_extraction_llm.api_key = self.corpus_context.api_key
        if not self.corpus_context.entity_extraction_llm.base_url:
            self.corpus_context.entity_extraction_llm.base_url = self.corpus_context.base_url
        if not self.corpus_context.entity_extraction_llm.model:
            self.corpus_context.entity_extraction_llm.model = self.corpus_context.model

        if not self.filtering.retrieval_llm.judge_api_key:
            self.filtering.retrieval_llm.judge_api_key = shared_llm_key
        if not self.filtering.retrieval_llm.judge_base_url:
            self.filtering.retrieval_llm.judge_base_url = shared_llm_base_url

        if not self.filtering.grounding_llm.judge_api_key:
            self.filtering.grounding_llm.judge_api_key = shared_llm_key
        if not self.filtering.grounding_llm.judge_base_url:
            self.filtering.grounding_llm.judge_base_url = shared_llm_base_url

        if not self.filtering.hop_count_validity.judge_api_key:
            self.filtering.hop_count_validity.judge_api_key = shared_llm_key
        if not self.filtering.hop_count_validity.judge_base_url:
            self.filtering.hop_count_validity.judge_base_url = shared_llm_base_url

        if not self.filtering.env_rollout.api_key:
            self.filtering.env_rollout.api_key = shared_llm_key
        if not self.filtering.env_rollout.base_url:
            self.filtering.env_rollout.base_url = shared_llm_base_url
        if not self.filtering.env_rollout.judge_api_key:
            self.filtering.env_rollout.judge_api_key = shared_llm_key
        if not self.filtering.env_rollout.judge_base_url:
            self.filtering.env_rollout.judge_base_url = shared_llm_base_url

        if not self.refinement.api_key:
            self.refinement.api_key = shared_llm_key
        if not self.refinement.base_url:
            self.refinement.base_url = shared_llm_base_url

        if not self.wiki_preprocessing.api_key:
            self.wiki_preprocessing.api_key = shared_llm_key
        if not self.wiki_preprocessing.base_url:
            self.wiki_preprocessing.base_url = shared_llm_base_url
        if not self.wiki_preprocessing.model:
            self.wiki_preprocessing.model = self.generation.llm_direct.model


@dataclass
class GenerationTask:
    """One QA generation intent."""

    task_id: str
    qa_type: str
    target_hop_count: int
    seed_chunk_id: str
    reasoning_mode: str = ""  # "factual" | "temporal" | "inference" | "sequential"
    regeneration_prompt: str = ""
    regeneration_attempt: int = 0
    source_task_id: str = ""
    previous_failure_type: str = ""
    previous_judge_reason_tag: str = ""
    overlap_triggered: bool = False
    expected_action: str = ""
    failed_question: str = ""
    failed_answer: str = ""


@dataclass
class PipelineContext:
    """Shared runtime context for pipeline components.

    Two storage patterns are intentionally co-located here:

    - **Typed attributes** (``profile``, ``config``, ``source``): first-class,
      structured data with known shapes.  Use these when the schema is fixed.
    - **``state`` dict** (``context["key"]`` / ``context.get("key")``): dynamic,
      ad-hoc data accumulated as the pipeline runs (e.g. corpus_summary,
      seed_chunks, filter stats).  Use this for keys that only exist after
      certain stages complete.

    Do not add new typed attributes for every pipeline key — reserve them for
    data that is genuinely structured and always present.
    """

    config: PipelineConfig
    source: Any
    rng: random.Random = field(default_factory=random.Random)
    state: dict[str, Any] = field(default_factory=dict)
    profile: Any = None  # CorpusProfile | None (Any to avoid circular import)
    metrics: Any = None  # PipelineMetrics | None (Any to avoid circular import)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def setdefault(self, key: str, value: Any) -> Any:
        return self.state.setdefault(key, value)

    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.state[key] = value


@dataclass
class RunStats:
    """Typed run stats emitted by Pipeline."""

    raw_candidates_total: int = 0
    passed_total: int = 0
    rejected_total: int = 0
    regenerated_total: int = 0
    round_limit: int = 0


def _load_distribution(raw: Any, *, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if not out:
        return dict(fallback)
    return out


def _load_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        out = [str(item).strip() for item in raw if str(item).strip()]
        return out
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return []


def allocate_largest_remainder_generic(
    total: int,
    distribution: Mapping[str, float],
) -> dict[str, int]:
    """Allocate integer counts over arbitrary keys using largest remainder.

    The output is deterministic:
    - input keys are sorted before fractional tie-breaking
    - ties are resolved by key name
    """
    if total <= 0:
        return {str(key): 0 for key in sorted(distribution)}

    sanitized = {str(key): max(0.0, float(value)) for key, value in distribution.items()}
    if not sanitized:
        return {}

    norm_total = sum(sanitized.values())
    if norm_total <= 0:
        uniform = 1.0 / len(sanitized)
        normalized = {key: uniform for key in sorted(sanitized)}
    else:
        normalized = {key: value / norm_total for key, value in sanitized.items()}

    keys = sorted(normalized)
    raw = {key: normalized[key] * total for key in keys}
    base = {key: int(math.floor(raw[key])) for key in keys}
    remainder = total - sum(base.values())
    if remainder > 0:
        order = sorted(
            keys,
            key=lambda key: (raw[key] - base[key], normalized[key], key),
            reverse=True,
        )
        for key in order[:remainder]:
            base[key] += 1
    return base


def _parse_model_cfg(raw: Any, *, fallback: ModelConfig) -> ModelConfig:
    if isinstance(raw, str):
        return ModelConfig(model=raw, api_key=fallback.api_key, base_url=fallback.base_url)
    if not isinstance(raw, dict):
        return fallback
    return ModelConfig(
        model=str(raw.get("model", fallback.model)),
        api_key=str(raw.get("api_key", fallback.api_key)),
        base_url=str(raw.get("base_url", fallback.base_url)),
    )


def _collect_removed_config_keys(raw: dict[str, Any]) -> list[str]:
    """Return removed config keys present in user YAML."""
    removed_keys: list[str] = []

    targets_raw = raw.get("targets", {}) or {}
    generation_raw = raw.get("generation", {}) or {}
    filtering_raw = raw.get("filtering", {}) or {}
    guards_raw = filtering_raw.get("deterministic_guards", {}) or {}
    retrieval_raw = filtering_raw.get("retrieval_llm", {}) or {}
    refinement_raw = raw.get("refinement", {}) or {}

    if "style_distribution_by_qa_type" in targets_raw:
        removed_keys.append("targets.style_distribution_by_qa_type")
    if "enforce_style_mismatch_guard" in guards_raw:
        removed_keys.append("filtering.deterministic_guards.enforce_style_mismatch_guard")

    if "llm_env" in generation_raw:
        removed_keys.append("generation.llm_env")
    if "mode" in filtering_raw:
        removed_keys.append("filtering.mode")
    if "llm_env" in filtering_raw:
        removed_keys.append("filtering.llm_env")
    if "strategy" in refinement_raw:
        removed_keys.append("refinement.strategy")

    for key in (
        "route_unsupported_to_failure",
        "unknown_answerable_confidence_threshold",
        "require_multi_chunk_evidence_for_non_lookup_pass",
        "min_matched_reference_chunks_for_non_lookup_pass",
    ):
        if key in retrieval_raw:
            removed_keys.append(f"filtering.retrieval_llm.{key}")

    return removed_keys


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load `PipelineConfig` from YAML."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    removed_keys = _collect_removed_config_keys(raw)
    if removed_keys:
        joined = ", ".join(sorted(removed_keys))
        raise ValueError(
            "Unsupported config keys for the clean-break pipeline filtering/refinement API: "
            f"{joined}. "
            "Use filtering.filters with supported stages {'grounding_llm', 'retrieval_too_easy_llm'} "
            "and generator-based refinement without refinement.strategy."
        )

    platform_raw = raw.get("platform", {}) or {}
    api_key = str(platform_raw.get("api_key", "")).strip()
    if not api_key:
        raise ValueError("Missing required config value: platform.api_key")
    platform = PlatformConfig(
        api_key=api_key,
        base_url=str(platform_raw.get("base_url", config.platform_url())).strip(),
        llm_api_key=str(platform_raw.get("llm_api_key", "")).strip(),
        llm_base_url=str(platform_raw.get("llm_base_url", config.llm_url())).strip(),
    )

    corpus_raw = raw.get("corpus", {}) or {}
    corpus = CorpusConfig(
        docs_path=str(corpus_raw.get("docs_path", "")).strip(),
        corpus_id=str(corpus_raw.get("corpus_id", "")).strip(),
        corpus_name=str(corpus_raw.get("corpus_name", "default-corpus")).strip(),
        show_summary=bool(corpus_raw.get("show_summary", True)),
        min_chunk_chars=max(0, int(corpus_raw.get("min_chunk_chars", 400))),
    )

    corpus_context_raw = raw.get("corpus_context", {}) or {}
    corpus_context = CorpusContextConfig(
        enabled=bool(corpus_context_raw.get("enabled", True)),
        description=str(corpus_context_raw.get("description", "")).strip(),
        example_queries=_load_string_list(corpus_context_raw.get("example_queries", [])),
        model=str(corpus_context_raw.get("model", "")).strip(),
        api_key=str(corpus_context_raw.get("api_key", "")).strip(),
        base_url=str(corpus_context_raw.get("base_url", "")).strip(),
        num_top_level_samples=max(0, int(corpus_context_raw.get("num_top_level_samples", 4))),
        num_random_samples=max(0, int(corpus_context_raw.get("num_random_samples", 4))),
        min_chunk_chars=max(0, int(corpus_context_raw.get("min_chunk_chars", 400))),
        system_prompt=str(corpus_context_raw.get("system_prompt", CORPUS_SYSTEM_PROMPT)).strip()
        or CORPUS_SYSTEM_PROMPT,
        user_template=str(corpus_context_raw.get("user_template", CORPUS_USER_TEMPLATE)).strip()
        or CORPUS_USER_TEMPLATE,
        generate_entity_patterns=bool(corpus_context_raw.get("generate_entity_patterns", True)),
        entity_extraction_llm=EntityExtractionLLMConfig(
            model=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("model", "")
            ).strip(),
            api_key=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("api_key", "")
            ).strip(),
            base_url=str(
                (corpus_context_raw.get("entity_extraction_llm") or {}).get("base_url", "")
            ).strip(),
        ),
    )

    targets_raw = raw.get("targets", {}) or {}
    hop_dist_raw = targets_raw.get("hop_distribution")
    if isinstance(hop_dist_raw, dict):
        hop_dist = {int(k): float(v) for k, v in hop_dist_raw.items()}
    else:
        hop_dist = dict(DEFAULT_HOP_DISTRIBUTION)
    targets = TargetsConfig(
        total_samples=max(1, int(targets_raw.get("total_samples", 200))),
        primary_type_distribution=_load_distribution(
            targets_raw.get("primary_type_distribution"),
            fallback=dict(DEFAULT_PRIMARY_TYPE_DISTRIBUTION),
        ),
        reasoning_mode_distribution=_load_distribution(
            targets_raw.get("reasoning_mode_distribution"),
            fallback=dict(DEFAULT_REASONING_MODE_DISTRIBUTION),
        ),
        hop_distribution=hop_dist,
    )

    linker_raw = raw.get("linker", {}) or {}
    linker = LinkerConfig(
        type=str(linker_raw.get("type", "metadata")).strip().lower() or "metadata",
    )

    generation_raw = raw.get("generation", {}) or {}
    direct_raw = generation_raw.get("llm_direct", {}) or {}
    generation = GenerationConfig(
        mode=str(generation_raw.get("mode", "llm_direct")).strip().lower() or "llm_direct",
        llm_direct=LLMDirectGenerationConfig(
            model=str(direct_raw.get("model", LLMDirectGenerationConfig().model)),
            api_key=str(direct_raw.get("api_key", "")),
            base_url=str(direct_raw.get("base_url", "")).strip(),
            max_completion_tokens=max(
                100,
                int(
                    direct_raw.get(
                        "max_completion_tokens",
                        LLMDirectGenerationConfig().max_completion_tokens,
                    )
                ),
            ),
            timeout=max(10.0, float(direct_raw.get("timeout", 120.0))),
            system_prompt=str(
                direct_raw.get("system_prompt", LLMDirectGenerationConfig().system_prompt)
            ),
            prompt_templates_by_qa_type=dict(
                direct_raw.get("prompt_templates_by_qa_type", {}) or {}
            ),
            max_concurrent=max(1, int(direct_raw.get("max_concurrent", 8))),
            batch_enabled=bool(direct_raw.get("batch_enabled", True)),
            show_batch_progress=bool(direct_raw.get("show_batch_progress", True)),
        ),
    )

    transformation = TransformationConfig()

    filtering_raw = raw.get("filtering", {}) or {}
    guards_raw = filtering_raw.get("deterministic_guards", {}) or {}
    filters_raw = filtering_raw.get("filters", None)
    if filters_raw is None:
        filters_raw = list(FilteringConfig().filters)
    rewrite_raw = filtering_raw.get("query_rewrite", {}) or {}
    retrieval_raw = filtering_raw.get("retrieval_llm", {}) or {}
    grounding_raw = filtering_raw.get("grounding_llm", {}) or {}
    if isinstance(filters_raw, str):
        chain_filters = [
            token.strip().lower() for token in filters_raw.split(",") if token and token.strip()
        ]
    else:
        chain_filters = [
            str(token).strip().lower() for token in list(filters_raw) if str(token).strip()
        ]
    filtering = FilteringConfig(
        deterministic_guards=DeterministicGuardsConfig(
            enabled=bool(guards_raw.get("enabled", True)),
            min_question_chars=max(1, int(guards_raw.get("min_question_chars", 12))),
            min_answer_chars=max(1, int(guards_raw.get("min_answer_chars", 50))),
            min_answer_chars_multi_hop=max(
                1, int(guards_raw.get("min_answer_chars_multi_hop", 100))
            ),
            min_reference_chunks=max(0, int(guards_raw.get("min_reference_chunks", 1))),
            chunk_overlap_threshold=max(
                0.0,
                min(1.0, float(guards_raw.get("chunk_overlap_threshold", 0.75))),
            ),
        ),
        filters=chain_filters,
        query_rewrite=QueryRewriteConfig(
            enabled=bool(rewrite_raw.get("enabled", True)),
            max_terms=max(1, int(rewrite_raw.get("max_terms", 16))),
            max_chars=max(20, int(rewrite_raw.get("max_chars", 140))),
            apply_to_all_rows=bool(rewrite_raw.get("apply_to_all_rows", True)),
        ),
        retrieval_llm=RetrievalLLMFilterConfig(
            enabled=bool(retrieval_raw.get("enabled", True)),
            judge_model=str(retrieval_raw.get("judge_model", "gpt-5.4-mini")),
            judge_api_key=str(retrieval_raw.get("judge_api_key", "")),
            judge_base_url=str(retrieval_raw.get("judge_base_url", "")).strip(),
            judge_system_prompt=(
                str(
                    retrieval_raw.get(
                        "judge_system_prompt",
                        DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT,
                    )
                ).strip()
                or DEFAULT_RETRIEVAL_JUDGE_SYSTEM_PROMPT
            ),
            judge_user_template=(
                str(
                    retrieval_raw.get(
                        "judge_user_template",
                        DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE,
                    )
                ).strip()
                or DEFAULT_RETRIEVAL_JUDGE_USER_TEMPLATE
            ),
            top_k=max(1, int(retrieval_raw.get("top_k", 5))),
            overlap_threshold=max(
                0.0, min(1.0, float(retrieval_raw.get("overlap_threshold", 0.5)))
            ),
            too_easy_confidence_threshold=max(
                0.0,
                min(1.0, float(retrieval_raw.get("too_easy_confidence_threshold", 0.75))),
            ),
            too_easy_overlap_threshold=max(
                0.0,
                min(1.0, float(retrieval_raw.get("too_easy_overlap_threshold", 0.65))),
            ),
            stats_key=(
                str(retrieval_raw.get("stats_key", "retrieval_too_easy_filter_stats")).strip()
                or "retrieval_too_easy_filter_stats"
            ),
            max_concurrent=max(1, int(retrieval_raw.get("max_concurrent", 8))),
            batch_enabled=bool(retrieval_raw.get("batch_enabled", True)),
            show_batch_progress=bool(retrieval_raw.get("show_batch_progress", True)),
        ),
        grounding_llm=GroundingLLMFilterConfig(
            enabled=bool(grounding_raw.get("enabled", True)),
            judge_model=str(grounding_raw.get("judge_model", "gpt-5.4-mini")),
            judge_api_key=str(grounding_raw.get("judge_api_key", "")),
            judge_base_url=str(grounding_raw.get("judge_base_url", "")).strip(),
            judge_system_prompt=(
                str(
                    grounding_raw.get(
                        "judge_system_prompt",
                        DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT,
                    )
                ).strip()
                or DEFAULT_GROUNDING_JUDGE_SYSTEM_PROMPT
            ),
            judge_user_template=(
                str(
                    grounding_raw.get(
                        "judge_user_template",
                        DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE,
                    )
                ).strip()
                or DEFAULT_GROUNDING_JUDGE_USER_TEMPLATE
            ),
            top_k=max(1, int(grounding_raw.get("top_k", 5))),
            stats_key=(
                str(grounding_raw.get("stats_key", "grounding_filter_stats")).strip()
                or "grounding_filter_stats"
            ),
            max_concurrent=max(1, int(grounding_raw.get("max_concurrent", 8))),
            batch_enabled=bool(grounding_raw.get("batch_enabled", True)),
            show_batch_progress=bool(grounding_raw.get("show_batch_progress", True)),
        ),
    )

    refinement_raw = raw.get("refinement", {}) or {}
    refinement = RefinementConfig(
        enabled=bool(refinement_raw.get("enabled", True)),
        model=str(refinement_raw.get("model", "gpt-5.4")),
        api_key=str(refinement_raw.get("api_key", "")),
        base_url=str(refinement_raw.get("base_url", "")).strip(),
        max_refinements_per_item=max(0, int(refinement_raw.get("max_refinements_per_item", 3))),
        max_same_seed_attempts_before_reanchor=max(
            0,
            int(
                refinement_raw.get(
                    "max_same_seed_attempts_before_reanchor",
                    RefinementConfig().max_same_seed_attempts_before_reanchor,
                )
            ),
        ),
        max_rounds=max(0, int(refinement_raw.get("max_rounds", 4))),
        prompt_template=str(
            refinement_raw.get("prompt_template", RefinementConfig().prompt_template)
        ),
    )

    split_raw = raw.get("split", {}) or {}
    split = SplitConfig(
        train_ratio=max(0.1, min(0.95, float(split_raw.get("train_ratio", 0.8)))),
        stratify_by=list(
            split_raw.get("stratify_by", ["qa_type", "style"]) or ["qa_type", "style"]
        ),
        seed=int(split_raw.get("seed", 42)),
    )

    output_raw = raw.get("output", {}) or {}
    output = OutputConfig(
        dir=str(output_raw.get("dir", "outputs/pipeline")).strip(),
        train_jsonl=str(output_raw.get("train_jsonl", "train.jsonl")).strip(),
        eval_jsonl=str(output_raw.get("eval_jsonl", "eval.jsonl")).strip(),
    )

    micro_batch_raw = raw.get("micro_batch", {}) or {}
    micro_batch = MicroBatchConfig(
        batch_size=max(1, int(micro_batch_raw.get("batch_size", 100))),
        checkpoint_dir=str(micro_batch_raw.get("checkpoint_dir", "")).strip(),
        resume=bool(micro_batch_raw.get("resume", True)),
        max_iterations=max(1, int(micro_batch_raw.get("max_iterations", 50))),
        max_parallel_batches=max(1, int(micro_batch_raw.get("max_parallel_batches", 1))),
    )

    scoring_raw = raw.get("scoring", {}) or {}
    scoring = ScoringConfig(
        min_quality_score=max(0.0, float(scoring_raw.get("min_quality_score", 0.0))),
        grounding_weight=max(0.0, float(scoring_raw.get("grounding_weight", 0.4))),
        retrieval_difficulty_weight=max(
            0.0, float(scoring_raw.get("retrieval_difficulty_weight", 0.3))
        ),
        hop_validity_weight=max(0.0, float(scoring_raw.get("hop_validity_weight", 0.3))),
    )

    cfg = PipelineConfig(
        platform=platform,
        corpus=corpus,
        corpus_context=corpus_context,
        targets=targets,
        linker=linker,
        generation=generation,
        transformation=transformation,
        filtering=filtering,
        refinement=refinement,
        split=split,
        output=output,
        scoring=scoring,
        micro_batch=micro_batch,
        random_seed=int(raw.get("random_seed", 42)),
    )
    cfg.resolve_api_keys()
    return cfg
