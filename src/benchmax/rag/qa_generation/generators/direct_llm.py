"""Direct LLM generator for CgftPipeline."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tqdm.auto import tqdm

from benchmax.rag.qa_generation.anchor_selector import AnchorBundle
from benchmax.rag.qa_generation.batch_processor import BatchResult, batch_process_sync
from benchmax.rag.qa_generation.cgft_models import (
    CgftContext,
    GenerationTask,
    LLMDirectGenerationConfig,
)
from benchmax.rag.qa_generation.generated_qa import GeneratedQA
from benchmax.rag.qa_generation.helpers import render_template
from benchmax.rag.qa_generation.models import QADataPoint, ReferenceChunk
from benchmax.rag.qa_generation.style_controls import (
    allocate_largest_remainder,
    get_style_distribution,
    style_sequence_from_counts,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
_ESCAPED_OPEN_BRACE = "__CGFT_ESCAPED_OPEN_BRACE__"
_ESCAPED_CLOSE_BRACE = "__CGFT_ESCAPED_CLOSE_BRACE__"

_REASONING_MODE_INSTRUCTIONS: dict[str, str] = {
    "temporal": (
        "The question should require understanding the chronological order or "
        "time-dependent relationships across sources. Focus on when events happened "
        "relative to each other, how interactions evolved over time, or what changed "
        "between different dates."
    ),
    "inference": (
        "The question should require logical deduction — the answer is not stated "
        "directly in any single source but must be inferred by connecting evidence "
        "across sources. The user needs to reason about implications, draw conclusions, "
        "or identify patterns that aren't explicitly stated."
    ),
    "sequential": (
        "The question should require following a step-by-step sequence of events or "
        "actions that spans multiple sources. Focus on what happened first, what "
        "followed, and how one action led to the next."
    ),
}

_DEFAULT_TEMPLATE = (
    "[[if corpus_language]]LANGUAGE REQUIREMENT: You MUST generate the question and answer in {corpus_language}. "
    "Do not use any other language.\n\n[[endif]]"
    "Your task is to generate a {qa_type} question that will require "
    "{target_hop_count} search steps to answer by gathering information from multiple sources.\n\n"
    "[[if reasoning_mode_instruction]]Reasoning focus: {reasoning_mode_instruction}\n\n[[endif]]"
    "You must first reason inside <think> and </think>:\n"
    "1. Identify the 3 most distinctive content terms from EACH chunk provided.\n"
    "2. Plan how to connect the chunks into a question framed around a SCENARIO or PROBLEM — not around the topics themselves.\n"  # noqa: E501
    "3. Verify your planned question does NOT contain any of the terms from step 1. If it does, rephrase using descriptions of what those terms DO.\n\n"  # noqa: E501
    "[[if regeneration_attempt]]attempt={regeneration_attempt}\n[[endif]]"
    "[[if source_task_id]]source_task_id={source_task_id}\n[[endif]]"
    "[[if previous_failure_type]]previous_failure_type={previous_failure_type}\n[[endif]]"
    "[[if previous_judge_reason_tag]]previous_judge_reason_tag={previous_judge_reason_tag}\n[[endif]]"
    "[[if overlap_triggered]]overlap_triggered={overlap_triggered}\n[[endif]]"
    "[[if expected_action]]expected_action={expected_action}\n[[endif]]"
    "Corpus summary:\n{corpus_summary}\n\n"
    "Example queries:\n{corpus_queries}\n\n"
    "Primary chunk:\n{primary_chunk}\n\n"
    "Secondary chunks:\n{secondary_chunks}\n\n"
    "[[if evidence_chain]]How these chunks connect (from linking analysis):\n"
    "{evidence_chain}\n\n[[endif]]"
    "[[if failed_question]]Failed question:\n{failed_question}\n\n[[endif]]"
    "[[if failed_answer]]Failed answer:\n{failed_answer}\n\n[[endif]]"
    "[[if regeneration_prompt]]Feedback:\n{regeneration_prompt}\n\n[[endif]]"
    "Requirements:\n"
    "- Generate a **complicated**, realistic, user-facing {qa_type} question connecting information across chunks.\n"
    "- The question should require around {target_hop_count} retrieval/search steps to answer and avoid single-lookup shortcuts.\n"
    "- Use only chunk evidence; do not use outside knowledge.\n"
    "- Keep the question standalone and understandable without seeing source chunks.\n"
    "- Ensure the answer is correct, specific, and uniquely determined by the question and provided evidence.\n"
    "- Prefer task-oriented documentation use-cases over internal implementation trivia.\n"
    "- Avoid forced cross-topic stitching that is unlikely for a single user intent.\n"
    "- RETRIEVAL DIFFICULTY — CRITICAL:\n"
    "  1. OBFUSCATE THE REASONING PATH: Your question must NOT reveal which chunks or topics are needed.\n"  # noqa: E501
    "     Frame it as a SCENARIO or PROBLEM the user is trying to solve.\n"
    "     The search agent should have to reason about what to search for.\n"
    "     BAD: 'What is PostHog's session recording feature and how does it integrate with feature flags?'\n"  # noqa: E501
    "     → Reveals both chunks. A keyword search trivially finds both.\n"
    "     GOOD: 'When debugging why a feature rollout isn't behaving as expected for certain users,\n"  # noqa: E501
    "     what built-in tool shows you exactly what those users experienced?'\n"
    "     → Requires reasoning to connect feature flags and session recording. The connection is implicit.\n"  # noqa: E501
    "     BAD: 'How do I make a tracking snippet load from my PHP theme layer?'\n"
    "     → 'tracking snippet', 'PHP theme' are chunk terms. Keyword search retrieves directly.\n"
    "     GOOD: 'How can I ensure visitor behavior is captured on every page of my CMS site without manually adding code to each template?'\n"  # noqa: E501
    "  2. PARAPHRASE CHUNK LANGUAGE: Replace key terms from source chunks with synonyms or descriptions.\n"  # noqa: E501
    "     If a human could find the answer by searching for 3+ words from your question, rewrite it.\n"  # noqa: E501
    "  If your question names specific features, APIs, or concepts from multiple chunks,\n"
    "  you've made it too easy. Describe the PROBLEM, not the SOLUTION SPACE.\n"
    "  3. ANSWER GROUNDING: While the QUESTION should be obfuscated, the ANSWER must stay\n"
    "     closely grounded in the exact language of the source chunks. Use the same terms,\n"
    "     phrases, and specifics from the chunks in your answer. Do NOT paraphrase the answer.\n"
    "[[if target_style_keyword]]- QUERY STYLE: Generate a keyword-style search query.\n"
    "  Write the query a real user would type into a documentation search bar.\n"
    "  Use 4-8 words. Include actual product names, feature names, SDK/language names,\n"
    "  and technical terms. DO NOT obfuscate or paraphrase product terminology.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation rules above DO NOT apply to keyword queries.\n"
    "  Difficulty comes from SPECIFICITY: narrow to a specific aspect, not just the topic.\n"
    "  BAD: 'capture text combine fields rewrite path' (noun soup, no one would search this)\n"
    "  GOOD: 'posthog feature flag multivariate python setup'\n"
    "  BAD: 'stable visitor-based data reduction' (abstract gibberish)\n"
    "  GOOD: 'posthog debug mode disable sending data locally'\n"
    "  The answer must still be grounded in chunk language.\n[[endif]]"
    "[[if target_style_natural]]- QUERY STYLE: Generate a natural-style question.\n"
    "  Full natural-language question with interrogative word and question mark.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation requirements above still apply.\n[[endif]]"
    "[[if target_style_expert]]- QUERY STYLE: Generate an expert-style question.\n"
    "  Technical/diagnostic framing with domain terminology.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation requirements above still apply.\n[[endif]]"
    "[[if previous_failure_type]]- If previous_failure_type=too_easy:\n"
    "  Your previous question was REJECTED because it shares keywords with the source chunk,\n"
    "  making it trivially retrievable via BM25 keyword search.\n"
    "  REWRITE by:\n"
    "  1. Read the overlapping terms listed in the feedback (if available) — these caused the rejection.\n"  # noqa: E501
    "  2. Replace ALL chunk-derived terms with goal/problem descriptions or synonyms.\n"
    "  3. Frame as a SCENARIO: what is the user trying to accomplish? What symptom are they seeing?\n"  # noqa: E501
    "  4. Self-check: would a keyword search for any 3-word substring of your new question find the chunk? If yes, rephrase again.\n"  # noqa: E501
    "  Previous failed question: {failed_question}\n"
    "  Hint: Describe EFFECTS and USE CASES, not features and mechanisms.\n[[endif]]"
    "[[if previous_failure_type]]- If previous_failure_type=unsupported, revise answer using current chunk evidence only.\n[[endif]]"
    "- Output exactly one question and one answer.\n"
    "- In chunks_used, list the indices of chunks you referenced (0=primary, 1+=secondary).\n"
    "- CRITICAL: If the provided chunks genuinely cannot support a valid {qa_type} question "
    "(e.g., they are completely unrelated or lack sufficient connectable information), "
    'return `{{"status": "cannot_generate", "reason": "<brief explanation>"}}` instead. '
    "Do NOT output a meta-question about the generation process itself.\n\n"
    "First output your reasoning in <think>...</think>, then provide:\n"
    '```json\n{{"question": "...", "answer": "...", "answering_steps": "...", "chunks_used": [0, 1, ...]}}\n```'
)

_LOOKUP_TEMPLATE = (
    "[[if corpus_language]]LANGUAGE REQUIREMENT: You MUST generate the question and answer in {corpus_language}. "
    "Do not use any other language.\n\n[[endif]]"
    "Your task is to generate a single-hop lookup question answerable from one chunk.\n\n"
    "You must first reason inside <think> and </think>:\n"
    "1. Identify the 5 most distinctive content terms in this chunk (feature names, config keys, product names, API methods).\n"  # noqa: E501
    "2. Plan a question that a real user would ask when they need this information — framed around their GOAL or PROBLEM.\n"  # noqa: E501
    "3. Verify your planned question does NOT contain any of the 5 terms from step 1. If it does, rephrase using synonyms or descriptions of what the term DOES.\n\n"  # noqa: E501
    "[[if regeneration_attempt]]attempt={regeneration_attempt}\n[[endif]]"
    "[[if source_task_id]]source_task_id={source_task_id}\n[[endif]]"
    "[[if previous_failure_type]]previous_failure_type={previous_failure_type}\n[[endif]]"
    "[[if previous_judge_reason_tag]]previous_judge_reason_tag={previous_judge_reason_tag}\n[[endif]]"
    "[[if overlap_triggered]]overlap_triggered={overlap_triggered}\n[[endif]]"
    "[[if expected_action]]expected_action={expected_action}\n[[endif]]"
    "Corpus summary:\n{corpus_summary}\n\n"
    "Example queries:\n{corpus_queries}\n\n"
    "Primary chunk:\n{primary_chunk}\n\n"
    "[[if failed_question]]Failed question:\n{failed_question}\n\n[[endif]]"
    "[[if failed_answer]]Failed answer:\n{failed_answer}\n\n[[endif]]"
    "[[if regeneration_prompt]]Feedback:\n{regeneration_prompt}\n\n[[endif]]"
    "Requirements:\n"
    "- The question must be answerable from the primary chunk alone.\n"
    "- RETRIEVAL DIFFICULTY — CRITICAL:\n"
    "  1. OBFUSCATE KEY TERMS: Your question must NOT use terminology from the chunk.\n"
    "     Describe the GOAL or PROBLEM, not the mechanism or feature.\n"
    "     If a human could find the answer by searching for 3+ content words from your question, rewrite it.\n"  # noqa: E501
    "     BAD: 'How do I set up a dynamic exclusion in Tableau?'\n"
    "     → Shares terminology with the chunk. Keyword search trivially retrieves it.\n"
    "     GOOD: 'How can I automatically hide future time periods in a dashboard based on the selected date grouping?'\n"  # noqa: E501
    "     BAD: 'How do I connect my PostHog account to a v0 chat assistant?'\n"
    "     → Shares 'PostHog', 'connect', 'chat assistant' with chunk.\n"
    "     GOOD: 'I want my product analytics tool to surface usage insights inside the AI coding interface I use — how do I wire that up?'\n"  # noqa: E501
    "  2. PARAPHRASE CHUNK LANGUAGE: Replace feature names, API methods, and config keys with\n"
    "     descriptions of what they DO or what PROBLEM they solve.\n"
    "  3. ANSWER GROUNDING (MANDATORY): While the QUESTION must be obfuscated, the ANSWER must stay\n"  # noqa: E501
    "     closely grounded in the exact language of the source chunk. Use the same terms,\n"
    "     phrases, and specifics from the chunk in your answer. Do NOT paraphrase the answer.\n"
    "     The question tests retrieval skill; the answer tests extraction accuracy.\n"
    "[[if target_style_keyword]]- QUERY STYLE: Generate a keyword-style search query.\n"
    "  Write the query a real user would type into a documentation search bar.\n"
    "  Use 4-8 words. Include actual product names, feature names, SDK/language names,\n"
    "  and technical terms. DO NOT obfuscate or paraphrase product terminology.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation rules above DO NOT apply to keyword queries.\n"
    "  Difficulty comes from SPECIFICITY: narrow to a specific aspect, not just the topic.\n"
    "  BAD: 'browser event tracking instance' (vague, meaningless without context)\n"
    "  GOOD: 'posthog flutter autocapture setup guide'\n"
    "  The answer must still be grounded in chunk language.\n[[endif]]"
    "[[if target_style_natural]]- QUERY STYLE: Generate a natural-style question.\n"
    "  Full natural-language question with interrogative word and question mark.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation requirements above still apply.\n[[endif]]"
    "[[if target_style_expert]]- QUERY STYLE: Generate an expert-style question.\n"
    "  Technical/diagnostic framing with domain terminology.\n"
    "  The RETRIEVAL DIFFICULTY obfuscation requirements above still apply.\n[[endif]]"
    "- The answer should be specific and grounded in chunk evidence.\n"
    "- Use only chunk evidence; do not use outside knowledge.\n"
    "- Prefer task-oriented use-cases over internal implementation trivia.\n"
    "[[if previous_failure_type]]- If previous_failure_type=too_easy:\n"
    "  Your previous question was REJECTED because it shares keywords with the source chunk,\n"
    "  making it trivially retrievable via BM25 keyword search.\n"
    "  REWRITE by:\n"
    "  1. Read the overlapping terms listed in the feedback (if available) — these caused the rejection.\n"  # noqa: E501
    "  2. Replace ALL chunk-derived terms with goal/problem descriptions or synonyms.\n"
    "  3. Frame as a SCENARIO: what is the user trying to accomplish? What symptom are they seeing?\n"  # noqa: E501
    "  4. Self-check: would a keyword search for any 3-word substring of your new question find the chunk? If yes, rephrase again.\n"  # noqa: E501
    "  Previous failed question: {failed_question}\n"
    "  Hint: Describe EFFECTS and USE CASES, not features and mechanisms.\n[[endif]]"
    "[[if previous_failure_type]]- If previous_failure_type=unsupported, revise answer "
    "using current chunk evidence only.\n[[endif]]"
    "- Output exactly one question and one answer.\n"
    "- In chunks_used, list the indices of chunks you referenced (0=primary).\n"
    "- CRITICAL: If the chunk is too generic, boilerplate, or navigation-only to support "
    "a meaningful question, return "
    '`{{"status": "cannot_generate", "reason": "<brief explanation>"}}` instead.\n\n'
    "First output your reasoning in <think>...</think>, then provide:\n"
    '```json\n{{"question": "...", "answer": "...", "chunks_used": [0]}}\n```'
)


def _render_template_safe(template: str, variables: dict[str, Any]) -> str:
    try:
        return render_template(template, variables)
    except KeyError:
        protected = template.replace("{{", _ESCAPED_OPEN_BRACE).replace("}}", _ESCAPED_CLOSE_BRACE)
        required_fields = set(_TEMPLATE_FIELD_RE.findall(protected))
        missing_fields = sorted(field for field in required_fields if field not in variables)
        if not missing_fields:
            raise

        logger.warning(
            "Template missing variables (%s); injecting empty values.",
            ", ".join(missing_fields),
        )
        merged = dict(variables)
        for field in missing_fields:
            merged[field] = ""
        return render_template(template, merged)


def _create_chat_completion_with_fallback(client: OpenAI, **kwargs: Any) -> Any:
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        msg = str(exc).lower()
        retry_kwargs = dict(kwargs)
        if "unsupported parameter" in msg and "max_tokens" in msg:
            retry_kwargs.pop("max_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        if "unsupported parameter" in msg and "max_completion_tokens" in msg:
            retry_kwargs.pop("max_completion_tokens", None)
            return client.chat.completions.create(**retry_kwargs)
        raise


def _parse_qa_response(raw_text: str) -> tuple[str, str, list[int] | None]:
    text = (raw_text or "").strip()
    if not text:
        return "", "", None

    # Strip <think> blocks before parsing
    text = _THINK_BLOCK_RE.sub("", text).strip()
    # Strip unclosed <think> blocks (e.g. model hit output_tokens limit mid-reasoning)
    text = _THINK_UNCLOSED_RE.sub("", text).strip()

    # Build a list of JSON candidates, best-first:
    # 1. Content from markdown code fences (matches the prompt's requested format)
    # 2. The greedy {.*} regex fallback
    candidates = list(_CODE_FENCE_RE.findall(text))
    candidates.extend(_JSON_BLOCK_RE.findall(text))
    # Also try the full text (in case it's raw JSON with no wrapper)
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        # Check for explicit generation failure signal
        status = str(payload.get("status", "")).strip().lower()
        if status == "cannot_generate":
            reason = str(payload.get("reason", "")).strip()
            logger.info(
                "Generator signaled cannot_generate: %s", reason[:200] if reason else "no reason"
            )
            return "", "", None

        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if question and answer:
            raw_chunks_used = payload.get("chunks_used")
            chunks_used: list[int] | None = None
            if isinstance(raw_chunks_used, list):
                chunks_used = [int(i) for i in raw_chunks_used if isinstance(i, (int, float))]
            return question, answer, chunks_used

    question_match = _QUESTION_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    if question_match and answer_match:
        return question_match.group(1).strip(), answer_match.group(1).strip(), None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0], "\n".join(lines[1:]), None

    logger.warning(
        "Failed to parse QA from LLM response (first 500 chars): %s",
        raw_text[:500] if raw_text else "(empty)",
    )
    return "", "", None


def _chunk_to_reference_chunk(chunk: Any) -> ReferenceChunk:
    metadata: dict[str, Any] = {}
    if hasattr(chunk, "metadata_dict"):
        metadata = dict(chunk.metadata_dict)
    elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        metadata = dict(chunk.metadata)
    elif isinstance(chunk, dict):
        metadata = dict(chunk.get("metadata", {}) or {})

    chunk_id = (
        getattr(chunk, "hash", None)
        or metadata.get("id")
        or metadata.get("file")
        or metadata.get("document_id")
        or str(chunk)[:80]
    )
    content = chunk.content if hasattr(chunk, "content") else str(chunk)
    return {"id": str(chunk_id), "metadata": metadata, "content": str(content)}


def _format_secondary_chunks(anchor: AnchorBundle) -> str:
    chunks = list(anchor.secondary_chunks)
    if not chunks:
        return "(none)"
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if hasattr(chunk, "chunk_str"):
            parts.append(f"[{idx}] {chunk.chunk_str()}")
        else:
            parts.append(f"[{idx}] {str(chunk)}")
    return "\n\n".join(parts)


@dataclass
class _PreparedTask:
    """Intermediate representation of a prepared generation task."""

    task: GenerationTask
    anchor: Any  # AnchorBundle
    prompt: str
    system_prompt: str
    resolved_qa_type: str
    resolved_hop_count: int
    requested_hop_count: int


class DirectLLMGenerator:
    """Direct-call generator using task+anchor intent."""

    def __init__(
        self,
        *,
        client: OpenAI,
        linker: Any,
        cfg: LLMDirectGenerationConfig,
    ) -> None:
        self.client = client
        self.linker = linker
        self.cfg = cfg

    def generate(self, tasks: list[GenerationTask], context: CgftContext) -> list[GeneratedQA]:
        if not self.cfg.batch_enabled or len(tasks) <= 1:
            return self._generate_sequential(tasks, context)
        return self._generate_batched(tasks, context)

    def _generate_sequential(
        self, tasks: list[GenerationTask], context: CgftContext
    ) -> list[GeneratedQA]:
        """Original sequential generation path."""
        generated: list[GeneratedQA] = []
        for pt in self._prepare_tasks(tasks, context):
            completion = _create_chat_completion_with_fallback(
                self.client,
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": pt.system_prompt},
                    {"role": "user", "content": pt.prompt},
                ],
                max_completion_tokens=self.cfg.max_completion_tokens,
                timeout=self.cfg.timeout,
                temperature=1.0,
            )
            raw_text = completion.choices[0].message.content or ""
            question, answer, chunks_used = _parse_qa_response(raw_text)
            if not question or not answer:
                logger.warning("Skipping task %s: failed to parse QA response.", pt.task.task_id)
                continue
            generated.append(self._build_generated_qa(pt, question, answer, chunks_used))
        return generated

    def _generate_batched(
        self, tasks: list[GenerationTask], context: CgftContext
    ) -> list[GeneratedQA]:
        """Parallel batch generation path."""
        prepared = self._prepare_tasks(tasks, context)
        if not prepared:
            return []

        result = batch_process_sync(
            client=self.client,
            model=self.cfg.model,
            prompts=[pt.prompt for pt in prepared],
            system_prompt=[pt.system_prompt for pt in prepared],
            max_tokens=self.cfg.max_completion_tokens,
            timeout=self.cfg.timeout,
            max_concurrent=self.cfg.max_concurrent,
            show_progress=self.cfg.show_batch_progress,
            temperature=1.0,
            desc="Generating QA candidates",
        )
        return self._process_batch_results(prepared, result)

    def _prepare_tasks(
        self, tasks: list[GenerationTask], context: CgftContext
    ) -> list[_PreparedTask]:
        """Resolve anchors and build prompts for all tasks."""
        seed_lookup: dict[str, Any] = context.get("seed_chunk_lookup", {})
        corpus_pool: list[Any] = context.get("corpus_pool", [])
        corpus_summary = str(context.get("corpus_summary", "") or "").strip()
        corpus_queries = str(context.get("corpus_queries", "") or "").strip()
        corpus_description = str(context.get("corpus_description", "") or "").strip()
        corpus_language = str(context.get("corpus_language", "") or "").strip()

        style_sequences: dict[str, list[str]] = {}
        for qa_type_key in ("lookup", "multi_hop"):
            type_tasks = [t for t in tasks if (t.qa_type or "lookup") == qa_type_key]
            if type_tasks:
                dist = get_style_distribution(qa_type_key)
                counts = allocate_largest_remainder(len(type_tasks), dist)
                seq = style_sequence_from_counts(counts)
                context.rng.shuffle(seq)
                style_sequences[qa_type_key] = seq

        style_iters: dict[str, Iterator[str]] = {k: iter(v) for k, v in style_sequences.items()}

        prepared: list[_PreparedTask] = []
        show_prep_progress = self.cfg.show_batch_progress and len(tasks) > 1
        for task in tqdm(tasks, desc="Linking chunks", disable=not show_prep_progress):
            seed_chunk = seed_lookup.get(task.seed_chunk_id)
            if seed_chunk is None and corpus_pool:
                seed_chunk = context.rng.choice(corpus_pool)
            if seed_chunk is None:
                logger.warning("Skipping task %s: no seed chunk available.", task.task_id)
                continue

            anchor = self.linker.link(
                seed_chunk,
                target_hop_count=task.target_hop_count,
                corpus_pool=corpus_pool,
                reasoning_mode=task.reasoning_mode,
            )

            # Re-anchor: if a multi-hop task got no secondaries, try other
            # seed chunks before giving up.
            needs_secondaries = (task.target_hop_count or 1) > 1
            if needs_secondaries and not anchor.secondary_chunks and corpus_pool:
                tried = {getattr(seed_chunk, "hash", id(seed_chunk))}
                candidates = [c for c in corpus_pool if getattr(c, "hash", id(c)) not in tried]
                context.rng.shuffle(candidates)
                for alt_seed in candidates[:3]:
                    anchor = self.linker.link(
                        alt_seed,
                        target_hop_count=task.target_hop_count,
                        corpus_pool=corpus_pool,
                        reasoning_mode=task.reasoning_mode,
                    )
                    if anchor.secondary_chunks:
                        break
                    tried.add(getattr(alt_seed, "hash", id(alt_seed)))

            # If still no secondaries for a multi-hop task, demote to
            # lookup instead of wasting an LLM call on a doomed generation.
            demoted = False
            if needs_secondaries and not anchor.secondary_chunks:
                task = GenerationTask(
                    task_id=task.task_id,
                    qa_type="lookup",
                    target_hop_count=1,
                    seed_chunk_id=task.seed_chunk_id,
                )
                demoted = True

            resolved_qa_type = str(task.qa_type).strip() or "lookup"
            requested_hop_count = max(1, int(task.target_hop_count or 1))
            if demoted:
                resolved_hop_count = requested_hop_count
            else:
                try:
                    resolved_hop_count = int(
                        getattr(anchor, "target_hop_count", requested_hop_count)
                    )
                except (TypeError, ValueError):
                    resolved_hop_count = requested_hop_count
                resolved_hop_count = max(1, resolved_hop_count)

            target_style = next(style_iters.get(resolved_qa_type, iter([])), "")

            builtin_default = (
                _LOOKUP_TEMPLATE if resolved_qa_type == "lookup" else _DEFAULT_TEMPLATE
            )
            template = self.cfg.prompt_templates_by_qa_type.get(resolved_qa_type, builtin_default)
            reasoning_mode_instruction = _REASONING_MODE_INSTRUCTIONS.get(task.reasoning_mode, "")
            variables = {
                "qa_type": resolved_qa_type,
                "target_hop_count": resolved_hop_count,
                "reasoning_mode": task.reasoning_mode,
                "reasoning_mode_instruction": reasoning_mode_instruction,
                "corpus_summary": corpus_summary,
                "corpus_queries": corpus_queries,
                "corpus_description": corpus_description,
                "corpus_language": corpus_language,
                "regeneration_attempt": task.regeneration_attempt,
                "regeneration_prompt": task.regeneration_prompt,
                "source_task_id": task.source_task_id,
                "previous_failure_type": task.previous_failure_type,
                "previous_judge_reason_tag": task.previous_judge_reason_tag,
                "overlap_triggered": "true" if task.overlap_triggered else "",
                "expected_action": task.expected_action,
                "failed_question": task.failed_question,
                "failed_answer": task.failed_answer,
                "target_style": target_style,
                "target_style_keyword": target_style == "keyword",
                "target_style_natural": target_style == "natural",
                "target_style_expert": target_style == "expert",
                "primary_chunk": (
                    f"[0] {anchor.primary_chunk.chunk_str()}"
                    if hasattr(anchor.primary_chunk, "chunk_str")
                    else f"[0] {anchor.primary_chunk!s}"
                ),
                "secondary_chunks": _format_secondary_chunks(anchor),
                "evidence_chain": (
                    anchor.structural_hints.get("evidence_chain", "")
                    if hasattr(anchor, "structural_hints")
                    else ""
                ),
            }
            prompt = _render_template_safe(template, variables)
            system_prompt = _render_template_safe(self.cfg.system_prompt, variables)

            regeneration_prompt = str(task.regeneration_prompt or "").strip()
            if regeneration_prompt:
                prompt = (
                    f"{prompt}\n\n"
                    "Use this regeneration feedback from the failed prior attempt:\n"
                    f"{regeneration_prompt}"
                )

            prepared.append(
                _PreparedTask(
                    task=task,
                    anchor=anchor,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    resolved_qa_type=resolved_qa_type,
                    resolved_hop_count=resolved_hop_count,
                    requested_hop_count=requested_hop_count,
                )
            )
        return prepared

    def _process_batch_results(
        self, prepared: list[_PreparedTask], result: BatchResult
    ) -> list[GeneratedQA]:
        """Parse batch LLM responses and build GeneratedQA objects."""
        generated: list[GeneratedQA] = []
        n_api_fail = 0
        n_parse_fail = 0
        n_empty = 0

        for pt, response in zip(prepared, result.responses):
            if response is None:
                n_api_fail += 1
                logger.warning("Skipping task %s: batch LLM call failed.", pt.task.task_id)
                continue

            raw = response.answer or ""
            if not raw.strip():
                n_empty += 1
                logger.warning("Skipping task %s: LLM returned empty response.", pt.task.task_id)
                continue

            question, answer, chunks_used = _parse_qa_response(raw)
            if not question or not answer:
                n_parse_fail += 1
                logger.warning(
                    "Skipping task %s: failed to parse QA response "
                    "(output_tokens=%d, first 300 chars): %s",
                    pt.task.task_id,
                    response.output_tokens,
                    raw[:300],
                )
                continue
            generated.append(self._build_generated_qa(pt, question, answer, chunks_used))

        total = len(prepared)
        n_ok = len(generated)
        n_fail = total - n_ok
        if n_fail > 0:
            summary = (
                f"Generation: {n_ok}/{total} parsed OK"
                f" | {n_api_fail} API failures"
                f" | {n_empty} empty responses"
                f" | {n_parse_fail} parse failures"
            )
            logger.warning(summary)
            # Also print so it's visible in notebooks where logging may be off
            from tqdm.auto import tqdm as _tqdm

            _tqdm.write(f"  ⚠ {summary}")
        return generated

    @staticmethod
    def _build_generated_qa(
        pt: _PreparedTask,
        question: str,
        answer: str,
        chunks_used: list[int] | None = None,
    ) -> GeneratedQA:
        """Build a GeneratedQA from a prepared task and parsed question/answer."""
        anchor = pt.anchor
        reference_chunks = [_chunk_to_reference_chunk(anchor.primary_chunk)]
        reference_chunks.extend(
            _chunk_to_reference_chunk(chunk) for chunk in anchor.secondary_chunks
        )

        qa_point: QADataPoint = {
            "question": question,
            "answer": answer,
            "reference_chunks": reference_chunks,
            "qa_type": pt.resolved_qa_type,
            "min_hop_count": pt.resolved_hop_count,
            "is_co_located": None,
            "filter_status": None,
            "filter_reasoning": None,
            "no_context_answer": None,
            "eval_scores": {},
        }
        return GeneratedQA(
            qa=qa_point,
            generation_metadata={
                "qa_type_target": pt.resolved_qa_type,
                "target_hop_count": pt.resolved_hop_count,
                "target_hop_count_requested": pt.requested_hop_count,
                "reasoning_mode": pt.task.reasoning_mode,
                "anchor_bundle": anchor,
                "linking_hints": dict(anchor.structural_hints) if anchor.structural_hints else {},
                "generation_mode": "llm_direct",
                "refinement_count": 0,
                "same_seed_refinement_count": 0,
                "task_id": pt.task.task_id,
                "source_task_id": pt.task.source_task_id or pt.task.task_id,
                "regeneration_attempt": pt.task.regeneration_attempt,
                "seed_chunk_id": pt.task.seed_chunk_id,
                "generator_chunks_used": chunks_used,
            },
        )
