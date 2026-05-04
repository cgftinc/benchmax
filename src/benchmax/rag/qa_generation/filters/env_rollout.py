"""Rollout-based evaluator filter for Pipeline."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from benchmax.rag.qa_generation.pipeline_config import PipelineContext, LLMEnvFilterConfig
from benchmax.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA
from benchmax.platform.client import RolloutClient

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

_JUDGE_SYSTEM = """\
Determine whether two answers are semantically equivalent.
Return JSON only:
{"is_equivalent": <bool>, "confidence": <float 0-1>, "reasoning": "<short reason>"}"""

_JUDGE_USER = """\
Reference answer:
{reference}

Candidate answer:
{candidate}

Are they equivalent?"""


def _parse_answer(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    match = _ANSWER_RE.search(value)
    if match:
        return match.group(1).strip()
    return value


def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


class EnvRolloutFilter:
    """RolloutClient-backed evaluator using env tools + LLM equivalence judge."""

    def __init__(
        self,
        *,
        rollout_client: RolloutClient,
        cfg: LLMEnvFilterConfig,
    ) -> None:
        self.rollout_client = rollout_client
        self.cfg = cfg
        self.judge_client = OpenAI(
            api_key=cfg.judge_api_key,
            base_url=cfg.judge_base_url,
        )

    def _run_rollout(self, raw_example: dict[str, Any], *, idx: int) -> dict[str, Any]:
        env_bundle = self.cfg.env_bundle
        limits = self.cfg.rollout_limits
        cls_bytes, meta_bytes = env_bundle.as_bytes_bundle()
        if env_bundle.has_paths():
            return self.rollout_client.stream_rollout(
                raw_example=raw_example,
                env_cls_path=env_bundle.env_cls_path,
                env_metadata_path=env_bundle.env_metadata_path,
                llm_model=self.cfg.model,
                llm_base_url=self.cfg.base_url,
                llm_api_key=self.cfg.api_key,
                max_turns=limits.max_turns,
                max_tool_calls=limits.max_tool_calls,
                max_completion_tokens=limits.max_completion_tokens,
                capture_messages=True,
                include_event_meta=False,
                example_index=idx,
            )

        if cls_bytes is None or meta_bytes is None:
            raise ValueError("LLM env filter requires env bundle paths or env bundle files.")
        return self.rollout_client.stream_rollout(
            raw_example=raw_example,
            env_cls_bytes=cls_bytes,
            env_metadata_bytes=meta_bytes,
            llm_model=self.cfg.model,
            llm_base_url=self.cfg.base_url,
            llm_api_key=self.cfg.api_key,
            max_turns=limits.max_turns,
            max_tool_calls=limits.max_tool_calls,
            max_completion_tokens=limits.max_completion_tokens,
            capture_messages=True,
            include_event_meta=False,
            example_index=idx,
        )

    def evaluate(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        max_refinements = context.config.refinement.max_refinements_per_item
        stats = context.setdefault(
            "env_filter_stats",
            {"passed": 0, "needs_refinement": 0, "rejected": 0, "errors": 0},
        )

        for idx, item in enumerate(items):
            if item.filter_verdict is not None and not item.is_passed:
                continue

            try:
                verdict = self._evaluate_item(item, idx=idx, max_refinements=max_refinements)
            except Exception:
                logger.exception("EnvRolloutFilter failed for one item")
                verdict = FilterVerdict(
                    status="passed",
                    reason="env_filter_error",
                    reasoning="Filter error; passed by default.",
                    metadata={
                        "filter_mode": "llm_env",
                        "reason_code": "filter_error",
                        "confidence": 0.0,
                        "retrieval_query": str(item.qa.get("question", "")),
                        "ref_overlap_ratio": None,
                        "feedback_type": None,
                        "refinement_hint": None,
                    },
                )
                stats["errors"] = int(stats.get("errors", 0)) + 1

            item.filter_verdict = verdict
            if verdict.status == "passed":
                stats["passed"] = int(stats.get("passed", 0)) + 1
            elif verdict.status == "needs_refinement":
                stats["needs_refinement"] = int(stats.get("needs_refinement", 0)) + 1
            else:
                stats["rejected"] = int(stats.get("rejected", 0)) + 1
        return items

    def _evaluate_item(self, item: GeneratedQA, *, idx: int, max_refinements: int) -> FilterVerdict:
        question = str(item.qa.get("question", "")).strip()
        answer = str(item.qa.get("answer", "")).strip()
        if not question:
            return FilterVerdict(
                status="rejected",
                reason="env_filter_rejected",
                reasoning="Missing question.",
                metadata={
                    "filter_mode": "llm_env",
                    "reason_code": "missing_question",
                    "confidence": 1.0,
                    "retrieval_query": "",
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": None,
                },
            )

        result = self._run_rollout(
            {
                "prompt": (
                    "Answer the question with tool use as needed. "
                    "Return final answer in <answer>...</answer>.\n\n"
                    f"Question: {question}"
                )
            },
            idx=idx,
        )
        candidate_answer = _parse_answer(
            str(result.get("final_assistant_text") or result.get("trace") or "")
        )
        tool_calls = _count_tool_calls(result.get("messages", []) or [])
        target_hops = int(item.generation_metadata.get("target_hop_count", 1))
        equivalent, confidence, reasoning = self._judge_equivalence(
            reference=answer,
            candidate=candidate_answer,
        )
        refinements = int(item.generation_metadata.get("refinement_count", 0))

        metadata = {
            "filter_mode": "llm_env",
            "reason_code": "env_filter_result",
            "confidence": confidence,
            "retrieval_query": question,
            "ref_overlap_ratio": None,
            "feedback_type": None,
            "refinement_hint": None,
            "tool_calls": tool_calls,
            "target_hop_count": target_hops,
            "candidate_answer": candidate_answer,
            "judge_reasoning": reasoning,
            "rollout_result": result,
        }

        if equivalent and tool_calls >= target_hops:
            metadata["reason_code"] = "env_filter_passed"
            return FilterVerdict(
                status="passed",
                reason="env_filter_passed",
                reasoning="Equivalent answer found with sufficient tool-call depth.",
                metadata=metadata,
            )

        if equivalent:
            metadata["reason_code"] = "too_easy"
            metadata["feedback_type"] = "same_anchor_feedback"
            metadata["refinement_hint"] = "Increase difficulty or retrieval depth."
            if refinements < max_refinements:
                return FilterVerdict(
                    status="needs_refinement",
                    reason="env_filter_needs_refinement",
                    reasoning=(
                        f"Equivalent answer found but only {tool_calls} tool calls "
                        f"(target={target_hops})."
                    ),
                    metadata=metadata,
                )
            return FilterVerdict(
                status="rejected",
                reason="env_filter_rejected",
                reasoning="Too easy and refinement budget exhausted.",
                metadata=metadata,
            )

        metadata["reason_code"] = "incorrect_or_unresolved"
        metadata["feedback_type"] = "same_anchor_feedback"
        metadata["refinement_hint"] = "Improve clarity or answerability."
        if refinements < max_refinements:
            return FilterVerdict(
                status="needs_refinement",
                reason="env_filter_needs_refinement",
                reasoning="Rollout answer is not equivalent to the reference answer.",
                metadata=metadata,
            )
        return FilterVerdict(
            status="rejected",
            reason="env_filter_rejected",
            reasoning="Unresolved after refinement budget exhausted.",
            metadata=metadata,
        )

    def _judge_equivalence(self, *, reference: str, candidate: str) -> tuple[bool, float, str]:
        if not candidate:
            return False, 0.0, "No candidate answer."
        response = self.judge_client.chat.completions.create(
            model=self.cfg.judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _JUDGE_USER.format(reference=reference, candidate=candidate),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            payload = json.loads(raw)
            return (
                bool(payload.get("is_equivalent", False)),
                float(payload.get("confidence", 0.0) or 0.0),
                str(payload.get("reasoning", "")).strip(),
            )
        except json.JSONDecodeError:
            logger.warning("Env filter judge parse failure: %s", raw[:240])
            return False, 0.0, "parse_error"
