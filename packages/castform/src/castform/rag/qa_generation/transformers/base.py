"""Base question transformer with shared metadata/stats handling."""

from __future__ import annotations

import logging
from typing import Any

from castform.rag.qa_generation.pipeline_config import PipelineContext
from castform.rag.qa_generation.generated_qa import GeneratedQA
from castform.rag.qa_generation.style_controls import classify_query_style

logger = logging.getLogger(__name__)


class BaseQuestionTransformer:
    """Annotation-only transformer: classifies query style and records metadata."""

    stats_mode = "base"

    def transform(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]:
        stats = self._stats_bucket(context)
        for item in items:
            stats["processed"] = int(stats.get("processed", 0)) + 1
            original_question = str(item.qa.get("question", "")).strip()

            try:
                candidate_question, step_meta = self._transform_question(
                    item,
                    context=context,
                    original_question=original_question,
                )
            except Exception:
                logger.exception("Question transformation failed for mode '%s'.", self.stats_mode)
                stats["errors"] = int(stats.get("errors", 0)) + 1
                candidate_question = original_question
                step_meta = {"error": "transform_exception"}

            candidate_question = str(candidate_question or "").strip() or original_question
            changed = bool(candidate_question and candidate_question != original_question)
            if changed:
                self._set_question_fields(item, candidate_question)
                stats["mutated"] = int(stats.get("mutated", 0)) + 1
            else:
                stats["unchanged"] = int(stats.get("unchanged", 0)) + 1

            final_question = str(item.qa.get("question", "")).strip()
            observed_style = self._annotate_observed_style(item, final_question)
            self._record_step_metadata(
                item,
                changed=changed,
                style_observed=observed_style,
                original_question=original_question,
                final_question=final_question,
                step_meta=step_meta,
            )
        return items

    def _transform_question(
        self,
        item: GeneratedQA,
        *,
        context: PipelineContext,
        original_question: str,
    ) -> tuple[str, dict[str, Any]]:
        del item, context
        return original_question, {}

    def _stats_bucket(self, context: PipelineContext) -> dict[str, Any]:
        stats = context.setdefault("transformation_stats", {})
        if not isinstance(stats, dict):
            stats = {}
            context["transformation_stats"] = stats
        mode_stats = stats.setdefault(self.stats_mode, {})
        if not isinstance(mode_stats, dict):
            mode_stats = {}
            stats[self.stats_mode] = mode_stats
        mode_stats.setdefault("processed", 0)
        mode_stats.setdefault("mutated", 0)
        mode_stats.setdefault("unchanged", 0)
        mode_stats.setdefault("errors", 0)
        return mode_stats

    @staticmethod
    def _set_question_fields(item: GeneratedQA, question: str) -> None:
        item.qa["question"] = question
        item.qa["user_question"] = question

    @staticmethod
    def _annotate_observed_style(item: GeneratedQA, question: str) -> str:
        observed = classify_query_style(question)
        eval_scores = dict(item.qa.get("eval_scores", {}) or {})
        eval_scores.pop("query_style_target", None)
        eval_scores["query_style_observed"] = observed
        item.qa["eval_scores"] = eval_scores
        item.qa["style_observed"] = observed
        return observed

    def _record_step_metadata(
        self,
        item: GeneratedQA,
        *,
        changed: bool,
        style_observed: str,
        original_question: str,
        final_question: str,
        step_meta: dict[str, Any],
    ) -> None:
        transform_meta = dict(item.generation_metadata.get("transformation", {}) or {})
        steps = list(transform_meta.get("steps", []) or [])
        entry: dict[str, Any] = {
            "mode": self.stats_mode,
            "changed": changed,
            "style_observed": style_observed,
        }
        if step_meta:
            entry.update(step_meta)
        if original_question:
            entry["original_question"] = original_question
        if final_question:
            entry["final_question"] = final_question
        steps.append(entry)
        transform_meta["steps"] = steps
        transform_meta["style_observed"] = style_observed
        transform_meta["changed"] = any(bool(step.get("changed")) for step in steps)
        item.generation_metadata["transformation"] = transform_meta
