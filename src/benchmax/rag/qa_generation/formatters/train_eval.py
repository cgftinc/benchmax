"""Train/eval formatter for Pipeline outputs."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmax.rag.qa_generation.pipeline_config import PipelineContext, OutputConfig, SplitConfig
from benchmax.rag.qa_generation.generated_qa import GeneratedQA


class TrainEvalFormatter:
    """Writes stratified train/eval JSONL artifacts."""

    def __init__(self, *, output_cfg: OutputConfig, split_cfg: SplitConfig) -> None:
        self.output_cfg = output_cfg
        self.split_cfg = split_cfg

    def format(self, items: list[GeneratedQA], context: PipelineContext) -> dict[str, Any]:
        rows = [self._to_row(item) for item in items]
        train_rows, eval_rows = self._stratified_split(rows)

        output_dir = Path(self.output_cfg.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / self.output_cfg.train_jsonl
        eval_path = output_dir / self.output_cfg.eval_jsonl

        with train_path.open("w", encoding="utf-8") as fh:
            for row in train_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with eval_path.open("w", encoding="utf-8") as fh:
            for row in eval_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "train_dataset": train_rows,
            "eval_dataset": eval_rows,
            "stats": {
                "total": len(rows),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "train_ratio": self.split_cfg.train_ratio,
                "stratify_by": list(self.split_cfg.stratify_by),
            },
            "output_paths": {
                "train_jsonl": str(train_path),
                "eval_jsonl": str(eval_path),
            },
            "rejected_dataset": context.get("rejected_items", []),
        }

    def _to_row(self, item: GeneratedQA) -> dict[str, Any]:
        qa_type = item.qa.get(
            "qa_type",
            item.generation_metadata.get("qa_type_target", "unknown"),
        )
        row: dict[str, Any] = {
            "question": item.qa.get("question", ""),
            "answer": item.qa.get("answer", ""),
            "qa_type": qa_type,
            "reference_chunks": item.qa.get("reference_chunks", []),
        }
        # Include reasoning_mode if present.
        reasoning_mode = item.generation_metadata.get("reasoning_mode", "")
        if reasoning_mode:
            row["reasoning_mode"] = reasoning_mode
        # Include difficulty_score if available from hop-count validity filter.
        if item.filter_verdict and item.filter_verdict.metadata:
            difficulty = item.filter_verdict.metadata.get("difficulty_score")
            if difficulty is not None:
                row["difficulty_score"] = difficulty
        # Include eval_scores if populated by the scoring system.
        eval_scores = item.qa.get("eval_scores")
        if eval_scores:
            row["eval_scores"] = eval_scores
        # Include linking hints for confidence analysis.
        linking_hints = item.generation_metadata.get("linking_hints")
        if linking_hints:
            row["linking_hints"] = linking_hints
        # Include unanswerable-pipeline fields when present.
        answerability = item.qa.get("answerability")
        if answerability is not None:
            row["answerability"] = answerability
        nearest_chunks = item.qa.get("nearest_chunks")
        if nearest_chunks is not None:
            row["nearest_chunks"] = nearest_chunks
        perturbation_type = item.qa.get("perturbation_type")
        if perturbation_type is not None:
            row["perturbation_type"] = perturbation_type
        return row

    def _stratified_split(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rng = random.Random(self.split_cfg.seed)
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            qa_type = str(row.get("qa_type", "unknown"))
            style = (
                str(row.get("style_observed", "")).strip()
                or str(row.get("style_target", "unknown")).strip()
            )
            answerability = str(row.get("answerability", "fully_answerable"))
            buckets[(qa_type, style, answerability)].append(row)

        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for bucket_rows in buckets.values():
            rng.shuffle(bucket_rows)
            split_idx = int(len(bucket_rows) * self.split_cfg.train_ratio)
            if split_idx == 0 and len(bucket_rows) > 1:
                split_idx = 1
            elif split_idx >= len(bucket_rows) and len(bucket_rows) > 1:
                split_idx = len(bucket_rows) - 1
            train_rows.extend(bucket_rows[:split_idx])
            eval_rows.extend(bucket_rows[split_idx:])

        rng.shuffle(train_rows)
        rng.shuffle(eval_rows)
        return train_rows, eval_rows
