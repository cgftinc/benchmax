"""User-facing traces pipeline — normalised traces → training examples.

Configure, call ``.run()``, get ``{"train_dataset", "eval_dataset",
"stats"}`` back.  Takes already-normalised traces as input — fetching
lives in the ``TraceAdapter`` implementations under
``castform.traces.<provider>``.

Simple filters are scalars/bools; only the LLM-backed importance filter
earns its own dataclass.  Stages run in fixed order — use
``apply_filters`` directly for custom ordering.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from castform.traces.adapter import NormalizedTrace
from castform.traces.processing import (
    MIN_TRAIN_SAMPLES,
    apply_filters,
    build_training_examples,
    detect_system_prompt,
    detect_tools,
    split_dataset,
)

logger = logging.getLogger(__name__)


def _print_progress(message: str, *, verbose: bool) -> None:
    if verbose:
        tqdm.write(message)


@dataclass
class ImportanceFilterConfig:
    """LLM-driven importance classification."""

    llm_client: Any
    model: str = "gpt-5.4-nano"
    min_importance_score: float = 0.6
    anchor_fraction: float = 0.1
    max_concurrency: int = 20
    checkpoint_dir: Path | None = None


@dataclass
class TracesPipeline:
    """Convert normalised traces into train/eval training examples."""

    traces: list[NormalizedTrace]
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    min_completion_chars: int | None = 40
    max_tool_output_overlap: float | None = 0.8
    excluded_tools: list[str] = field(default_factory=list)
    max_example_similarity: float | None = 0.85
    importance_filter: ImportanceFilterConfig | None = None
    max_examples: int = 1000
    train_fraction: float = 0.9
    random_seed: int = 42
    output_dir: str | Path | None = None
    verbose: bool = True

    def run(self) -> dict[str, Any]:
        """Execute the pipeline.  Returns ``{train_dataset, eval_dataset, stats}``."""
        v = self.verbose

        _print_progress(
            f"[1/6] Detecting system prompt + tools across {len(self.traces)} traces...",
            verbose=v,
        )
        detected_sp = detect_system_prompt(self.traces) if self.system_prompt is None else None
        sp_text = (
            self.system_prompt
            if self.system_prompt is not None
            else (detected_sp.prompt if detected_sp else "")
        )
        detected_tools = detect_tools(self.traces) if self.tools is None else None
        if detected_sp is not None:
            _print_progress(
                f"[1/6] Detected system prompt in {detected_sp.count}/{detected_sp.total_traces} "
                f"traces ({len(detected_sp.variants)} variants)",
                verbose=v,
            )
        elif self.system_prompt is not None:
            _print_progress(
                f"[1/6] Using explicit system_prompt (len={len(self.system_prompt)})",
                verbose=v,
            )
        if detected_tools is not None:
            tool_names = [t.name for t in detected_tools.tools]
            _print_progress(
                f"[1/6] Detected {len(tool_names)} tool(s): {', '.join(tool_names) or '(none)'}",
                verbose=v,
            )

        _print_progress("[2/6] Building training examples from traces...", verbose=v)
        examples = build_training_examples(self.traces, include_system_prompt=False)
        _print_progress(
            f"[2/6] Built {len(examples)} training examples "
            f"(avg {len(examples) / max(len(self.traces), 1):.1f} per trace)",
            verbose=v,
        )

        stages: list[tuple[str, dict[str, Any]]] = []
        if self.min_completion_chars is not None:
            stages.append(("heuristic", {"min_completion_chars": self.min_completion_chars}))
        if self.max_tool_output_overlap is not None:
            stages.append(("tool_relay", {"overlap_threshold": self.max_tool_output_overlap}))
        if self.excluded_tools:
            stages.append(("tool_calls", {"exclude_tools": list(self.excluded_tools)}))
        if self.max_example_similarity is not None:
            stages.append(("dedup", {"similarity_threshold": self.max_example_similarity}))

        _print_progress(
            f"[3/6] Applying {len(stages)} filter stage(s): "
            f"{', '.join(name for name, _ in stages) or '(none)'}",
            verbose=v,
        )
        filter_result = apply_filters(examples, stages)
        kept = filter_result.kept
        _print_progress(
            f"[3/6] Filters kept {len(kept)}/{len(examples)} examples "
            f"({len(filter_result.dropped)} dropped)",
            verbose=v,
        )

        importance_stats: dict[str, Any] | None = None
        if self.importance_filter is not None:
            from castform.traces.pivot import apply_pivot_filter_async

            pre = len(kept)
            cfg = self.importance_filter
            _print_progress(
                f"[4/6] Running importance filter on {pre} examples "
                f"(model={cfg.model}, min_score={cfg.min_importance_score}, "
                f"concurrency={cfg.max_concurrency})...",
                verbose=v,
            )

            pbar = tqdm(total=pre, desc="importance filter", disable=not v, unit="turn")

            def _on_progress(completed: int, total: int) -> None:
                pbar.total = total
                pbar.n = completed
                pbar.refresh()

            try:
                result = asyncio.run(
                    apply_pivot_filter_async(
                        examples=kept,
                        traces=self.traces,
                        llm_client=cfg.llm_client,
                        model=cfg.model,
                        threshold=cfg.min_importance_score,
                        anchor_fraction=cfg.anchor_fraction,
                        max_concurrency=cfg.max_concurrency,
                        checkpoint_dir=cfg.checkpoint_dir,
                        progress_callback=_on_progress,
                    )
                )
            finally:
                pbar.close()

            kept = result.kept
            importance_stats = {
                "pre": pre,
                "post": len(kept),
                "min_importance_score": cfg.min_importance_score,
            }
            _print_progress(
                f"[4/6] Importance filter kept {len(kept)}/{pre} examples",
                verbose=v,
            )
        else:
            _print_progress("[4/6] Skipping importance filter (not configured)", verbose=v)

        pre_cap = len(kept)
        if self.max_examples and len(kept) > self.max_examples:
            import random

            rng = random.Random(self.random_seed)
            kept = rng.sample(kept, self.max_examples)
            _print_progress(
                f"[5/6] Capped to max_examples={self.max_examples} (sampled from {pre_cap})",
                verbose=v,
            )
        else:
            _print_progress(
                f"[5/6] No cap needed ({pre_cap} ≤ max_examples={self.max_examples})",
                verbose=v,
            )

        if len(kept) < MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Only {len(kept)} examples survived filtering; need at least "
                f"{MIN_TRAIN_SAMPLES}.  Loosen filters, or lower thresholds, "
                f"or fetch more traces."
            )
        train_count = max(int(round(len(kept) * self.train_fraction)), MIN_TRAIN_SAMPLES)
        eval_count = len(kept) - train_count
        train_data, eval_data = split_dataset(kept, train_count, eval_count, seed=self.random_seed)
        _print_progress(
            f"[6/6] Split complete: {len(train_data)} train, {len(eval_data)} eval",
            verbose=v,
        )

        stats = {
            "total_traces": len(self.traces),
            "examples_built": len(examples),
            "kept_after_filters": len(filter_result.kept),
            "final_kept": len(kept),
            "train_count": len(train_data),
            "eval_count": len(eval_data),
            "detected_system_prompt": (
                {
                    "prompt": detected_sp.prompt,
                    "count": detected_sp.count,
                    "total_traces": detected_sp.total_traces,
                    "n_variants": len(detected_sp.variants),
                }
                if detected_sp
                else None
            ),
            "detected_tools": (
                [{"name": t.name, "call_count": t.call_count} for t in detected_tools.tools]
                if detected_tools
                else None
            ),
            "importance_filter": importance_stats,
        }

        if self.output_dir:
            out = Path(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(out / "train.jsonl", train_data)
            _write_jsonl(out / "eval.jsonl", eval_data)
            with (out / "metadata.json").open("w") as f:
                json.dump(stats, f, indent=2, default=str)
            _print_progress(
                f"Wrote {len(train_data)} train + {len(eval_data)} eval to {out}",
                verbose=v,
            )

        return {
            "train_dataset": train_data,
            "eval_dataset": eval_data,
            "stats": stats,
            "system_prompt": sp_text,
            "tools": (
                self.tools
                if self.tools is not None
                else [
                    {"name": t.name, "param_keys": sorted(list(t.param_keys))}
                    for t in (detected_tools.tools if detected_tools else [])
                ]
            ),
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
