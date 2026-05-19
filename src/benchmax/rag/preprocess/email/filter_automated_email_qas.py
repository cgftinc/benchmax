"""LLM-based post-filter to remove QA pairs grounded in automated emails.

This is intended as a final curation pass after QA generation, e.g.:
- calendar invite updates
- Zoom invites/meeting notices
- payment app notifications (Venmo, etc.)
- git/bot/system notifications
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmax.rag.qa_generation.batch_processor import batch_process_sync
from benchmax.rag.qa_generation.storage import (
    load_qa_dataset_jsonl,
    save_qa_dataset,
    save_qa_dataset_jsonl,
)

FILTER_SYSTEM_PROMPT = """You are a strict dataset curator.

Task: Determine whether the references that generated a question-answer datapoint is from an automated/system email.

Treat as automated/system when evidence is mostly from:
- calendar invite/update/cancellation emails
- Zoom/Meet scheduling or join-link notices
- transactional notifications (Venmo/payment receipts, sign-in codes, verification codes)
- git/bot/devops notification emails
- no-reply marketing/product lifecycle notifications

Treat as NOT automated when the primary evidence is a real human conversation,
even if it references tools like Zoom, GitHub, calendar, or Venmo.

Drop if it's from an automated email, keep if it's not.

Return JSON only with:
{
  "drop": true | false,
  "confidence": number,   // 0.0 to 1.0
  "reason": "short reason"
}
"""


@dataclass
class FilterDecision:
    drop: bool
    confidence: float
    reason: str


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _build_prompt(dp: dict[str, Any], max_chunk_chars: int) -> str:
    refs = dp.get("reference_chunks", [])
    ref_blocks: list[str] = []
    for i, ref in enumerate(refs[:3], start=1):
        metadata = ref.get("metadata", {}) if isinstance(ref, dict) else {}
        subject = metadata.get("subject", "")
        thread_id = metadata.get("thread_id", "")
        participants = metadata.get("participants", [])
        content = ref.get("content", "") if isinstance(ref, dict) else ""
        ref_blocks.append(
            f"[Reference {i}]\n"
            f"thread_id: {thread_id}\n"
            f"subject: {subject}\n"
            f"participants: {participants}\n"
            f"content_excerpt:\n{_clip(str(content), max_chunk_chars)}"
        )

    return (
        f"Question:\n{dp.get('question', '')}\n\n"
        f"Answer:\n{dp.get('answer', '')}\n\n"
        f"QA Type: {dp.get('qa_type', '')}\n\n"
        f"References:\n\n" + "\n\n".join(ref_blocks)
    )


def _parse_decision(raw: str) -> FilterDecision | None:
    if not raw:
        return None
    text = raw.strip()
    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        drop = bool(obj.get("drop", False))
        confidence_raw = obj.get("confidence", 0.0)
        reason = str(obj.get("reason", "")).strip()
        try:
            confidence = float(confidence_raw)
        except (ValueError, TypeError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        return FilterDecision(drop=drop, confidence=confidence, reason=reason)
    return None


def _filter_dataset(
    dataset: list[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    max_concurrent: int,
    max_chunk_chars: int,
    min_drop_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [_build_prompt(dp, max_chunk_chars=max_chunk_chars) for dp in dataset]

    batch = batch_process_sync(
        client=client,
        model=model,
        prompts=prompts,
        system_prompt=FILTER_SYSTEM_PROMPT,
        max_tokens=120,
        timeout=60.0,
        max_concurrent=max_concurrent,
        show_progress=True,
    )

    kept: list[dict[str, Any]] = []
    dropped = 0
    parse_failures = 0
    low_conf_drop_reversed = 0
    reasons: dict[str, int] = {}

    for dp, resp in zip(dataset, batch.responses, strict=False):
        if resp is None:
            parse_failures += 1
            kept.append(dp)
            continue
        decision = _parse_decision(resp.answer)
        if decision is None:
            parse_failures += 1
            kept.append(dp)
            continue

        should_drop = decision.drop
        if should_drop and decision.confidence < min_drop_confidence:
            should_drop = False
            low_conf_drop_reversed += 1

        if should_drop:
            dropped += 1
            reason_key = decision.reason or "unspecified"
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        else:
            kept.append(dp)

    report = {
        "total": len(dataset),
        "kept": len(kept),
        "dropped": dropped,
        "drop_rate": (dropped / len(dataset)) if dataset else 0.0,
        "parse_failures_kept": parse_failures,
        "low_confidence_drops_kept": low_conf_drop_reversed,
        "drop_reasons_top": sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    return kept, report


def _output_jsonl_path(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    if output_dir is None:
        return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def run_for_file(
    input_jsonl: Path,
    *,
    output_dir: Path | None,
    output_suffix: str,
    client: OpenAI,
    model: str,
    max_concurrent: int,
    max_chunk_chars: int,
    min_drop_confidence: float,
) -> dict[str, Any]:
    dataset = load_qa_dataset_jsonl(input_jsonl)
    filtered, report = _filter_dataset(
        dataset,
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        max_chunk_chars=max_chunk_chars,
        min_drop_confidence=min_drop_confidence,
    )

    out_jsonl = _output_jsonl_path(input_jsonl, output_dir, output_suffix)
    out_yaml = out_jsonl.with_suffix(".yaml")
    save_qa_dataset_jsonl(filtered, out_jsonl)
    save_qa_dataset(filtered, out_yaml)

    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(out_jsonl),
        "output_yaml": str(out_yaml),
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drop QA pairs grounded in automated/system emails.")
    p.add_argument(
        "--input-jsonl",
        nargs="+",
        required=True,
        help="One or more dataset JSONL paths (e.g., train/eval).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. If empty, writes next to input files.",
    )
    p.add_argument(
        "--output-suffix",
        default="_filtered",
        help="Suffix added to output filename stem.",
    )
    p.add_argument("--model", required=True, help="LLM model name.")
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", config.llm_url()),
        help="OpenAI-compatible base URL.",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (or set OPENAI_API_KEY).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent LLM calls.",
    )
    p.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1400,
        help="Max chars per reference chunk excerpt sent to classifier.",
    )
    p.add_argument(
        "--min-drop-confidence",
        type=float,
        default=0.6,
        help="Only drop rows when model confidence >= this value.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    all_results: list[dict[str, Any]] = []
    for input_path in args.input_jsonl:
        result = run_for_file(
            Path(input_path).expanduser(),
            output_dir=output_dir,
            output_suffix=args.output_suffix,
            client=client,
            model=args.model,
            max_concurrent=args.max_concurrent,
            max_chunk_chars=args.max_chunk_chars,
            min_drop_confidence=args.min_drop_confidence,
        )
        all_results.append(result)

    print("\nAutomated-email QA filter summary")
    for item in all_results:
        report = item["report"]
        print(f"\nInput:  {item['input_jsonl']}")
        print(f"Output: {item['output_jsonl']}")
        print(f"YAML:   {item['output_yaml']}")
        print(
            f"Kept {report['kept']}/{report['total']} "
            f"(dropped {report['dropped']}, {report['drop_rate']:.1%})"
        )
        print(
            f"Parse failures kept: {report['parse_failures_kept']} | "
            f"Low-confidence drops kept: {report['low_confidence_drops_kept']}"
        )
        if report["drop_reasons_top"]:
            print("Top drop reasons:")
            for reason, count in report["drop_reasons_top"]:
                print(f"  - {count}: {reason}")


if __name__ == "__main__":
    main()
