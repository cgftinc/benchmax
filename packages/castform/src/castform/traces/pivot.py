"""Pivot filtering — LLM-based turn importance classification.

Identifies critical decision points in agentic traces using counterfactual
reasoning.  Based on PivotRL (arXiv:2603.21383): ~71% of turns produce zero
gradient during RL training.  Filtering to pivots yields +5.37 over SFT
with 4x fewer rollout turns.

Call strategy: 1 LLM call per turn (independent per-turn estimation), with
the full trace as context.  Uses gpt-5.4-mini by default — this is a filtering
step, not a reward signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from castform.utils.checkpoint import CheckpointBase
from castform.traces.adapter import NormalizedTrace, TraceMessage
from castform.traces.processing import MIN_TRAIN_SAMPLES, TrainingExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

_VALID_CATEGORIES = {"pivot", "downstream", "trivial"}


@dataclass
class PivotRating:
    """Per-turn importance rating from counterfactual analysis."""

    trace_id: str
    turn_index: int
    importance_score: float  # 0.0–1.0
    category: Literal["pivot", "downstream", "trivial"]
    reasoning: str

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            self.category = "downstream"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "turn_index": self.turn_index,
            "importance_score": self.importance_score,
            "category": self.category,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PivotRating:
        return cls(
            trace_id=d["trace_id"],
            turn_index=d["turn_index"],
            importance_score=float(d["importance_score"]),
            category=d["category"],
            reasoning=d["reasoning"],
        )


@dataclass
class PivotFilterResult:
    """Result of pivot filtering."""

    kept: list[TrainingExample]
    dropped: list[TrainingExample]
    ratings: list[PivotRating]

    @property
    def summary(self) -> dict[str, Any]:
        cats = {"pivot": 0, "downstream": 0, "trivial": 0}
        for r in self.ratings:
            cats[r.category] = cats.get(r.category, 0) + 1
        total = len(self.kept) + len(self.dropped)
        return {
            "categories": cats,
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "retention_rate": len(self.kept) / total if total else 0,
        }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class PivotCheckpointManager(CheckpointBase):
    """Incremental checkpoint for LLM pivot ratings.

    Appends ratings to a JSONL file as they complete.  On resume, loads
    cached ratings and skips already-rated turns.  Only the ``model``
    invalidates the cache — changing ``threshold`` or upstream filters
    reuses cached ratings.
    """

    _RATINGS_FILE = "pivot_ratings.jsonl"

    def __init__(self, checkpoint_dir: Path, model: str) -> None:
        super().__init__(checkpoint_dir)
        self.model = model

    def resume(self) -> dict[tuple[str, int], PivotRating]:
        """Load cached LLM ratings.  Returns empty dict if none or model changed."""
        manifest = self.load_manifest()
        if manifest is None:
            return {}

        if manifest.get("model") != self.model:
            logger.warning(
                "Pivot checkpoint model mismatch (cached=%s, current=%s) — clearing",
                manifest.get("model"),
                self.model,
            )
            self.cleanup()
            return {}

        cached: dict[tuple[str, int], PivotRating] = {}
        for row in self.load_jsonl(self._RATINGS_FILE):
            try:
                r = PivotRating.from_dict(row)
                cached[(r.trace_id, r.turn_index)] = r
            except KeyError:
                break

        logger.info("Pivot checkpoint: loaded %d cached ratings", len(cached))
        return cached

    def save_batch(self, ratings: list[PivotRating]) -> None:
        """Append a batch of ratings to the checkpoint JSONL."""
        self.append_jsonl(self._RATINGS_FILE, [r.to_dict() for r in ratings])
        self.save_manifest({"model": self.model})


# ---------------------------------------------------------------------------
# Heuristic fast path
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|welcome|greetings)\b",
    re.IGNORECASE,
)

_AUTH_ASK_RE = re.compile(
    r"\b(provide|share|give|tell)\b.*\b(name|email|order\s*(id|number)|zip|account)\b",
    re.IGNORECASE,
)


def _is_heuristic_trivial(
    ex: TrainingExample,
    turn_index_in_trace: int,
) -> bool:
    """Fast-path: detect obviously trivial turns without an LLM call.

    Returns True for:
    - First assistant turn that's a greeting with no tool calls
    - Turns that just ask for identity/auth info with no tool calls
    - Single lookup tool call to a get_*/find_* function
    """
    if not ex.completion_messages:
        return True

    first_completion = ex.completion_messages[0]

    # Greeting as first turn
    if turn_index_in_trace == 0 and not first_completion.tool_calls:
        if _GREETING_RE.search(first_completion.content):
            return True

    # Auth verification ask (no tool calls)
    if not first_completion.tool_calls and _AUTH_ASK_RE.search(first_completion.content):
        if len(first_completion.content) < 200:
            return True

    # Single lookup tool call (get_*, find_*)
    if first_completion.tool_calls and len(first_completion.tool_calls) == 1:
        tc = first_completion.tool_calls[0]
        if tc.name.startswith(("get_", "find_", "lookup_", "fetch_")):
            # Only if content is empty or very short (just calling the tool)
            if len(first_completion.content) < 50:
                return True

    return False


# ---------------------------------------------------------------------------
# LLM counterfactual analysis
# ---------------------------------------------------------------------------

_PIVOT_PROMPT = """\
You are analyzing an agentic trace to identify critical decision points.

TASK SUMMARY: {outcome_summary}

FULL TRACE:
{trace_messages}

CURRENT TURN (index {turn_index}):
Previous message: {previous_message}
Agent action: {current_turn}

Question: If the agent had made a DIFFERENT decision at this specific turn, \
how likely is it that the final outcome would have changed?

Consider:
- Is this a routine/formulaic action (greeting, auth lookup, confirmation)?
- Does this turn involve a judgment call (which tool to use, what parameters, \
how to interpret ambiguous information)?
- Is this turn mechanically executing a decision already made in a prior turn?
- Could a reasonable alternative action at this point lead to a different outcome?

Respond with JSON only:
{{"importance_score": <float 0.0-1.0>, "category": "pivot" | "downstream" | "trivial", \
"reasoning": "<one sentence>"}}"""


def _format_trace_messages(messages: list[TraceMessage], max_chars: int = 6000) -> str:
    """Render trace messages for the LLM prompt, truncating if needed."""
    parts: list[str] = []
    for m in messages:
        role = m.role.upper()
        content = m.content[:300] if len(m.content) > 300 else m.content
        line = f"[{role}] {content}"
        if m.tool_calls:
            for tc in m.tool_calls:
                line += f"\n  → {tc.name}({tc.arguments[:100]})"
        parts.append(line)

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _build_outcome_summary(trace: NormalizedTrace) -> str:
    """Generate a 1-sentence outcome summary from scores and final messages."""
    parts: list[str] = []

    # Use provider scores if available
    task_success = trace.scores.get("task_success")
    if task_success is not None:
        outcome = "SUCCESS" if task_success >= 0.5 else "FAILED"
        parts.append(f"Outcome: {outcome} (task_success={task_success})")

    # Add last assistant message as context
    last_assistant = None
    for msg in reversed(trace.messages):
        if msg.role == "assistant" and msg.content:
            last_assistant = msg.content[:150]
            break
    if last_assistant:
        parts.append(f"Final response: {last_assistant}")

    return " | ".join(parts) if parts else "Outcome unknown"


async def _rate_single_turn(
    client: Any,
    model: str,
    trace_context: tuple[str, str],
    ex: TrainingExample,
    semaphore: asyncio.Semaphore,
) -> PivotRating:
    """Rate a single turn using LLM counterfactual analysis."""
    trace_text, outcome_summary = trace_context

    # Previous message context
    prev_msg = "(start of conversation)"
    if ex.prompt_messages:
        last = ex.prompt_messages[-1]
        prev_msg = f"[{last.role.upper()}] {last.content[:200]}"

    # Current turn
    current = ""
    if ex.completion_messages:
        c = ex.completion_messages[0]
        current = f"[{c.role.upper()}] {c.content[:300]}"
        if c.tool_calls:
            for tc in c.tool_calls:
                current += f"\n  → {tc.name}({tc.arguments[:100]})"

    prompt = _PIVOT_PROMPT.format(
        outcome_summary=outcome_summary,
        trace_messages=trace_text,
        turn_index=ex.turn_index,
        previous_message=prev_msg,
        current_turn=current,
    )

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                )
                text = str(response.choices[0].message.content).strip()
                # Parse JSON from response (handle markdown fences)
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                data = json.loads(text)

                # Validate category
                cat = data.get("category", "downstream")
                if cat not in _VALID_CATEGORIES:
                    cat = "downstream"

                return PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=max(0.0, min(1.0, float(data["importance_score"]))),
                    category=cat,
                    reasoning=data.get("reasoning", ""),
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                # Parse/validation errors — don't retry, return fallback
                return PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=0.5,
                    category="downstream",
                    reasoning=f"LLM response parse error: {e}",
                )
            except Exception as e:
                # Transient/rate-limit errors — retry with jitter
                if attempt == 4:
                    logger.warning(
                        "Pivot rating failed for trace=%s turn=%d: %s",
                        ex.trace_id,
                        ex.turn_index,
                        e,
                    )
                    return PivotRating(
                        trace_id=ex.trace_id,
                        turn_index=ex.turn_index,
                        importance_score=0.5,
                        category="downstream",
                        reasoning=f"LLM rating failed: {e}",
                    )
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                base_wait = (2**attempt) * (3 if is_rate_limit else 1)
                jitter = base_wait * (0.5 + random.random())
                await asyncio.sleep(jitter)

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_CHECKPOINT_EVERY = 10


async def _rate_all_turns(
    client: Any,
    model: str,
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    max_concurrency: int,
    *,
    checkpoint: PivotCheckpointManager | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    completed_offset: int = 0,
    total_llm: int | None = None,
) -> list[PivotRating]:
    """Rate all non-heuristic-trivial turns via LLM with streaming progress."""
    trace_map = {t.id: t for t in traces}
    semaphore = asyncio.Semaphore(max_concurrency)

    # Pre-compute trace context once per trace (not per turn)
    trace_contexts: dict[str, tuple[str, str]] = {}
    for ex in examples:
        if ex.trace_id not in trace_contexts:
            trace = trace_map.get(ex.trace_id)
            if trace is not None:
                trace_contexts[ex.trace_id] = (
                    _format_trace_messages(trace.messages),
                    _build_outcome_summary(trace),
                )

    total = total_llm if total_llm is not None else len(examples)

    # Launch all tasks — semaphore controls actual concurrency
    tasks: list[asyncio.Task[PivotRating]] = []
    for ex in examples:
        ctx = trace_contexts.get(ex.trace_id)
        if ctx is not None:
            tasks.append(
                asyncio.create_task(_rate_single_turn(client, model, ctx, ex, semaphore))
            )

    # Stream results as they complete
    results: list[PivotRating] = []
    checkpoint_batch: list[PivotRating] = []

    for coro in asyncio.as_completed(tasks):
        rating = await coro
        results.append(rating)
        checkpoint_batch.append(rating)

        if len(checkpoint_batch) >= _CHECKPOINT_EVERY:
            if checkpoint is not None:
                checkpoint.save_batch(checkpoint_batch)
            checkpoint_batch = []

        if progress_callback is not None:
            progress_callback(completed_offset + len(results), total)

    # Flush remaining checkpoint batch
    if checkpoint_batch and checkpoint is not None:
        checkpoint.save_batch(checkpoint_batch)

    return results


def apply_pivot_filter(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    *,
    llm_client: Any = None,
    model: str = "gpt-5.4-nano",
    threshold: float = 0.6,
    anchor_fraction: float = 0.1,
    ratings: list[PivotRating] | None = None,
    seed: int = 42,
    max_concurrency: int = 20,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PivotFilterResult:
    """Filter training examples to high-signal pivot turns.

    Sync wrapper around ``apply_pivot_filter_async``.  Creates a new event
    loop to avoid monkey-patching the global loop.  If you are already in
    an async context, call ``apply_pivot_filter_async`` directly.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            apply_pivot_filter_async(
                examples,
                traces,
                llm_client=llm_client,
                model=model,
                threshold=threshold,
                anchor_fraction=anchor_fraction,
                ratings=ratings,
                seed=seed,
                max_concurrency=max_concurrency,
                checkpoint_dir=checkpoint_dir,
                progress_callback=progress_callback,
            )
        )
    finally:
        loop.close()


async def apply_pivot_filter_async(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    *,
    llm_client: Any = None,
    model: str = "gpt-5.4-nano",
    threshold: float = 0.6,
    anchor_fraction: float = 0.1,
    ratings: list[PivotRating] | None = None,
    seed: int = 42,
    max_concurrency: int = 20,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PivotFilterResult:
    """Filter training examples to high-signal pivot turns (async version).

    Two-phase approach:
    1. Heuristic fast path tags obviously trivial turns (no LLM cost)
    2. LLM counterfactual analysis rates remaining turns

    Args:
        examples: output of ``build_training_examples()``
        traces: the ``NormalizedTrace`` objects (needed for full-trace context)
        llm_client: AsyncOpenAI-compatible client. Required if ``ratings``
            is not provided.
        model: LLM model for counterfactual analysis
        threshold: importance score cutoff (default 0.6)
        anchor_fraction: fraction of below-threshold turns to keep (default 0.1)
        ratings: pre-computed ratings — skips LLM if provided.
        seed: random seed for anchor sampling reproducibility
        max_concurrency: max concurrent LLM requests
        checkpoint_dir: directory for incremental LLM rating checkpoints.
            Cached ratings are reused across runs; only ``model`` changes
            invalidate the cache.
        progress_callback: called with ``(completed, total)`` after each
            LLM batch.

    Returns:
        ``PivotFilterResult`` with kept, dropped, and all ratings.

    Raises:
        ``ValueError`` if fewer than ``MIN_TRAIN_SAMPLES`` examples remain.
    """
    if ratings is not None:
        all_ratings = list(ratings)
    else:
        if llm_client is None:
            raise ValueError("Either llm_client or ratings must be provided")
        all_ratings = await _compute_ratings_async(
            examples,
            traces,
            llm_client,
            model,
            max_concurrency,
            checkpoint_dir=checkpoint_dir,
            progress_callback=progress_callback,
        )

    # Build lookup: (trace_id, turn_index) → PivotRating
    rating_map: dict[tuple[str, int], PivotRating] = {
        (r.trace_id, r.turn_index): r for r in all_ratings
    }

    # Split into kept and dropped
    above: list[TrainingExample] = []
    below: list[TrainingExample] = []

    for ex in examples:
        key = (ex.trace_id, ex.turn_index)
        rating = rating_map.get(key)
        if rating is None:
            below.append(ex)
            continue
        if rating.importance_score >= threshold:
            above.append(ex)
        else:
            below.append(ex)

    # Anchor sampling: keep a fraction of below-threshold turns
    rng = random.Random(seed)
    if anchor_fraction > 0 and below:
        anchor_count = max(1, int(len(below) * anchor_fraction))
        anchors = rng.sample(below, min(anchor_count, len(below)))
    else:
        anchors = []

    anchor_keys = {(ex.trace_id, ex.turn_index) for ex in anchors}
    kept = above + anchors
    dropped = [ex for ex in below if (ex.trace_id, ex.turn_index) not in anchor_keys]

    if len(kept) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Pivot filtering produced {len(kept)} examples, need at least "
            f"{MIN_TRAIN_SAMPLES}. Lower the threshold (currently {threshold}) "
            f"or add more traces."
        )

    return PivotFilterResult(kept=kept, dropped=dropped, ratings=all_ratings)


async def _compute_ratings_async(
    examples: list[TrainingExample],
    traces: list[NormalizedTrace],
    llm_client: Any,
    model: str,
    max_concurrency: int,
    *,
    checkpoint_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[PivotRating]:
    """Compute pivot ratings: heuristic fast path + LLM for the rest.

    Always recomputes heuristic ratings (instant).  LLM ratings are
    loaded from checkpoint if available, and only uncached turns are
    sent to the LLM.
    """
    # Phase 1: heuristic fast path (always recomputed)
    heuristic_ratings: list[PivotRating] = []
    llm_examples: list[TrainingExample] = []

    for ex in examples:
        if _is_heuristic_trivial(ex, ex.turn_index):
            heuristic_ratings.append(
                PivotRating(
                    trace_id=ex.trace_id,
                    turn_index=ex.turn_index,
                    importance_score=0.0,
                    category="trivial",
                    reasoning="Heuristic: trivial pattern detected",
                )
            )
        else:
            llm_examples.append(ex)

    logger.info(
        "Pivot filtering: %d heuristic trivial, %d need LLM",
        len(heuristic_ratings),
        len(llm_examples),
    )

    # Phase 2: load cached LLM ratings, diff to find uncached
    checkpoint = None
    cached_ratings: dict[tuple[str, int], PivotRating] = {}

    if checkpoint_dir is not None:
        checkpoint = PivotCheckpointManager(checkpoint_dir, model)
        cached_ratings = checkpoint.resume()

    uncached = [ex for ex in llm_examples if (ex.trace_id, ex.turn_index) not in cached_ratings]

    if cached_ratings:
        logger.info(
            "Pivot checkpoint: %d cached, %d uncached (of %d LLM examples)",
            len(llm_examples) - len(uncached),
            len(uncached),
            len(llm_examples),
        )

    # Total for progress = all LLM examples (cached count as already done)
    total_llm = len(llm_examples)
    completed_offset = total_llm - len(uncached)

    # Report initial progress (cached items)
    if progress_callback is not None and completed_offset > 0:
        progress_callback(completed_offset, total_llm)

    # Phase 3: rate uncached turns via LLM
    if uncached:
        new_ratings = await _rate_all_turns(
            llm_client,
            model,
            uncached,
            traces,
            max_concurrency,
            checkpoint=checkpoint,
            progress_callback=progress_callback,
            completed_offset=completed_offset,
            total_llm=total_llm,
        )
    else:
        new_ratings = []

    # Merge: cached + new LLM ratings
    all_llm_ratings: list[PivotRating] = []
    new_map = {(r.trace_id, r.turn_index): r for r in new_ratings}
    for ex in llm_examples:
        key = (ex.trace_id, ex.turn_index)
        rating = cached_ratings.get(key) or new_map.get(key)
        if rating is not None:
            all_llm_ratings.append(rating)

    return heuristic_ratings + all_llm_ratings
