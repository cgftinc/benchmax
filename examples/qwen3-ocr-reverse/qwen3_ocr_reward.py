"""Infinity-Doc OCR layout reward.

Scores document-parsing rollouts as ``r_dist + r_count + r_order`` in ``[0, 3]``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from html import unescape
from typing import Any

from benchmax.envs import Messages

logger = logging.getLogger(__name__)


def assistant_text(messages: Messages) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)
    return ""


def clean_text(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].strip().lower() in (
            "```",
            "```markdown",
            "```md",
            "```html",
        ):
            text = "\n".join(lines[1:-1]).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return unescape(text).strip()


def collapse_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([$([{])\s+", r"\1", text)
    return text.strip()


def strip_markup(value: str) -> str:
    text = str(value or "")
    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return collapse_text(text)


def html_table_rows(value: str) -> list[str]:
    rows: list[str] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", value, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        normalized_cells = [strip_markup(cell) for cell in cells]
        normalized_cells = [cell for cell in normalized_cells if cell]
        if normalized_cells:
            rows.append(" | ".join(normalized_cells))
    return rows


def markdown_table_rows(value: str) -> list[str]:
    rows: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if "|" not in stripped:
            continue
        compact = stripped.strip("|").strip()
        if not compact:
            continue
        if re.fullmatch(r":?-{3,}:?(\s*\|\s*:?-{3,}:?)+", compact):
            continue
        cells = [strip_markup(cell) for cell in compact.split("|")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def html_blocks(value: str) -> list[str]:
    if not re.search(r"<[a-zA-Z][^>]*>", value):
        return []
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(
        r"</\s*(p|div|section|article|h[1-6]|li|ul|ol|tr|table|thead|tbody)\s*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    stripped = strip_markup(text)
    return [line.strip() for line in stripped.splitlines() if line.strip()]


def segments(value: str) -> list[str]:
    text = clean_text(value)
    if not text:
        return []

    table_rows = html_table_rows(text)
    table_rows.extend(markdown_table_rows(text))
    if table_rows:
        return table_rows

    blocks = html_blocks(text)
    if len(blocks) > 1:
        return blocks

    blank_blocks = [
        segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()
    ]
    if len(blank_blocks) > 1:
        return blank_blocks

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    return [text.strip()]


def normalize_for_similarity(value: str) -> str:
    text = strip_markup(value).lower()
    text = re.sub(r"[*_`#]+", " ", text)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(reference: str, prediction: str) -> float:
    ref = normalize_for_similarity(reference)
    pred = normalize_for_similarity(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0

    char_score = SequenceMatcher(None, ref, pred, autojunk=False).ratio()
    ref_tokens = re.findall(r"[a-z0-9]+|[$%()/.+-]", ref)
    pred_tokens = re.findall(r"[a-z0-9]+|[$%()/.+-]", pred)
    if not ref_tokens or not pred_tokens:
        return char_score

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    token_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return max(char_score, token_f1)


def hungarian_maximize(scores: list[list[float]]) -> list[tuple[int, int]]:
    if not scores or not scores[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        cost = [[-score for score in row] for row in scores]
        row_indices, col_indices = linear_sum_assignment(cost)
        return [(int(i), int(j)) for i, j in zip(row_indices, col_indices)]
    except Exception as exc:
        logger.warning(
            "[infinity_doc_reward_probe] scipy_hungarian_failed=%r using_greedy_fallback",
            exc,
        )
        candidates = sorted(
            ((score, i, j) for i, row in enumerate(scores) for j, score in enumerate(row)),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _score, i, j in candidates:
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            matches.append((i, j))
        return matches


def count_inversions(values: list[int]) -> int:
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                inversions += 1
    return inversions


def infinity_doc_reward(
    messages: Messages,
    task: dict[str, Any] | None,
    **kwargs: Any,
) -> float:
    """Return ``answer_correct`` in ``[0, 1]`` for Infinity-Doc OCR.

    Mean of three equally weighted [0, 1] components: per-segment similarity
    (max of char-ratio and token-F1, Hungarian-matched), segment-count
    agreement, and matched-segment ordering.
    """
    del kwargs  # BaseEnv compatibility
    reference = clean_text(str((task or {}).get("answer", "")))
    prediction = clean_text(assistant_text(messages))

    ref_segments = segments(reference)
    pred_segments = segments(prediction)
    n_ref = len(ref_segments)
    n_pred = len(pred_segments)

    if n_ref == 0:
        reward = 1.0 if n_pred == 0 else 0.0
        logger.info(
            "[infinity_doc_reward_probe] reward=%s r_dist=%s r_count=%s r_order=%s "
            "n_ref=%d n_pred=%d matched=%d raw_matched=%d reference_chars=%d prediction_chars=%d",
            reward,
            reward,
            reward,
            reward,
            n_ref,
            n_pred,
            0,
            0,
            len(reference),
            len(prediction),
        )
        return reward

    if n_pred == 0:
        logger.info(
            "[infinity_doc_reward_probe] reward=0.0 r_dist=0.0 r_count=0.0 r_order=0.0 "
            "n_ref=%d n_pred=0 matched=0 raw_matched=0 reference_chars=%d prediction_chars=%d",
            n_ref,
            len(reference),
            len(prediction),
        )
        return 0.0

    sim_matrix = [
        [similarity(ref, pred) for pred in pred_segments] for ref in ref_segments
    ]
    raw_matches = hungarian_maximize(sim_matrix)
    good_matches = [
        (i, j, sim_matrix[i][j]) for i, j in raw_matches if sim_matrix[i][j] >= 0.12
    ]

    best_per_ref = [max(row) if row else 0.0 for row in sim_matrix]
    r_dist = sum(best_per_ref) / max(n_ref, n_pred)
    r_dist = max(0.0, min(1.0, r_dist))

    r_count = min(n_ref, n_pred) / max(n_ref, n_pred)
    r_count = max(0.0, min(1.0, r_count))

    if len(good_matches) < 2:
        r_order = 0.0
    else:
        pred_order = [j for _i, j, _score in sorted(good_matches)]
        max_inv = len(pred_order) * (len(pred_order) - 1) / 2
        inversions = count_inversions(pred_order)
        r_order = 1.0 - inversions / max_inv if max_inv else 0.0
        r_order = max(0.0, min(1.0, r_order))

    reward = (r_dist + r_count + r_order) / 3.0
    logger.info(
        "[infinity_doc_reward_probe] reward=%s r_dist=%s r_count=%s r_order=%s "
        "n_ref=%d n_pred=%d matched=%d raw_matched=%d reference_chars=%d prediction_chars=%d",
        reward,
        r_dist,
        r_count,
        r_order,
        n_ref,
        n_pred,
        len(good_matches),
        len(raw_matches),
        len(reference),
        len(prediction),
    )
    return reward
