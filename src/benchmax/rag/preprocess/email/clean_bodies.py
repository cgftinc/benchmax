#!/usr/bin/env python3
"""Clean parsed-email JSONL bodies by removing quoted replies and signatures.

Intended position in pipeline:
1) parse (.mbox / other source) -> canonical parsed-email JSONL
2) dedupe / spam prefilter (optional)
3) clean_bodies (this module)
4) chunk with EmailChunker

This cleaner is conservative:
- Removes quote blocks only when they likely duplicate other messages in the same thread.
- Removes common signature tails from the remaining body text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from benchmax.rag.preprocess.email.schema import validate_rows

DEFAULT_INPUT_PATH = "emails_deduped.jsonl"
DEFAULT_OUTPUT_PATH = "emails_cleaned.jsonl"

WS_RE = re.compile(r"\s+")
ALNUM_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9']+")

# Common quote starters found in email clients.
QUOTE_START_PATTERNS = [
    re.compile(r"^\s*-{2,}\s*original message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*on .+wrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*begin forwarded message:\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*forwarded message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*>+\s*from:\s+.+$", re.IGNORECASE),
    re.compile(r"^\s*from:\s+.+$", re.IGNORECASE),
    re.compile(r"^\s*sent:\s+.+$", re.IGNORECASE),
    re.compile(r"^\s*to:\s+.+$", re.IGNORECASE),
    re.compile(r"^\s*subject:\s+.+$", re.IGNORECASE),
]

# Inline quote markers (same line paragraph style, not necessarily line-start clean).
INLINE_QUOTE_MARKERS = [
    re.compile(r"\bon\s+.+?\s+wrote:\s*", re.IGNORECASE),
    re.compile(r"-{2,}\s*original message\s*-{2,}", re.IGNORECASE),
    re.compile(r"\bbegin forwarded message:\b", re.IGNORECASE),
    re.compile(r"-{2,}\s*forwarded message\s*-{2,}", re.IGNORECASE),
    re.compile(r"\bfrom:\s.+\bsent:\s.+\bsubject:\s", re.IGNORECASE),
]

# Signature markers, checked only near tail.
SIGNATURE_PATTERNS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*__+\s*$"),
    re.compile(r"^\s*best[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*best regards[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*regards[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*thanks[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*thank you[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*cheers[,!\s]*$", re.IGNORECASE),
    re.compile(r"^\s*sent from my .+$", re.IGNORECASE),
]

try:
    from email_reply_parser import EmailReplyParser
except Exception:
    EmailReplyParser = None

try:
    from talon.signature.bruteforce import extract_signature
except Exception:
    extract_signature = None


def _normalize_for_match(text: str) -> str:
    lowered = (text or "").lower()
    alnum_space = ALNUM_RE.sub(" ", lowered)
    return WS_RE.sub(" ", alnum_space).strip()


def _tokenize_norm(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _shingles(tokens: list[str], size: int = 4) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at line {line_num} in {path}")
            rows.append(obj)
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _find_quote_start(lines: list[str]) -> int | None:
    # 1) explicit quote header markers
    for i, line in enumerate(lines):
        for pat in QUOTE_START_PATTERNS:
            if pat.match(line):
                return i

    # 2) contiguous trailing quoted lines ("> ...")
    tail_start: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith(">"):
            tail_start = i
        elif lines[i].strip() == "":
            continue
        else:
            break
    return tail_start


def _find_inline_quote_cut(text: str) -> int | None:
    if not text:
        return None
    cut: int | None = None
    for pat in INLINE_QUOTE_MARKERS:
        m = pat.search(text)
        if not m:
            continue
        idx = m.start()
        if cut is None or idx < cut:
            cut = idx
    return cut


def _candidate_quote_text(lines: list[str], quote_start: int) -> str:
    quoted_lines = lines[quote_start:]
    cleaned = [ln.lstrip("> ").rstrip() for ln in quoted_lines]
    return "\n".join(cleaned).strip()


def _looks_like_known_message(
    candidate_norm: str, thread_body_norms: list[str], min_chars: int
) -> bool:
    if len(candidate_norm) < min_chars:
        return False
    for other in thread_body_norms:
        if not other:
            continue
        if candidate_norm == other:
            return True
        # Candidate can include legal/footer tail while still fully containing ancestor text.
        if len(other) >= 80 and other in candidate_norm:
            return True
        if len(candidate_norm) >= 80 and candidate_norm in other:
            return True
    return False


def _candidate_matches_thread_by_jaccard(
    candidate_text: str,
    other_thread_shingles: list[set[str]],
    min_chars: int,
    threshold: float,
) -> bool:
    norm = _normalize_for_match(candidate_text)
    if len(norm) < min_chars:
        return False
    candidate_tokens = _tokenize_norm(norm)
    cand_shingles = _shingles(candidate_tokens, size=4)
    if not cand_shingles:
        return False

    max_sim = 0.0
    for other in other_thread_shingles:
        if not other:
            continue
        sim = _jaccard_similarity(cand_shingles, other)
        if sim > max_sim:
            max_sim = sim
        if sim >= threshold:
            return True
        # If candidate is a superset (e.g., quoted block + disclaimer), Jaccard can be low.
        # Ancestor coverage catches this by measuring how much of ancestor appears in candidate.
        overlap = len(cand_shingles & other)
        ancestor_coverage = overlap / len(other) if other else 0.0
        if ancestor_coverage >= threshold:
            return True
    return False


def _strip_signature_tail(lines: list[str], tail_window: int) -> list[str]:
    if not lines:
        return lines
    start = max(0, len(lines) - tail_window)
    for i in range(start, len(lines)):
        line = lines[i]
        for pat in SIGNATURE_PATTERNS:
            if pat.match(line):
                return lines[:i]
    return lines


def _clean_one_body(
    body: str,
    ancestor_body_norms: list[str],
    ancestor_body_shingles: list[set[str]],
    quote_match_min_chars: int,
    quote_match_jaccard_threshold: float,
    signature_tail_window: int,
) -> tuple[str, bool, bool]:
    """Return (cleaned_body, quote_removed, signature_removed)."""
    original = (body or "").strip()
    working = original
    quote_removed = False
    signature_removed = False

    # Preferred path: email_reply_parser (library-based).
    if EmailReplyParser is not None:
        try:
            parsed = EmailReplyParser.parse_reply(working).strip()
            if parsed and parsed != working:
                candidate = working[len(parsed) :].strip()
                if _candidate_matches_thread_by_jaccard(
                    candidate_text=candidate,
                    other_thread_shingles=ancestor_body_shingles,
                    min_chars=quote_match_min_chars,
                    threshold=quote_match_jaccard_threshold,
                ) or _looks_like_known_message(
                    _normalize_for_match(candidate),
                    ancestor_body_norms,
                    quote_match_min_chars,
                ):
                    working = parsed
                    quote_removed = True
        except Exception:
            pass

    # Fallback path: inline marker cut + ancestor confirmation.
    inline_cut = _find_inline_quote_cut(working)
    if inline_cut is not None:
        prefix = working[:inline_cut].rstrip()
        candidate = working[inline_cut:].strip()
        if len(_normalize_for_match(prefix)) >= 20 and (
            _candidate_matches_thread_by_jaccard(
                candidate_text=candidate,
                other_thread_shingles=ancestor_body_shingles,
                min_chars=quote_match_min_chars,
                threshold=quote_match_jaccard_threshold,
            )
            or _looks_like_known_message(
                _normalize_for_match(candidate), ancestor_body_norms, quote_match_min_chars
            )
        ):
            working = prefix
            quote_removed = True
    lines = working.splitlines()
    quote_start = _find_quote_start(lines)
    if quote_start is not None:
        candidate = _candidate_quote_text(lines, quote_start)
        candidate_norm = _normalize_for_match(candidate)
        if _candidate_matches_thread_by_jaccard(
            candidate_text=candidate,
            other_thread_shingles=ancestor_body_shingles,
            min_chars=quote_match_min_chars,
            threshold=quote_match_jaccard_threshold,
        ) or _looks_like_known_message(candidate_norm, ancestor_body_norms, quote_match_min_chars):
            lines = lines[:quote_start]
            quote_removed = True

    # Preferred path: talon signature extractor.
    if extract_signature is not None:
        try:
            text_no_sig, sig = extract_signature("\n".join(lines))
            text_no_sig = (text_no_sig or "").strip()
            if sig and text_no_sig and text_no_sig != "\n".join(lines).strip():
                lines = text_no_sig.splitlines()
                signature_removed = True
        except Exception:
            pass

    # Fallback path: conservative tail stripping.
    if not signature_removed:
        before_sig = list(lines)
        lines = _strip_signature_tail(lines, tail_window=signature_tail_window)
        signature_removed = len(lines) < len(before_sig)

    cleaned = "\n".join(lines).strip()
    # Keep original if cleaning would leave no meaningful body.
    if len(_normalize_for_match(cleaned)) < 8:
        return original, False, False
    return cleaned, quote_removed, signature_removed


def clean_email_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    quote_match_min_chars: int = 40,
    quote_match_jaccard_threshold: float = 0.9,
    signature_tail_window: int = 12,
    keep_original_body_field: str = "",
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = _read_jsonl(input_path)

    warnings = validate_rows(rows)
    if warnings:
        print(f"Schema warnings: {len(warnings)} (showing up to 10)")
        for w in warnings[:10]:
            print(f"  - {w}")

    by_thread: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        thread_id = str(row.get("thread_id") or "")
        by_thread.setdefault(thread_id, []).append(row)

    quote_removed_count = 0
    signature_removed_count = 0
    changed_count = 0

    for _thread_id, thread_rows in by_thread.items():
        id_to_row: dict[str, dict[str, Any]] = {}
        for row in thread_rows:
            rid = str(row.get("id") or "").strip()
            if rid and rid not in id_to_row:
                id_to_row[rid] = row

        norm_by_id: dict[str, str] = {}
        shingles_by_id: dict[str, set[str]] = {}
        for rid, row in id_to_row.items():
            norm = _normalize_for_match(str(row.get("body") or ""))
            norm_by_id[rid] = norm
            shingles_by_id[rid] = _shingles(_tokenize_norm(norm), size=4)

        def ancestor_ids(row: dict[str, Any]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            cur = str(row.get("reply_to") or "").strip()
            while cur and cur not in seen and cur in id_to_row:
                seen.add(cur)
                out.append(cur)
                cur = str(id_to_row[cur].get("reply_to") or "").strip()
            return out

        for row in thread_rows:
            body = str(row.get("body") or "")
            anc_ids = ancestor_ids(row)
            anc_norms = [norm_by_id[aid] for aid in anc_ids if aid in norm_by_id]
            anc_shingles = [shingles_by_id[aid] for aid in anc_ids if aid in shingles_by_id]
            cleaned, quote_removed, sig_removed = _clean_one_body(
                body=body,
                ancestor_body_norms=anc_norms,
                ancestor_body_shingles=anc_shingles,
                quote_match_min_chars=quote_match_min_chars,
                quote_match_jaccard_threshold=quote_match_jaccard_threshold,
                signature_tail_window=signature_tail_window,
            )
            changed = cleaned != body
            if keep_original_body_field and keep_original_body_field not in row and changed:
                row[keep_original_body_field] = body
            if changed:
                row["body"] = cleaned
                changed_count += 1
            if quote_removed and changed:
                quote_removed_count += 1
            if sig_removed and changed:
                signature_removed_count += 1

    _write_jsonl(rows, output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "reply_parser_available": EmailReplyParser is not None,
        "signature_parser_available": extract_signature is not None,
        "total_rows": len(rows),
        "changed_rows": changed_count,
        "quote_removed_rows": quote_removed_count,
        "signature_removed_rows": signature_removed_count,
    }


def clean_email_folder(
    folder: str | Path,
    input_name: str = "_deduped.jsonl",
    output_name: str = "_cleaned.jsonl",
    quote_match_min_chars: int = 40,
    quote_match_jaccard_threshold: float = 0.9,
    signature_tail_window: int = 12,
    keep_original_body_field: str = "",
) -> dict[str, Any]:
    """Clean email bodies in a preprocessed JSONL file within a folder.

    Reads <folder>/<input_name>, cleans quoted replies and signatures,
    writes <folder>/<output_name>.

    Returns the cleaning stats dict.
    """
    folder = Path(folder)
    input_path = folder / input_name
    output_path = folder / output_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            f"Run dedupe_email_folder first to produce {input_name}."
        )

    print(f"Cleaning bodies in {input_path.name} ...")
    stats = clean_email_jsonl(
        input_path=input_path,
        output_path=output_path,
        quote_match_min_chars=quote_match_min_chars,
        quote_match_jaccard_threshold=quote_match_jaccard_threshold,
        signature_tail_window=signature_tail_window,
        keep_original_body_field=keep_original_body_field,
    )
    print(
        f"Clean: {stats['changed_rows']}/{stats['total_rows']} bodies cleaned "
        f"(quotes: {stats['quote_removed_rows']}, signatures: {stats['signature_removed_rows']})"
    )
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean parsed-email JSONL bodies (quoted replies + signatures).",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Input parsed-email JSONL path")
    group.add_argument("--folder", help="Folder with preprocessed _deduped.jsonl to clean")
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Output cleaned JSONL path (for --input mode)"
    )
    p.add_argument(
        "--quote-match-min-chars",
        type=int,
        default=40,
        help="Minimum normalized quote length to attempt thread-level match",
    )
    p.add_argument(
        "--quote-match-jaccard-threshold",
        type=float,
        default=0.9,
        help="Thread-level suffix Jaccard threshold for quote removal",
    )
    p.add_argument(
        "--signature-tail-window",
        type=int,
        default=12,
        help="How many trailing lines to scan for signature markers",
    )
    p.add_argument(
        "--keep-original-body-field",
        default="",
        help="Optional field name to store original body before cleaning",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.folder:
        clean_email_folder(
            folder=args.folder,
            quote_match_min_chars=args.quote_match_min_chars,
            quote_match_jaccard_threshold=args.quote_match_jaccard_threshold,
            signature_tail_window=args.signature_tail_window,
            keep_original_body_field=args.keep_original_body_field,
        )
    else:
        stats = clean_email_jsonl(
            input_path=args.input,
            output_path=args.output,
            quote_match_min_chars=args.quote_match_min_chars,
            quote_match_jaccard_threshold=args.quote_match_jaccard_threshold,
            signature_tail_window=args.signature_tail_window,
            keep_original_body_field=args.keep_original_body_field,
        )
        print("Body cleaning complete")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
