"""Two-stage filter to remove automated/system emails from parsed JSONL.

Intended position in pipeline:
1) parse (.mbox / other source) -> canonical parsed-email JSONL
2) dedupe (optional)
3) filter_automated_emails (this module)
4) clean_bodies
5) chunk with EmailChunker

Stage 1 — fast heuristic prefilter with high recall:
  Pattern-matches sender addresses and subject lines to flag likely automated
  messages. Designed to catch ~all automated emails, accepting some false positives.

Stage 2 — batched LLM verification for precision:
  Groups flagged emails into batches of N, sends just sender + subject to the LLM,
  and asks which are truly automated. Dramatically reduces LLM calls (e.g., 1000
  suspects / 10 per batch = 100 calls instead of 10,000).

Public API:
- filter_automated_emails(rows, ...) — in-memory, returns (kept_rows, report)
- filter_automated_emails_jsonl(input, output, ...) — file-based convenience wrapper
- filter_automated_emails_folder(folder, ...) — folder-based convenience wrapper
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmax.rag.qa_generation.batch_processor import batch_process_sync

# ---------------------------------------------------------------------------
# Stage 1: heuristic prefilter
# ---------------------------------------------------------------------------

_AUTOMATED_SENDER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"no[\-_.]?reply",
        r"do[\-_.]?not[\-_.]?reply",
        r"mailer[\-_.]?daemon",
        r"postmaster@",
        r"calendar[\-_.]?(notification|noreply|server)",
        r"notify@",
        r"notification",
        r"bounce",
        r"automated",
        r"robot@",
        r"bot@",
        r"daemon@",
        r"jenkins",
        r"jira@",
        r"confluence@",
        r"github\.com",
        r"gitlab\.com",
        r"bitbucket\.org",
        r"circleci\.com",
        r"travis-ci",
        r"dependabot",
        r"renovate",
        r"snyk",
        r"sentry",
        r"pagerduty",
        r"opsgenie",
        r"slack(bot)?@",
        r"zoom\.us",
        r"(google|microsoft|apple)[\-.]?calendar",
        r"venmo@",
        r"paypal@",
        r"cashapp",
        r"square\.com",
        r"stripe\.com",
        r"shopify\.com",
        r"uber\.com",
        r"lyft\.com",
        r"doordash\.com",
        r"grubhub\.com",
        r"linkedin\.com",
        r"facebook(mail)?\.com",
        r"twitter\.com",
        r"x\.com",
        r"instagram\.com",
        r"tiktok\.com",
        r"pinterest\.com",
        r"reddit\.com",
        r"quora\.com",
        r"meetup\.com",
        r"eventbrite\.com",
        r"mailchimp\.com",
        r"sendgrid\.(com|net)",
        r"amazonses\.com",
        r"ses\.amazonaws\.com",
        r"mailgun\.(com|org)",
        r"constantcontact\.com",
        r"hubspot\.com",
        r"marketo\.com",
        r"salesforce\.com",
        r"zendesk\.com",
        r"freshdesk\.com",
        r"intercom",
        r"helpscout",
    ]
]

_AUTOMATED_SUBJECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^invitation:\s",
        r"^updated invitation:\s",
        r"^canceled( event)?:\s",
        r"^accepted:\s",
        r"^declined:\s",
        r"^tentative:\s",
        r"^reminder:\s.*meeting",
        r"zoom meeting invitation",
        r"zoom meeting reminder",
        r"is inviting you to a scheduled zoom meeting",
        r"join (our|my|the) (zoom|teams|meet|webex)",
        r"\bsign[\-\s]?in\b.*(code|alert|notification|attempt)",
        r"\bverification code\b",
        r"\bverify your\b",
        r"\bconfirm your (email|account|registration)\b",
        r"\bpassword reset\b",
        r"\btwo[\-\s]?factor\b",
        r"\b2fa\b",
        r"\bone[\-\s]?time (password|code|passcode)\b",
        r"\breceipt\b.*\b(payment|order|purchase|transaction)\b",
        r"\border confirmation\b",
        r"\bshipping (confirmation|notification|update)\b",
        r"\bdelivery (notification|update|status)\b",
        r"\bsubscription (confirm|renew|cancel|expir)",
        r"\bunsubscribe\b",
        r"\bnewsletter\b",
        r"\bdigest\b",
        r"\bbuild (failed|succeeded|broken|fixed)\b",
        r"\bpipeline (failed|succeeded)\b",
        r"\bCI\/CD\b",
        r"\bpull request\b",
        r"\bmerge request\b",
        r"\bcommit\b.*\bpush",
        r"\bissue (created|updated|closed|assigned)\b",
        r"\bticket\b.*(created|updated|closed|assigned|resolved)",
        r"\balert:\s",
        r"\bincident\b.*(created|resolved|triggered)",
        r"\bout[\-\s]?of[\-\s]?office\b",
        r"\bautomatic reply\b",
        r"\bauto[\-\s]?reply\b",
        r"\bmail delivery\b.*(failed|failure|error)",
        r"\bundeliverable\b",
        r"\bbounced?\b",
        r"\bmailer[\-\s]?daemon\b",
    ]
]


def _extract_email_addr(from_field: Any) -> str:
    """Extract a plain email string from the 'from' field (dict or str)."""
    if isinstance(from_field, dict):
        return (from_field.get("email") or from_field.get("name") or "").strip().lower()
    return str(from_field or "").strip().lower()


def _heuristic_is_suspect(row: dict[str, Any]) -> bool:
    """Return True if the email looks like it *might* be automated."""
    sender = _extract_email_addr(row.get("from"))
    subject = str(row.get("subject") or "").strip()

    for pat in _AUTOMATED_SENDER_PATTERNS:
        if pat.search(sender):
            return True
    for pat in _AUTOMATED_SUBJECT_PATTERNS:
        if pat.search(subject):
            return True
    return False


# ---------------------------------------------------------------------------
# Stage 2: batched LLM verification
# ---------------------------------------------------------------------------

_BATCH_SYSTEM_PROMPT = """You are a strict email classifier.

You will receive a numbered list of emails (sender + subject only).
For each, decide if it is AUTOMATED/system-generated or HUMAN-written.

Automated examples: calendar invites, Zoom scheduling, payment receipts,
sign-in codes, CI/CD notifications, no-reply marketing, mailing list digests,
bounce notices, out-of-office auto-replies.

Human examples: real conversations between people, even if they mention
tools like Zoom, GitHub, calendar, or Venmo.

Return a JSON array of objects, one per email, in the same order:
[
  {"index": 1, "automated": true, "reason": "short reason"},
  {"index": 2, "automated": false, "reason": "short reason"},
  ...
]
Return ONLY the JSON array, no other text."""


def _format_address(addr: Any) -> str:
    if isinstance(addr, dict):
        name = (addr.get("name") or "").strip()
        email = (addr.get("email") or "").strip()
        if name and email:
            return f"{name} <{email}>"
        return name or email or ""
    return str(addr or "").strip()


def _build_batch_prompt(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        sender = _format_address(row.get("from"))
        subject = str(row.get("subject") or "")
        lines.append(f"{i}. From: {sender}\n   Subject: {subject}")
    return "\n".join(lines)


def _parse_batch_response(raw: str, batch_size: int) -> list[bool | None]:
    """Parse LLM batch response into a list of automated booleans.

    Returns a list of length batch_size. None means parse failure for that index.
    """
    results: list[bool | None] = [None] * batch_size
    if not raw:
        return results

    text = raw.strip()
    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return results

    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return results

    if not isinstance(arr, list):
        return results

    for item in arr:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (ValueError, TypeError):
            continue
        if 1 <= idx <= batch_size:
            results[idx - 1] = bool(item.get("automated", False))

    return results


# ---------------------------------------------------------------------------
# Main filter logic
# ---------------------------------------------------------------------------


def filter_automated_emails(
    rows: list[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    max_concurrent: int = 50,
    batch_size: int = 30,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two-stage filter: heuristic prefilter + batched LLM verification.

    Args:
        rows: Parsed email rows (canonical JSONL schema).
        client: OpenAI-compatible client for LLM calls.
        model: Model name for classification.
        max_concurrent: Max parallel LLM calls.
        batch_size: Number of suspect emails per LLM call.
        show_progress: Show progress bar.

    Returns:
        (kept_rows, report) where report contains counts and drop reasons.
    """
    empty_report = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "drop_rate": 0.0,
        "heuristic_suspects": 0,
        "llm_batches": 0,
        "llm_confirmed_drops": 0,
        "llm_reversed_to_keep": 0,
        "parse_failures_kept": 0,
    }
    if not rows:
        return [], empty_report

    # Stage 1: heuristic prefilter
    suspect_indices: list[int] = []
    clean_indices: list[int] = []
    for i, row in enumerate(rows):
        if _heuristic_is_suspect(row):
            suspect_indices.append(i)
        else:
            clean_indices.append(i)

    print(
        f"Heuristic prefilter: {len(suspect_indices)} suspects / "
        f"{len(rows)} total ({len(clean_indices)} passed directly)"
    )

    # Stage 2: batch LLM verification of suspects
    suspect_rows = [rows[i] for i in suspect_indices]
    n_batches = math.ceil(len(suspect_rows) / batch_size) if suspect_rows else 0

    # Build one prompt per batch
    prompts: list[str] = []
    batch_sizes: list[int] = []
    for b in range(n_batches):
        chunk = suspect_rows[b * batch_size : (b + 1) * batch_size]
        prompts.append(_build_batch_prompt(chunk))
        batch_sizes.append(len(chunk))

    llm_confirmed_drops = 0
    llm_reversed = 0
    parse_failures = 0
    # Per-suspect: True = drop, False = keep, None = parse failure (keep)
    suspect_decisions: list[bool] = []

    if prompts:
        print(f"Sending {len(prompts)} batches to LLM for verification ...")
        batch_result = batch_process_sync(
            client=client,
            model=model,
            prompts=prompts,
            system_prompt=_BATCH_SYSTEM_PROMPT,
            max_tokens=50 * batch_size,
            timeout=90.0,
            max_concurrent=max_concurrent,
            show_progress=True,
        )

        for resp, bs in zip(batch_result.responses, batch_sizes, strict=False):
            if resp is None:
                # Entire batch failed — keep all (safe default)
                parse_failures += bs
                suspect_decisions.extend([False] * bs)
                continue
            decisions = _parse_batch_response(resp.answer, bs)
            for d in decisions:
                if d is None:
                    parse_failures += 1
                    suspect_decisions.append(False)  # keep on failure
                elif d:
                    llm_confirmed_drops += 1
                    suspect_decisions.append(True)
                else:
                    llm_reversed += 1
                    suspect_decisions.append(False)

    # Assemble output: keep clean rows + suspects that LLM said are not automated
    kept: list[dict[str, Any]] = [rows[i] for i in clean_indices]

    for idx, should_drop in zip(suspect_indices, suspect_decisions, strict=False):
        if not should_drop:
            kept.append(rows[idx])

    # Restore original order
    kept_set = set(id(r) for r in kept)
    kept_ordered = [r for r in rows if id(r) in kept_set]

    dropped = len(rows) - len(kept_ordered)

    report = {
        "total": len(rows),
        "kept": len(kept_ordered),
        "dropped": dropped,
        "drop_rate": (dropped / len(rows)) if rows else 0.0,
        "heuristic_suspects": len(suspect_indices),
        "llm_batches": n_batches,
        "llm_confirmed_drops": llm_confirmed_drops,
        "llm_reversed_to_keep": llm_reversed,
        "parse_failures_kept": parse_failures,
    }
    return kept_ordered, report


def filter_automated_emails_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    client: OpenAI,
    model: str,
    max_concurrent: int = 50,
    batch_size: int = 30,
) -> dict[str, Any]:
    """Read parsed-email JSONL, filter out automated messages, write output."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    kept, report = filter_automated_emails(
        rows,
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        batch_size=batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["input_path"] = str(input_path)
    report["output_path"] = str(output_path)
    return report


def filter_automated_emails_folder(
    folder: str | Path,
    input_name: str = "_deduped.jsonl",
    output_name: str = "_filtered.jsonl",
    *,
    client: OpenAI,
    model: str,
    max_concurrent: int = 50,
    batch_size: int = 30,
) -> dict[str, Any]:
    """Filter automated emails from a preprocessed JSONL file within a folder.

    Reads <folder>/<input_name>, runs two-stage filter (heuristic + batched LLM),
    writes <folder>/<output_name> with automated messages removed.

    Returns the filter report dict.
    """
    folder = Path(folder)
    input_path = folder / input_name
    output_path = folder / output_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            f"Run dedupe_email_folder first to produce {input_name}."
        )

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    print(f"Filtering {len(rows)} messages in {input_path.name} for automated emails ...")
    kept, report = filter_automated_emails(
        rows,
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        batch_size=batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["input_path"] = str(input_path)
    report["output_path"] = str(output_path)
    print(
        f"Automated filter: kept {report['kept']}/{report['total']} "
        f"(dropped {report['dropped']}, {report['drop_rate']:.1%})"
    )
    print(
        f"  Heuristic suspects: {report['heuristic_suspects']} | "
        f"LLM confirmed: {report['llm_confirmed_drops']} | "
        f"LLM reversed: {report['llm_reversed_to_keep']} | "
        f"Parse failures kept: {report['parse_failures_kept']}"
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter automated/system emails from parsed JSONL.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Input parsed-email JSONL path")
    group.add_argument("--folder", help="Folder with preprocessed _deduped.jsonl to filter")
    p.add_argument("--output", default="", help="Output filtered JSONL path (for --input mode)")
    p.add_argument("--model", required=True, help="LLM model name")
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible base URL",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (or set OPENAI_API_KEY)",
    )
    p.add_argument("--max-concurrent", type=int, default=50, help="Max concurrent LLM calls")
    p.add_argument("--batch-size", type=int, default=30, help="Emails per LLM batch call")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.folder:
        filter_automated_emails_folder(
            args.folder,
            client=client,
            model=args.model,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
        )
    else:
        input_path = Path(args.input).expanduser()
        if args.output:
            output_path = Path(args.output).expanduser()
        else:
            output_path = input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}")

        report = filter_automated_emails_jsonl(
            input_path,
            output_path,
            client=client,
            model=args.model,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
        )

        print(
            f"Kept {report['kept']}/{report['total']} "
            f"(dropped {report['dropped']}, {report['drop_rate']:.1%})"
        )


if __name__ == "__main__":
    main()
