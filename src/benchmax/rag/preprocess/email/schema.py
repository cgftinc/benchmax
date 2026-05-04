"""Canonical email schema + normalization helpers.

Canonical input schema (JSONL rows):
- One object per parsed email message.
- Required fields:
  - "id": stable message id.
  - "thread_id": conversation/thread identifier.
  - "date": sortable timestamp string.
  - "subject": subject text (can be empty).
  - "body": body text (can be empty).
- "from": address object.
- Required linkage fields:
  - "reply_to": parent message id (empty for root messages).
- Optional fields (used when present):
  - "to": list of address objects.
  - "cc": list of address objects.
  - other metadata fields are preserved in message-level output.

Expected usage:
- Parsers/preprocessing normalize participant/date fields with:
  `clean_participant_label(...)`, `extract_participants(...)`,
  and `date_yyyy_mm_dd(...)`.
- Pipelines produce canonical message rows according to the schema.
- Dedupe/chunking call `validate_rows(...)` to collect non-fatal warnings.

Validation is non-fatal by design: this module surfaces warnings instead of
raising, so pipelines can continue on large corpora with sparse bad rows.
"""

from __future__ import annotations

import re
from typing import Any

REQUIRED_FIELDS = ("id", "thread_id", "date", "subject", "body", "from", "reply_to")
WS_RE = re.compile(r"\s+")
MAILTO_RE = re.compile(r"^\s*mailto:\s*", re.IGNORECASE)
ANGLE_GROUP_RE = re.compile(r"<([^<>]+)>")
BRACKET_CONTENT_RE = re.compile(r"\[[^\]]*\]")
PUNCT_EDGE_RE = re.compile(r"^[\s,;:._\-]+|[\s,;:._\-]+$")


def as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def email_to_label(value: str) -> str:
    """Convert email-like text into a filter-friendly label."""
    text = WS_RE.sub(" ", (value or "").strip().lower())
    if "@" not in text:
        return ""

    local_part, domain_part = text.split("@", 1)
    local_tokens = [part for part in re.split(r"[^a-z0-9]+", local_part) if part]
    domain_tokens = [part for part in re.split(r"[^a-z0-9]+", domain_part) if part]
    return " ".join(local_tokens + domain_tokens)


def clean_participant_label(value: str) -> str:
    """Normalize participant labels conservatively for broad email datasets."""
    text = WS_RE.sub(" ", (value or "").strip())
    if not text:
        return ""

    text = MAILTO_RE.sub("", text)

    # Strip outer wrapper brackets repeatedly: "[[Name]]" -> "Name"
    while text.startswith("[") and text.endswith("]") and len(text) >= 2:
        text = text[1:-1].strip()
    text = MAILTO_RE.sub("", text)

    angle_groups = ANGLE_GROUP_RE.findall(text)
    if angle_groups:
        outside = ANGLE_GROUP_RE.sub(" ", text)
        outside = WS_RE.sub(" ", outside).strip()
        outside = PUNCT_EDGE_RE.sub("", outside)
        text = outside if outside else angle_groups[0].strip()

    stripped = BRACKET_CONTENT_RE.sub(" ", text)
    stripped = WS_RE.sub(" ", stripped).strip()
    stripped = PUNCT_EDGE_RE.sub("", stripped)

    # Keep fallback if stripping removed all signal.
    candidate = stripped if re.search(r"[A-Za-z0-9]", stripped) else text
    candidate = WS_RE.sub(" ", candidate).strip()
    candidate = PUNCT_EDGE_RE.sub("", candidate)

    if not re.search(r"[A-Za-z0-9]", candidate):
        return ""

    if "@" in candidate:
        email_candidate = email_to_label(candidate)
        if email_candidate:
            return email_candidate

    return candidate


def extract_participants(email_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract participant display text and canonical filter tokens from one pass."""
    display_values: list[str] = []
    tokens: list[str] = []
    seen_tokens: set[str] = set()

    def add_label(raw_value: str) -> None:
        cleaned = clean_participant_label(raw_value)
        if not cleaned:
            return
        token = cleaned.lower()
        if token in seen_tokens:
            return
        seen_tokens.add(token)
        tokens.append(token)
        display_values.append(cleaned)

    for email_message in email_messages:
        from_address = (
            email_message.get("from") if isinstance(email_message.get("from"), dict) else {}
        )
        add_label((from_address.get("name") or from_address.get("email") or "").strip())

        to_addresses = email_message.get("to") if isinstance(email_message.get("to"), list) else []
        cc_addresses = email_message.get("cc") if isinstance(email_message.get("cc"), list) else []
        for address in to_addresses + cc_addresses:
            if not isinstance(address, dict):
                continue
            add_label((address.get("name") or address.get("email") or "").strip())

    return {
        "tokens": tokens,
        "display": ", ".join(display_values),
    }


def date_yyyy_mm_dd(value: Any) -> str:
    """Return YYYY-MM-DD prefix for ISO-like date strings, else empty string."""
    return str(value or "")[:10]


def validate_rows(rows: list[dict[str, Any]], max_warnings: int = 200) -> list[str]:
    """Collect schema warnings for canonical parsed-email rows.

    Returns:
    - list of warning strings (empty when no issues detected)
    """

    warnings: list[str] = []
    for idx, row in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            warnings.append(
                f"row={idx} missing required fields={missing} (id={as_str(row.get('id'))!r}, "
                f"thread_id={as_str(row.get('thread_id'))!r})"
            )
            if len(warnings) >= max_warnings:
                break
            continue

        if not as_str(row.get("id")):
            warnings.append(
                f"row={idx} has empty 'id' (thread_id={as_str(row.get('thread_id'))!r})"
            )
        if not as_str(row.get("thread_id")):
            warnings.append(f"row={idx} has empty 'thread_id' (id={as_str(row.get('id'))!r})")
        if not isinstance(row.get("from"), dict):
            warnings.append(
                f"row={idx} has invalid 'from' type={type(row.get('from')).__name__}; expected object"
            )
        for opt_list in ("to", "cc"):
            if opt_list in row and not isinstance(row.get(opt_list), list):
                warnings.append(
                    f"row={idx} has invalid '{opt_list}' type={type(row.get(opt_list)).__name__}; "
                    "expected list"
                )
        if len(warnings) >= max_warnings:
            break

    return warnings
