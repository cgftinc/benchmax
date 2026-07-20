"""Helpers for inspecting and rendering OpenAI-style message content.

Content on a message can be a plain string or a list of typed parts (text,
image references, etc.). These helpers give one shared surface for pulling
text out, previewing arbitrary content for logs/UIs, walking image
references, and turning raw image bytes into a data URI.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from benchmax.envs.base.openai_types import Messages

_MIME_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_IMAGE_PLACEHOLDER = "[image]"


def message_text(message: Mapping[str, Any]) -> str:
    """Extract the text content of a message.

    Str content is returned unchanged. List content is the string values of
    text-bearing parts (a dict with a ``text`` or ``content`` key), joined
    with ``"\\n"`` in part order. Image and other non-text parts contribute
    nothing. ``None``/missing content returns ``""``.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if text:
                parts.append(str(text))
        elif item:
            parts.append(str(item))
    return "\n".join(parts)


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def content_preview(content: Any, limit: int) -> str:
    """Render a single-line preview of raw message ``content``, at most `limit` chars.

    Image parts render as an ``[image]`` placeholder. Never raises on
    malformed content shapes — falls back to a truncated ``repr``. Does
    raise ``ValueError`` if ``limit < 1``.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    try:
        preview = _render_preview(content)
    except Exception:
        preview = repr(content)

    preview = " ".join(preview.splitlines())
    if len(preview) <= limit:
        return preview
    if limit <= 3:
        return preview[:limit]
    return preview[: limit - 3] + "..."


def _render_preview(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if _is_image_part(item):
                values.append(_IMAGE_PLACEHOLDER)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                values.append(str(text) if text else repr(item))
            elif item:
                values.append(str(item))
        return " ".join(values)
    return repr(content)


def iter_image_refs(messages: Messages) -> Iterator[str]:
    """Yield the ``image_url.url`` string of every image part, in transcript order.

    Iterates messages in order, then parts within each message's content
    list in order. Messages with str content contribute nothing. Malformed
    image parts (missing keys, non-string url, etc.) are skipped.
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not _is_image_part(part):
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str):
                yield url


def image_to_data_uri(source: str | Path | bytes) -> str:
    """Turn an image path/bytes into a ``data:`` URI, sniffing MIME from magic numbers.

    Strings already starting with ``data:`` or ``https:`` pass through
    untouched. Any other string is treated as a filesystem path. Supports
    PNG, JPEG, GIF, and WEBP signatures; raises ``ValueError`` on
    unrecognized bytes rather than guessing a MIME type.
    """
    if isinstance(source, str) and (source.startswith("data:") or source.startswith("https:")):
        return source

    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source

    mime = _sniff_mime(data)
    if mime is None:
        raise ValueError("unrecognized image byte signature")

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _sniff_mime(data: bytes) -> str | None:
    if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for signature, mime in _MIME_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None
