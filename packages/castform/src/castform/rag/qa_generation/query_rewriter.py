"""Shared query rewriting utilities for BM25-oriented retrieval."""

from __future__ import annotations

import re

BM25_STOPWORDS = {
    "a",
    "about",
    "according",
    "across",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "exact",
    "find",
    "for",
    "from",
    "get",
    "give",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "need",
    "of",
    "on",
    "once",
    "out",
    "or",
    "please",
    "show",
    "should",
    "that",
    "the",
    "tell",
    "than",
    "then",
    "there",
    "these",
    "this",
    "those",
    "to",
    "under",
    "up",
    "use",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "why",
    "with",
    "would",
    "you",
    "your",
}

BOOL_OPERATORS = {"AND", "OR", "NOT"}


def heuristic_query_rewrite(
    query: str,
    *,
    max_terms: int = 16,
    max_chars: int = 140,
) -> str:
    """Rewrite verbose natural-language text into a BM25-friendly query."""
    normalized = re.sub(r"\s+", " ", query.strip())
    if not normalized:
        return normalized

    max_terms = max(1, int(max_terms))
    max_chars = max(20, int(max_chars))

    quoted_phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r'"([^"]+)"', normalized):
        phrase = re.sub(r"\s+", " ", match.group(1).strip())
        if len(phrase) < 2:
            continue
        wrapped = f'"{phrase}"'
        low = wrapped.lower()
        if low not in seen:
            seen.add(low)
            quoted_phrases.append(wrapped)

    token_source = re.sub(r'"[^"]+"', " ", normalized)
    token_source = re.sub(r"[?!,;(){}\[\]]", " ", token_source)

    tokens: list[str] = []
    for raw_token in re.findall(r"[A-Za-z0-9$@._:/-]+", token_source):
        token = raw_token.strip()
        if not token:
            continue

        lower = token.lower()
        if token in BOOL_OPERATORS:
            normalized_token = token
        else:
            if lower in BM25_STOPWORDS:
                continue
            if len(token) == 1 and not token.isdigit():
                continue
            normalized_token = token

        key = normalized_token.lower()
        if key not in seen:
            seen.add(key)
            tokens.append(normalized_token)

    site_tokens = [token for token in tokens if token.lower().startswith("site:")]
    other_tokens = [token for token in tokens if not token.lower().startswith("site:")]
    merged = [*site_tokens, *quoted_phrases, *other_tokens]
    merged = merged[:max_terms]
    rewritten = " ".join(merged).strip()

    if not rewritten:
        return normalized
    if len(rewritten) <= max_chars:
        return rewritten

    clipped = rewritten[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped or normalized
