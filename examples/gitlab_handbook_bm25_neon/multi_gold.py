"""Blind, judge-confirmed multi-gold expansion for retrieval-eval rows.

Exact-hash gold undercounts: a question is often answered by chunks other than the
single qa-gen anchor (adjacent chunks of the same document, an overlapping
section), so a retriever that returns an equally-correct chunk is scored a miss.
This module expands a row's gold set to include SAME-FILE chunks that a strict
judge confirms also answer the question.

It is non-circular (F12/B8): candidates come from the in-memory corpus by file
metadata and are judged on their CONTENT, never from a Neon retrieval result — so
the gold is still authored blind of what the provider returns. The judge is
deliberately strict (a chunk counts only if it independently answers the specific
question) so expansion credits genuine equivalents, not merely topical neighbours.

Reproducibility (committed cache)
---------------------------------
The judge is an LLM, so a fresh expansion is not bit-reproducible and can admit
non-answering same-file siblings that inflate any-hit rates. The frozen dataset
therefore ships a COMMITTED expansion cache (``datasets/gold_expansion_cache.json``,
keyed by qa-gen query) that records the AUDITED final gold union per natural row —
the judge (and downstream equivalence) result, hand-pruned of confirmed
non-answering hashes. When an expander is given the cache, ``expand`` returns the
cached union verbatim and never calls the judge, so re-freeze is deterministic; the
live judge path remains as the seeding method and the fallback for any uncached row.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from castform.platform.credentials import resolve_judge_key
from castform.rag.chunkers.models import ChunkCollection

_JUDGE_MODEL = "gpt-5.4-mini"
# Same-file candidates nearest (by chunk index) the anchor to consider, per row.
_MAX_CANDIDATES = 8

_JUDGE_SYSTEM = (
    "You decide which handbook excerpts independently answer a question. Be "
    "strict: an excerpt counts only if, on its own, it states the specific fact, "
    "value, name, step, or decision the question asks for — not merely the same "
    "topic. Reply with a JSON array of the 0-based indices that qualify, e.g. "
    "[0, 2], or [] if none do. No prose."
)

_INDEX_RE = re.compile(r"-?\d+")


class MultiGoldExpander:
    """Expands gold sets with judge-confirmed same-file chunks (one call per row).

    Args:
        collection: The in-memory re-chunked corpus (source of candidate content).
        base_url: OpenAI-compatible endpoint for the judge (pinned to the platform).
        model: Judge model id.
        cache: Optional committed expansion cache (query -> audited final gold union).
            A cache hit is returned verbatim and the judge is never called.
    """

    def __init__(
        self,
        collection: ChunkCollection,
        *,
        base_url: str,
        model: str = _JUDGE_MODEL,
        cache: dict[str, list[str]] | None = None,
    ) -> None:
        self._by_file: dict[str, list] = defaultdict(list)
        self._by_hash: dict[str, object] = {}
        for chunk in collection.chunks:
            md = dict(chunk.metadata)
            self._by_file[str(md.get("file", ""))].append(chunk)
            self._by_hash[chunk.hash] = chunk
        for chunks in self._by_file.values():
            chunks.sort(key=lambda c: int(dict(c.metadata).get("index", 0) or 0))
        self._base_url = base_url
        self._model = model
        self._client = None
        self._cache = cache or {}

    def cached(self, question: str) -> bool:
        """Whether ``question`` is covered by the committed expansion cache."""
        return question in self._cache

    def _judge(self, question: str, candidates: list) -> list[int]:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=resolve_judge_key("", self._base_url),
            )
        blocks = "\n\n".join(
            f"[{i}] {c.content[:1200]}" for i, c in enumerate(candidates)
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"QUESTION: {question}\n\nEXCERPTS:\n{blocks}",
                    },
                ],
            )
            text = resp.choices[0].message.content or "[]"
        except Exception:  # judge failure must never drop the anchor; expand nothing
            return []
        return _parse_indices(text, len(candidates))

    def expand(self, question: str, anchor_hashes: list[str]) -> list[str]:
        """Return ``anchor_hashes`` plus judge-confirmed same-file supporting chunks.

        A committed-cache hit is returned verbatim (deterministic, no judge). Otherwise
        candidates are the anchor file's other chunks nearest the anchor by index
        (capped at :data:`_MAX_CANDIDATES`); the anchor set is always preserved.
        """
        if question in self._cache:
            return list(dict.fromkeys(self._cache[question]))
        gold = list(dict.fromkeys(anchor_hashes))
        anchor = next(
            (self._by_hash.get(h) for h in anchor_hashes if h in self._by_hash), None
        )
        if anchor is None:
            return gold
        file = str(dict(anchor.metadata).get("file", ""))
        anchor_idx = int(dict(anchor.metadata).get("index", 0) or 0)
        siblings = [c for c in self._by_file.get(file, []) if c.hash not in set(gold)]
        siblings.sort(
            key=lambda c: abs(int(dict(c.metadata).get("index", 0) or 0) - anchor_idx)
        )
        candidates = siblings[:_MAX_CANDIDATES]
        if not candidates:
            return gold
        for idx in self._judge(question, candidates):
            if 0 <= idx < len(candidates):
                gold.append(candidates[idx].hash)
        return list(dict.fromkeys(gold))


def load_expansion_cache(path: Path | None) -> dict[str, list[str]]:
    """Load the committed expansion cache (query -> gold union), or ``{}`` if absent."""
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_indices(text: str, n: int) -> list[int]:
    try:
        data = json.loads(text[text.index("[") : text.rindex("]") + 1])
        return [int(x) for x in data if isinstance(x, (int, float)) and 0 <= int(x) < n]
    except (ValueError, TypeError):
        return [int(m) for m in _INDEX_RE.findall(text) if 0 <= int(m) < n]
