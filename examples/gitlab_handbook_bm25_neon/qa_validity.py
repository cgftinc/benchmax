"""Answerability / validity filter for generated qa pairs (blind of retrieval).

The spine of the fix-round data-quality control: a strict LLM judge decides, from
the QUERY TEXT and the LABELED GOLD CONTENT ALONE, whether a pair is answerable and
self-contained — never from whether Neon retrieves the gold. Hardness is not a
defect: a self-contained query whose gold genuinely answers it is KEPT even if it
would retrieve poorly (a paraphrase with 0 lexical overlap, an answer buried in a
mermaid diagram). Only genuinely defective pairs are DROPPED, each with a recorded
reason and a citation of the defect, so the drop log is auditable.

Verdicts are three-way. KEEP and FIX are both RETAINED (only DROP removes a pair):

* ``KEEP`` — clean: self-contained query, gold genuinely answers it.
* ``FIX`` — retained but flagged: answerable and self-contained (so NOT a defect),
  yet with a minor non-fatal issue a human author might polish (mildly awkward
  phrasing, slight ambiguity, gold answers only partially). Never dropped; the flag
  is advisory signal for downstream review.
* ``DROP`` — one of the four clear intrinsic defects in :data:`DROP_REASONS`.

Every verdict carries a ``citation``: for KEEP/FIX it QUOTES the gold line that
answers the query (positive grounding); for DROP it names the defect. Judging is
cacheable (:func:`judge_records` accepts a ``cache``) so the same single judge pass
feeds both the standalone ``verdicts_v2`` audit and the frozen-set build without a
re-spend, and the frozen-set drops reconcile exactly with the audit file.

Scope (important): only NATURAL-LANGUAGE rows are judged. The curated
retrieval-precision probes (a single distinctive token or a bag of rare words plus
a metadata filter) are answerable *by construction* — the token is unique-in-section
and the gold literally contains it — so a natural-language answerability rubric
mis-flags them "unanswerable". Judging them would wrongly drop valid probes, so they
are exempt here; their validity is enforced by their construction and the live
dual-leg / filter ablations in the gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from castform.platform.credentials import resolve_judge_key
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

JUDGE_MODEL = "gpt-5.4-mini"

# Capabilities judged for answerability. Everything else (curated single-token /
# bag-of-words filter + hybrid probes) is answerable-by-construction and exempt.
NATURAL_CAPABILITIES = frozenset({"vector_lookup", "lexical_lookup"})

# The four intrinsic defects that cause a DROP (Step-3 validity filter). A pair is
# dropped ONLY for one of these; retrieval difficulty is never a drop reason.
DROP_REASONS = (
    "stripped-anchor",  # entity-stripped: the one disambiguating anchor is gone
    "wrong-gold",  # gold-doesn't-answer
    "non-unique-templated-answer",  # templated answer with no unique gold
    "unanswerable",  # not answerable from the gold as written
)

_VERDICTS = ("keep", "fix", "drop")

# First N chars of gold content recorded per verdict so the audit file is
# self-describing to an offline cross-vendor reviewer (no corpus needed).
_GOLD_PREVIEW_CHARS = 280

_SYSTEM = (
    "You are a STRICT retrieval-QA answerability auditor. You are given ONE query "
    "and the LABELED GOLD chunk(s) it is supposed to retrieve from a corpus (the "
    "GitLab public handbook, ~31,665 chunks). Judge ONLY whether the pair is "
    "answerable/valid FROM THE QUERY TEXT AND THE GOLD CONTENT. You must NOT "
    "consider whether any particular search engine would rank the gold highly — "
    "hardness is NOT a defect.\n\n"
    'Return a JSON object: {"verdict": "keep"|"fix"|"drop", "reason": "<see below>", '
    '"citation": "<for keep/fix: the exact gold sentence or phrase, quoted, that '
    'answers the query; for drop: which disambiguating anchor was removed, or why '
    'the gold does not answer, or why the answer is non-unique>", "confidence": '
    "0.0-1.0}\n\n"
    "DROP the pair ONLY if one of these is clearly true (reason = the matching "
    "token):\n"
    "(stripped-anchor) the gold's answer is keyed to a SPECIFIC proper noun / "
    "person name / product name / unique identifier, but the query elides that "
    'anchor and refers to it only through a GENERIC placeholder ("this engineering '
    'contact", "a chosen codebase", "the built-in tool") such that the query would '
    "equally describe MANY different chunks — no retriever could pick THIS single "
    "gold from the query alone. Paraphrase alone is NOT a defect; the defect is that "
    "the ONE distinguishing token is gone and what remains is generic.\n"
    "(wrong-gold) the labeled gold chunk does not actually contain the answer to "
    "the query.\n"
    "(non-unique-templated-answer) the answer is boilerplate/templated text that "
    "recurs near-identically across many chunks, so no unique gold exists.\n"
    "(unanswerable) the query is not answerable from the gold as written.\n\n"
    "Otherwise the pair is RETAINED. Choose between:\n"
    "(keep) the query is SELF-CONTAINED (carries enough distinctive conceptual "
    "content — a specific procedure, field name, config key, outcome, or named "
    "concept — to point at this gold) AND the gold genuinely answers it. KEEP even "
    "if the query is lexically dissimilar to the gold and would retrieve poorly "
    "(valid-but-hard = keep). reason = valid.\n"
    "(fix) same as keep — answerable and self-contained, NOT a defect — but with a "
    "minor non-fatal issue a human author might polish: mildly awkward phrasing, "
    "slight ambiguity, or the gold answers only partially. reason = a short phrase "
    "naming the issue. This is still RETAINED; use it only as advisory signal.\n\n"
    "DEFAULT TO KEEP when genuinely unsure. Only DROP when the defect is clear; only "
    "FIX when a real minor issue exists. Output ONLY the JSON object, nothing else."
)


@dataclass(frozen=True)
class Verdict:
    """One answerability verdict.

    Args:
        verdict: ``"keep"``, ``"fix"``, or ``"drop"``. KEEP and FIX are retained;
            only DROP removes the pair.
        reason: ``"valid"`` (keep), a short issue phrase (fix), or one of
            :data:`DROP_REASONS` (drop).
        citation: For keep/fix, the quoted gold line that answers the query; for
            drop, the concrete defect.
        confidence: Judge self-reported confidence in ``[0, 1]``.
    """

    verdict: str
    reason: str
    citation: str
    confidence: float

    @property
    def is_drop(self) -> bool:
        """Whether this verdict removes the pair from the golden set."""
        return self.verdict == "drop"


def _render_gold(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        md = dict(c.metadata)
        head = f"file={md.get('file', '')} section={md.get('handbook_section', '')}"
        blocks.append(f"[GOLD {i}] {head}\n{c.content}")
    return "\n\n".join(blocks)


def pair_key(record: NeonEvalRecord) -> str:
    """Stable identity for a (query, gold-set) pair, for verdict caching.

    Uses the query and the sorted gold hashes as they are at JUDGE time (before any
    multi-gold / equivalence expansion), so the same pair keys the same verdict in
    the standalone audit run and the frozen-set build.
    """
    return record.query + "\x00" + "|".join(sorted(record.gold_chunk_hashes))


class QaValidityJudge:
    """Strict answerability judge over ``llm.<domain>`` (one call per pair).

    Args:
        base_url: OpenAI-compatible endpoint (pinned to the platform).
        model: Judge model id.
    """

    def __init__(self, *, base_url: str, model: str = JUDGE_MODEL) -> None:
        self._base_url = base_url
        self._model = model
        self._client = None
        self.resolved_model: str | None = None

    def judge(self, query: str, gold_chunks: list[Chunk]) -> Verdict:
        """Return the answerability verdict for one (query, gold) pair.

        On any judge error the pair is KEPT (``valid``) — the filter must never drop
        a pair because the judge was unavailable, only on a clear defect.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=resolve_judge_key("", self._base_url),
            )
        user = f"SEARCH QUERY:\n{query}\n\nLABELED GOLD:\n{_render_gold(gold_chunks)}"
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            self.resolved_model = resp.model or self.resolved_model
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception:  # never drop a pair on judge failure
            return Verdict("keep", "valid", "judge unavailable — kept", 0.0)
        verdict = str(data.get("verdict", "keep")).strip().lower()
        if verdict not in _VERDICTS:
            verdict = "keep"
        reason = str(data.get("reason", "valid"))
        if verdict == "drop" and reason not in DROP_REASONS:
            reason = "unanswerable"
        if verdict == "keep":
            reason = "valid"
        return Verdict(
            verdict=verdict,
            reason=reason,
            citation=str(data.get("citation", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )


def make_verdict_row(
    record: NeonEvalRecord,
    gold_chunks: list[Chunk],
    *,
    natural: bool,
    verdict: Verdict,
) -> dict:
    """Build a self-describing audit row for one record's verdict.

    Args:
        record: The eval record judged.
        gold_chunks: Resolved gold chunks (first supplies the preview); may be empty.
        natural: Whether the row was judged (natural-language) or exempt (curated).
        verdict: The verdict to serialize.
    """
    preview = ""
    if gold_chunks:
        preview = gold_chunks[0].content[:_GOLD_PREVIEW_CHARS]
    return {
        "capability": record.capability,
        "search_mode": record.search_mode,
        "query": record.query,
        "gold_chunk_hashes": list(record.gold_chunk_hashes),
        "natural": natural,
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "citation": verdict.citation,
        "confidence": verdict.confidence,
        "gold_preview": preview,
    }


def judge_records(
    records: list[NeonEvalRecord],
    collection: ChunkCollection,
    *,
    base_url: str,
    model: str = JUDGE_MODEL,
    cache: dict[str, Verdict] | None = None,
) -> tuple[list[dict], QaValidityJudge]:
    """Judge every NATURAL row; return ``(verdict_rows, judge)``.

    One self-describing verdict row is emitted PER record (natural and exempt), so
    the returned list fully accounts for the input. Natural rows are judged (or
    served from ``cache`` when present, avoiding a re-spend); curated probes are
    recorded as ``verdict="keep"`` / ``natural=False`` with an exempt reason and are
    never judged. A natural row with no resolvable gold content is kept unjudged.

    Args:
        records: The generated eval records to screen.
        collection: In-memory corpus, the source of gold content (blind of Neon).
        base_url: Judge endpoint.
        model: Judge model id.
        cache: Optional ``pair_key -> Verdict`` map to reuse prior verdicts.
    """
    by_hash = {c.hash: c for c in collection.chunks}
    judge = QaValidityJudge(base_url=base_url, model=model)
    cache = cache or {}
    rows: list[dict] = []
    for rec in records:
        gold_chunks = [by_hash[h] for h in rec.gold_chunk_hashes if h in by_hash]
        if rec.capability not in NATURAL_CAPABILITIES:
            v = Verdict(
                "keep", "answerable-by-construction (curated probe, exempt)", "", 1.0
            )
            rows.append(make_verdict_row(rec, gold_chunks, natural=False, verdict=v))
            continue
        if not gold_chunks:
            v = Verdict("keep", "valid", "no gold content resolvable — kept", 0.0)
            rows.append(make_verdict_row(rec, gold_chunks, natural=True, verdict=v))
            continue
        v = cache.get(pair_key(rec)) or judge.judge(rec.query, gold_chunks)
        rows.append(make_verdict_row(rec, gold_chunks, natural=True, verdict=v))
    return rows, judge


def apply_validity(
    records: list[NeonEvalRecord], verdict_rows: list[dict]
) -> tuple[list[NeonEvalRecord], list[dict]]:
    """Split ``records`` into ``(kept, dropped)`` using ``verdict_rows``.

    Drops ONLY records whose verdict is ``drop`` (one of :data:`DROP_REASONS`); KEEP,
    FIX, and exempt rows are all retained. Records and verdict rows must align 1:1 in
    order (as returned by :func:`judge_records`).

    Args:
        records: The judged records, in the same order as ``verdict_rows``.
        verdict_rows: The per-record verdicts from :func:`judge_records`.
    """
    if len(records) != len(verdict_rows):
        raise ValueError("records and verdict_rows must align 1:1")
    kept: list[NeonEvalRecord] = []
    dropped: list[dict] = []
    for rec, row in zip(records, verdict_rows, strict=True):
        if row["verdict"] == "drop":
            dropped.append(
                {
                    "search_mode": rec.search_mode,
                    "capability": rec.capability,
                    "query": rec.query,
                    "gold_chunk_hash": rec.gold_chunk_hashes[0],
                    "reason": row["reason"],
                    "citation": row["citation"],
                    "confidence": row["confidence"],
                }
            )
        else:
            kept.append(rec)
    return kept, dropped


def load_verdict_cache(path: Path) -> dict[str, Verdict]:
    """Load a ``verdicts_v2`` jsonl into a ``pair_key -> Verdict`` cache.

    Only natural (judged) rows are cached; exempt curated rows are re-derived on the
    fly. Missing / empty file yields an empty cache.
    """
    cache: dict[str, Verdict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("natural", False):
            continue
        key = row["query"] + "\x00" + "|".join(sorted(row.get("gold_chunk_hashes", [])))
        cache[key] = Verdict(
            verdict=str(row.get("verdict", "keep")),
            reason=str(row.get("reason", "valid")),
            citation=str(row.get("citation", "")),
            confidence=float(row.get("confidence", 0.0) or 0.0),
        )
    return cache


def filter_records(
    records: list[NeonEvalRecord],
    collection: ChunkCollection,
    *,
    base_url: str,
    model: str = JUDGE_MODEL,
    cache: dict[str, Verdict] | None = None,
) -> tuple[list[NeonEvalRecord], list[dict], QaValidityJudge, list[dict]]:
    """Drop defective natural-language pairs; return ``(kept, dropped, judge, verdicts)``.

    Convenience over :func:`judge_records` + :func:`apply_validity`. ``verdicts`` is
    the full per-record audit (natural + exempt) suitable for ``verdicts_v2``.

    Args:
        records: The generated eval records to screen.
        collection: In-memory corpus, the source of gold content (blind of Neon).
        base_url: Judge endpoint.
        model: Judge model id.
        cache: Optional ``pair_key -> Verdict`` map to reuse prior verdicts.
    """
    verdicts, judge = judge_records(
        records, collection, base_url=base_url, model=model, cache=cache
    )
    kept, dropped = apply_validity(records, verdicts)
    return kept, dropped, judge, verdicts
