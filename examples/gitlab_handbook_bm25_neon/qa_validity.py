"""Answerability / validity filter for generated qa pairs (blind of retrieval).

The spine of the fix-round data-quality control: a strict LLM judge decides, from
the QUERY TEXT and the LABELED GOLD CONTENT ALONE, whether a pair is answerable and
self-contained — never from whether Neon retrieves the gold. Hardness is not a
defect: a self-contained query whose gold genuinely answers it is KEPT even if it
would retrieve poorly (a paraphrase with 0 lexical overlap, an answer buried in a
mermaid diagram). Only genuinely defective pairs are dropped, each with a recorded
reason and a one-line citation of the defect, so the drop log is auditable.

Scope (important): only NATURAL-LANGUAGE rows are judged. The curated
retrieval-precision probes (a single distinctive token or a bag of rare words plus
a metadata filter) are answerable *by construction* — the token is unique-in-section
and the gold literally contains it — so a natural-language answerability rubric
mis-flags them "unanswerable". Judging them would wrongly drop valid probes, so they
are exempt here; their validity is enforced by their construction and the live
dual-leg / filter ablations in the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from castform.platform.credentials import resolve_judge_key
from castform.rag.chunkers.models import Chunk, ChunkCollection
from castform.rag.corpus.neon.eval_schema import NeonEvalRecord

JUDGE_MODEL = "gpt-5.4-mini"

# Capabilities judged for answerability. Everything else (curated single-token /
# bag-of-words filter + hybrid probes) is answerable-by-construction and exempt.
NATURAL_CAPABILITIES = frozenset({"vector_lookup", "lexical_lookup"})

DROP_REASONS = (
    "stripped-anchor",
    "wrong-gold",
    "non-unique-templated-answer",
    "unanswerable",
)

# Verbatim the rubric validated by the answerability audit (3.4% defective / 117,
# zero over-drops of valid-but-hard pairs).
_SYSTEM = (
    "You are a STRICT retrieval-QA answerability auditor. You are given ONE query "
    "and the LABELED GOLD chunk(s) it is supposed to retrieve from a corpus (the "
    "GitLab public handbook, ~31,665 chunks). Judge ONLY whether the pair is "
    "answerable/valid FROM THE QUERY TEXT AND THE GOLD CONTENT. You must NOT "
    "consider whether any particular search engine would rank the gold highly — "
    "hardness is NOT a defect.\n\n"
    'Return a JSON object: {"verdict": "KEEP"|"DROP", "reason": "<one of: valid | '
    'stripped-anchor | wrong-gold | non-unique-templated-answer | unanswerable>", '
    '"citation": "<one concrete line: which disambiguating anchor was removed, or '
    'why the gold does not answer, or why non-unique>", "confidence": 0.0-1.0}\n\n'
    "DROP the pair ONLY if one of these is clearly true:\n"
    "(a) STRIPPED-ANCHOR: the gold's answer is keyed to a SPECIFIC proper noun / "
    "person name / product name / unique identifier, but the query elides that "
    'anchor and refers to it only through a GENERIC placeholder ("this engineering '
    'contact", "a chosen codebase", "the built-in tool") such that the query would '
    "equally describe MANY different chunks — no retriever could pick THIS single "
    "gold from the query alone. Paraphrase alone is NOT a defect; the defect is that "
    "the ONE distinguishing token is gone and what remains is generic.\n"
    "(b) WRONG-GOLD: the labeled gold chunk does not actually contain the answer to "
    "the query.\n"
    "(c) NON-UNIQUE-TEMPLATED-ANSWER: the answer is boilerplate/templated text that "
    "recurs near-identically across many chunks, so no unique gold exists.\n"
    "(d) UNANSWERABLE: the query is not answerable from the gold as written.\n\n"
    "KEEP the pair if the query is SELF-CONTAINED (carries enough distinctive "
    "conceptual content — a specific procedure, field name, config key, outcome, or "
    "named concept — to point at this gold) AND the gold genuinely answers it. KEEP "
    "even if the query is lexically dissimilar to the gold and would retrieve poorly "
    "(valid-but-hard = KEEP).\n\n"
    "DEFAULT TO KEEP when genuinely unsure. Only DROP when the defect is clear. "
    "Output ONLY the JSON object, nothing else."
)


@dataclass(frozen=True)
class Verdict:
    """One answerability verdict.

    Args:
        verdict: ``"KEEP"`` or ``"DROP"``.
        reason: ``"valid"`` or one of :data:`DROP_REASONS`.
        citation: One concrete line naming the defect (or why it is valid).
        confidence: Judge self-reported confidence in ``[0, 1]``.
    """

    verdict: str
    reason: str
    citation: str
    confidence: float


def _render_gold(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        md = dict(c.metadata)
        head = f"file={md.get('file', '')} section={md.get('handbook_section', '')}"
        blocks.append(f"[GOLD {i}] {head}\n{c.content}")
    return "\n\n".join(blocks)


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
        import json

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
            return Verdict("KEEP", "valid", "judge unavailable — kept", 0.0)
        verdict = "DROP" if str(data.get("verdict", "")).upper() == "DROP" else "KEEP"
        reason = str(data.get("reason", "valid"))
        if verdict == "DROP" and reason not in DROP_REASONS:
            reason = "unanswerable"
        return Verdict(
            verdict=verdict,
            reason=reason if verdict == "DROP" else "valid",
            citation=str(data.get("citation", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )


def filter_records(
    records: list[NeonEvalRecord],
    collection: ChunkCollection,
    *,
    base_url: str,
    model: str = JUDGE_MODEL,
) -> tuple[list[NeonEvalRecord], list[dict], QaValidityJudge]:
    """Drop defective natural-language pairs; return ``(kept, dropped, judge)``.

    Curated probe capabilities (anything outside :data:`NATURAL_CAPABILITIES`) are
    kept unjudged. Each dropped entry records mode, capability, query, gold hash,
    reason, citation, and confidence for the audit manifest.

    Args:
        records: The generated eval records to screen.
        collection: In-memory corpus, the source of gold content (blind of Neon).
        base_url: Judge endpoint.
        model: Judge model id.
    """
    by_hash = {c.hash: c for c in collection.chunks}
    judge = QaValidityJudge(base_url=base_url, model=model)
    kept: list[NeonEvalRecord] = []
    dropped: list[dict] = []
    for rec in records:
        if rec.capability not in NATURAL_CAPABILITIES:
            kept.append(rec)
            continue
        gold_chunks = [by_hash[h] for h in rec.gold_chunk_hashes if h in by_hash]
        if not gold_chunks:
            kept.append(rec)  # cannot judge without gold content; keep
            continue
        v = judge.judge(rec.query, gold_chunks)
        if v.verdict == "DROP":
            dropped.append(
                {
                    "search_mode": rec.search_mode,
                    "capability": rec.capability,
                    "query": rec.query,
                    "gold_chunk_hash": rec.gold_chunk_hashes[0],
                    "reason": v.reason,
                    "citation": v.citation,
                    "confidence": v.confidence,
                }
            )
        else:
            kept.append(rec)
    return kept, dropped, judge
