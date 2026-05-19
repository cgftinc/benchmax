"""Quality gate filter — cheap heuristic checks run before expensive LLM filters."""

from __future__ import annotations

import re

from benchmax.rag.qa_generation.pipeline_config import PipelineContext, QualityGateConfig
from benchmax.rag.qa_generation.generated_qa import FilterVerdict, GeneratedQA

# --------------------------------------------------------------------------- #
# Check 1 — Fragment detection signals
# --------------------------------------------------------------------------- #

_COMMON_VERBS = re.compile(
    r"\b("
    r"is|are|do|can|get|set|use|want|need|make|run|find|show|check|configure"
    r"|enable|install|deploy|connect|handle|ensure|prevent|create|update|delete"
    r"|manage|build|start|stop|add|remove|change|have|has|work|support|allow"
    r"|provide|require|include|send|receive|call|return|save|load|read|write"
    r"|open|close|move|copy|link|push|pull|fetch|post|put|go|come|see|does|did"
    r"|should|will|would|could|might|must|shall|may"
    r"|capture|mark|track|log|record|test|validate|verify|debug|analyze|monitor"
    r"|display|render|filter|sort|export|import|migrate|sync|authenticate|sign"
    r"|click|select|define|specify|implement|fix|resolve|submit|reset|clear"
    r")\b",
    re.I,
)

_INTERROGATIVE_START = re.compile(
    r"^(how|what|why|when|where|which|can|should|does)\b",
    re.I,
)

_SCENARIO_FRAMING = re.compile(
    r"\b(I|you|a\s+team|my|our|we|the)\b",
    re.I,
)

# Coordinating and subordinating conjunctions that indicate a clause boundary.
# Prepositions (after, before, with, in, etc.) are intentionally excluded so
# that short keyword queries like "webhook retry config after upgrade" still
# pass as valid search-style questions.
_CLAUSE_CONNECTOR = re.compile(
    r"\b(and|or|but|nor|yet|so|while|because|when|if|since|though|although|unless|until)\b",
    re.I,
)


def _fragment_signals(question: str) -> tuple[bool, bool, bool]:
    """Return (signal_a, signal_b, signal_c) for fragment detection.

    Signals B and C only fire when the phrase contains a clause connector
    (conjunction). This ensures pure keyword queries like "posthog cli ci api key"
    or "CMS auto-capture visitor behavior every page" trigger at most 1 signal
    and are therefore not rejected as fragments.
    """
    has_verb = bool(_COMMON_VERBS.search(question))
    has_interrogative = bool(_INTERROGATIVE_START.match(question))
    has_scenario = bool(_SCENARIO_FRAMING.search(question))
    has_connector = bool(_CLAUSE_CONNECTOR.search(question))
    word_count = len(question.split())
    has_question_mark = "?" in question

    signal_a = not has_verb
    # B requires a connector so bare noun/keyword phrases don't fire.
    signal_b = not has_interrogative and not has_scenario and has_connector
    # C also requires a connector for the same reason.
    signal_c = word_count <= 8 and not has_question_mark and not has_scenario and has_connector

    return signal_a, signal_b, signal_c


# --------------------------------------------------------------------------- #
# Check 2 — Structural / meta-doc query
# --------------------------------------------------------------------------- #

_STRUCTURAL_PATTERNS = [
    re.compile(r"\bmain\s+headings?\b", re.I),
    re.compile(r"\btable\s+of\s+contents?\b", re.I),
    re.compile(r"\bsections?\s+on\s+that\s+page\b", re.I),
    re.compile(r"\bpage\s+(layout|structure|outline)\b", re.I),
    re.compile(r"\blist\s+the\s+main\s+(sections?|topics?|headings?)\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+main\s+(steps?|sections?|headings?)\b", re.I),
]


def _is_structural(question: str) -> bool:
    return any(p.search(question) for p in _STRUCTURAL_PATTERNS)


# --------------------------------------------------------------------------- #
# Check 3 — Guide pointer
# --------------------------------------------------------------------------- #

_GUIDE_POINTER_QUESTION_PATTERNS = [
    re.compile(r"\bwhere\s+can\s+I\s+(find|look|read|start)\b", re.I),
    re.compile(r"\bwhich\s+(guide|doc|page)\s+should\b", re.I),
    re.compile(r"\bstarting\s+point\s+for\b", re.I),
    re.compile(r"\bpoint\s+me\s+(to|toward)\b", re.I),
    re.compile(r"\brecommend\s+a\s+(guide|doc)\b", re.I),
]

_GUIDE_POINTER_ANSWER_PATTERNS = [
    re.compile(r"\bcheck\s+out\s+the\s+(docs?|guide)\b", re.I),
    re.compile(r"\bsee\s+the\s+documentation\b", re.I),
    re.compile(r"\bread\s+(the|our)\b", re.I),
    re.compile(r"\bgreat\s+starting\s+point\b", re.I),
]


def _is_guide_pointer(question: str, answer: str) -> bool:
    if any(p.search(question) for p in _GUIDE_POINTER_QUESTION_PATTERNS):
        # If answer gives a concrete navigation path, it's a real question — not a guide pointer
        if re.search(r"\bSettings\b.*>|\bGo\s+to\b|\bNavigate\s+to\b", answer, re.I):
            return False
        return True
    if any(p.search(answer) for p in _GUIDE_POINTER_ANSWER_PATTERNS):
        return True
    return False


# --------------------------------------------------------------------------- #
# Check 4 — Thin answer
# --------------------------------------------------------------------------- #


def _thin_answer_hint(question: str, answer: str) -> str | None:
    q_words = len(question.split())
    a_words = len(answer.split())

    if a_words < 15 and a_words < q_words * 0.5:
        return "Provide a more detailed, specific answer grounded in chunk evidence."

    q_sentences = len(re.split(r"[.!?]+", question.strip()))
    a_sentences = len(re.split(r"[.!?]+", answer.strip()))
    if q_sentences > 1 and a_sentences <= 1 and a_words < 20:
        return "Provide a more detailed, specific answer grounded in chunk evidence."

    return None


# --------------------------------------------------------------------------- #
# Filter class
# --------------------------------------------------------------------------- #


class QualityGateFilter:
    """Cheap heuristic quality gate — runs before LLM filters to save cost."""

    def __init__(self, cfg: QualityGateConfig) -> None:
        self.cfg = cfg

    def evaluate(self, items: list[GeneratedQA], context: PipelineContext) -> list[GeneratedQA]:
        if not self.cfg.enabled:
            return items

        stats = context.setdefault(
            "quality_gate_stats",
            {"passed": 0, "rejected": 0, "needs_refinement": 0},
        )
        for item in items:
            if item.filter_verdict is not None:
                continue

            question = str(item.qa.get("question", "")).strip()
            answer = str(item.qa.get("answer", "")).strip()

            reject_reason = self._check_reject(question, answer)
            if reject_reason:
                item.filter_verdict = FilterVerdict(
                    status="rejected",
                    reason=reject_reason,
                    reasoning=f"Quality gate rejected: {reject_reason}",
                    metadata={
                        "filter_mode": "quality_gate",
                        "reason_code": reject_reason,
                        "confidence": 1.0,
                        "retrieval_query": str(item.qa.get("retrieval_query", "")).strip()
                        or question,
                        "ref_overlap_ratio": None,
                        "feedback_type": None,
                        "refinement_hint": None,
                    },
                )
                stats["rejected"] = int(stats.get("rejected", 0)) + 1
                continue

            refinement_hint = self._check_thin(question, answer)
            if refinement_hint:
                item.filter_verdict = FilterVerdict(
                    status="needs_refinement",
                    reason="quality_gate_thin_answer",
                    reasoning="Answer is too thin relative to the question.",
                    metadata={
                        "filter_mode": "quality_gate",
                        "reason_code": "quality_gate_thin_answer",
                        "confidence": 1.0,
                        "retrieval_query": str(item.qa.get("retrieval_query", "")).strip()
                        or question,
                        "ref_overlap_ratio": None,
                        "feedback_type": "needs_refinement",
                        "refinement_hint": refinement_hint,
                    },
                )
                stats["needs_refinement"] = int(stats.get("needs_refinement", 0)) + 1
                continue

            item.filter_verdict = FilterVerdict(
                status="passed",
                reason="quality_gate_passed",
                reasoning="Passed quality gate.",
                metadata={
                    "filter_mode": "quality_gate",
                    "reason_code": "quality_gate_passed",
                    "confidence": 1.0,
                    "retrieval_query": str(item.qa.get("retrieval_query", "")).strip() or question,
                    "ref_overlap_ratio": None,
                    "feedback_type": None,
                    "refinement_hint": None,
                },
            )
            stats["passed"] = int(stats.get("passed", 0)) + 1

        return items

    def _check_reject(self, question: str, answer: str) -> str | None:
        if self.cfg.reject_fragments:
            sig_a, sig_b, sig_c = _fragment_signals(question)
            if sum([sig_a, sig_b, sig_c]) >= 2:
                return "quality_gate_fragment"

        if self.cfg.reject_structural and _is_structural(question):
            return "quality_gate_structural"

        if self.cfg.reject_guide_pointers and _is_guide_pointer(question, answer):
            return "quality_gate_guide_pointer"

        return None

    def _check_thin(self, question: str, answer: str) -> str | None:
        if not self.cfg.refine_thin_answers:
            return None
        return _thin_answer_hint(question, answer)
