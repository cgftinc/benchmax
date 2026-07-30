"""Derived, mode-agnostic query-style proxy.

The 430 A/B rows carry NO style field — their only keys are ``question`` /
``answer`` / ``reference_chunks``. Any per-row style label is therefore
**derived**, not ground truth, and this module is the single place the derivation
lives so the report and the harness can never disagree about the rule.

The rule deliberately reads the QUESTION TEXT ONLY. It never looks at gold, at
retrieval output, or at any per-row search mode, so it cannot leak the outcome it
is used to slice. (In particular it does NOT reuse the 156-row
``gitlab_handbook_bm25_neon`` golden, which bakes a favourable ``search_mode``
into each row — exactly the confound this A/B removes.)

Rule
----
A question is labelled ``keyword`` when ALL THREE hold:

1. it is short — fewer than :data:`SHORT_WORD_LIMIT` whitespace tokens;
2. its first token is not an interrogative (``what``/``where``/...) or an
   auxiliary/modal (``is``/``do``/``can``/...);
3. it contains no first- or second-person pronoun.

Otherwise it is labelled ``paraphrase``. The intent is to separate terse
bag-of-terms lookups ("HAProxy CloudFlare firewall rules Cells") from full
natural-language questions ("After I've approved the cost in Brilliant, what do I
need to add ...?"). It is a proxy: individual rows can be mislabelled, and the
bucket sizes must be reported alongside any per-bucket metric.
"""

from __future__ import annotations

SHORT_WORD_LIMIT = 12
"""Token count below which a question counts as short (criterion 1)."""

INTERROGATIVE_LEADS = frozenset(
    {"what", "where", "when", "who", "whom", "whose", "why", "how", "which"}
)
"""Wh-words that, in lead position, mark a natural-language question."""

AUXILIARY_LEADS = frozenset(
    {
        "is", "are", "was", "were", "am", "be", "been",
        "do", "does", "did",
        "has", "have", "had",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    }
)
"""Auxiliaries/modals that, in lead position, mark a subject-inverted question."""

PERSONAL_PRONOUNS = frozenset(
    {
        "i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself",
        "we", "we're", "we've", "we'd", "we'll", "us", "our", "ours",
        "you", "you're", "you've", "you'd", "you'll", "your", "yours",
    }
)
"""First/second-person pronouns; their presence marks conversational phrasing."""

_STRIP = ".,?!:;\"'()[]{}"

# Typographic apostrophes appear in the generated questions; fold them so
# "I've" and "I’ve" hit the same pronoun entry.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'"})


def tokenize(question: str) -> list[str]:
    """Split *question* into lowercase, punctuation-stripped whitespace tokens."""
    out: list[str] = []
    for raw in question.translate(_APOSTROPHES).split():
        token = raw.lower().strip(_STRIP)
        if token:
            out.append(token)
    return out


def classify(question: str) -> str:
    """Return the derived style label for *question* — ``keyword`` or ``paraphrase``.

    See the module docstring for the exact rule. A question with no tokens at all
    is labelled ``paraphrase`` (it cannot satisfy the lead-token criterion).
    """
    tokens = tokenize(question)
    if not tokens:
        return "paraphrase"
    is_short = len(tokens) < SHORT_WORD_LIMIT
    plain_lead = tokens[0] not in INTERROGATIVE_LEADS and tokens[0] not in AUXILIARY_LEADS
    impersonal = not (set(tokens) & PERSONAL_PRONOUNS)
    return "keyword" if (is_short and plain_lead and impersonal) else "paraphrase"
