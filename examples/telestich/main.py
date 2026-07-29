"""TelestichEnv — reward env for telestich (last-letter acrostic) poems.

A telestich hides a target word in the last letter of each line. Reward per
rollout in a GRPO group (see ``compute_group_rewards``):

  1. HARD RULES (no LLM): a perfect telestich is CORRECT and goes to the judge.
     A near-miss (answer present, right line count, no cheat, >25% of lines
     correct) is PARTIAL — a small graded reward so there's always a gradient.
     Cheating / no answer / wrong line count / ≤25% correct → 0.
  2. QUALITY: the shared multi-anchor rubric judge ranks the CORRECT poems against
     acceptable + great reference anchors, in batches of <=JUDGE_BATCH poems per call
     (ranking the whole group at once lets siblings contaminate each other's score).
  3. BONUSES (positive, quality-scaled): rhyme (any correct poem), plus diversity
     and conciseness (shorter generation + tool thrift) on the group's top band.

Final reward = quality + rhyme + diversity + conciseness (all components >= 0).

`python main.py validate` splits the committed dataset, uploads the dataset and
bundle, and runs local and hosted checks. `python main.py launch` follows the
same path before asking to spend credits.

Import-safe: stages run only from the ``if __name__ == "__main__"`` block.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pronouncing  # pyright: ignore[reportMissingImports]
from benchmax.bundle import dump_bundle
from benchmax.envs import (
    BaseEnv,
    BaseRollout,
    DatasetSplit,
    Example,
    InjectedAuth,
    JsonlDataset,
    JsonRow,
    Messages,
    ModelAuth,
    Tool,
    canonical_example_id,
)
from benchmax.envs.base import resolve_dataset_path
from benchmax.envs.logging import rollout_context as bind_rollout_context
from benchmax.rewards import (
    Judge,
    RankingAnchor,
    Rubric,
    evaluate_rubric_ranking,
    extract_completion_text,
)
from castform.platform import ensure_session, upload_assets
from english_words import get_english_words_set  # pyright: ignore[reportMissingImports]
from wordfreq import word_frequency  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
JUDGE_MODEL = "gpt-5.4"
# Max model poems ranked against the anchors in ONE judge call. Ranking a whole GRPO
# group at once lets near-identical siblings contaminate each other's placement and the
# judge over-rates them above the anchors; batches of 3 fix it (run 0ec8e2dc audit:
# full-group 15/36 false-better vs batch-of-3 0/36, ~83% agreement with a Sonnet audit).
JUDGE_BATCH = 3

# Multi-anchor ELO judge (default; replaces the disjoint JUDGE_BATCH ranker). Each judge
# call ranks a small SLICE of poems against BOTH anchors (kept in every call); poems sit
# in overlapping slices so a Bradley-Terry fit — with the anchors pinned as fixed-rating
# reference players — recovers one calibrated ranking. A single 9+2 ranking lets the judge
# float poems above the blind anchors (~20% of poems spuriously above baseline); slice+BT
# holds the scale (~1%) — golden audit over 30 prompts (M3 vs M1).
ELO_JUDGE = True  # False → fall back to the JUDGE_BATCH disjoint-batch ranker
ELO_SLICE = 5  # model poems per judge call; +2 anchors = 7 items (judge-reliable range)
ELO_STRIDE = 3  # slice stride; adjacent slices overlap by ELO_SLICE-ELO_STRIDE poems
ELO_GREAT_RATING = 1.2  # BT-rating gap acceptable(0.0)→great; P(great>acc)≈0.77
ELO_BT_ITERS = 500
ELO_BT_LR = 0.1

# CORRECT poems are ranked by the multi-anchor rubric judge against `acceptable`
# and `great` reference anchors; the band score maps onto a reward ladder
# (_map_correct_score): below acceptable → [0.1,0.4) "below", acceptable..great →
# [0.4,0.7] "mid", above great → (0.7,1.0] "above".
ACCEPTABLE_EDGE = 0.4  # quality for tying the acceptable ("normal") anchor
GREAT_EDGE = 0.7  # quality for tying the great anchor
MIN_CORRECT = 0.1  # floor for a correct poem ranked dead-last
# Partial credit for a near-miss (right line count, real poem lines, no cheat) with
# >MIN_CORRECT_FRAC of lines correct. Reward scales PARTIAL_HI (1 miss) → PARTIAL_LO
# (most misses still above the bar). Ordering: gated 0 < partial < worst correct 0.1.
PARTIAL_HI = 0.05  # best partial (exactly one miss)
PARTIAL_LO = 0.01  # worst partial (most misses still above the bar)
MIN_CORRECT_FRAC = 0.25  # need MORE than 25% of lines correct to earn partial credit
QUALITY_RUBRIC = Rubric(
    title="poetic quality",
    description=(
        "Judge the poem's craft only (the telestich spelling is already verified). TWO "
        "axes dominate, equally weighted:\n"
        "(1) FIT TO THE BRIEF. The poem must honor the request's subject, occasion, and "
        "especially its VOICE / TONE / REGISTER (e.g. casual or slangy, terse, formal, "
        "playful, archaic). A poem that abandons the asked-for voice for generic "
        "'poetic' prettiness is LOW quality no matter how vivid. Do NOT reward ornate, "
        "image-stuffed, or literary-sounding writing as a substitute for doing what was "
        "asked — flowery imagery is not a virtue when the brief wants something plain, "
        "blunt, funny, or colloquial. A short, plain poem that nails the requested voice "
        "BEATS a lavish one that drifts off-voice.\n"
        "(2) COHERENCE. The lines must connect into one deliberate poem with a real "
        "through-line of sense. A line that is a non-sequitur, or whose ENDING word "
        "is clearly bolted on just to land its letter so the line stops making sense "
        "(e.g. ending on 'you', 'so', 'rev', 'ref' where it doesn't fit), makes the "
        "poem INCOHERENT — rank such a poem below an otherwise-comparable one that "
        "reads cleanly all the way through, and do NOT let vivid imagery rescue a "
        "poem whose lines don't actually parse. HEAVILY penalize interchangeable "
        "lines, list-like poems, or sentences that don't actually parse.\n"
        "Then, secondary: concrete imagery that SERVES the brief (not decoration for its "
        "own sake), and natural un-forced line endings. HEAVILY penalize forced/filler "
        "endings — interjections or words tacked on as a non-sequitur just to land a "
        "letter (e.g. 'hi', 'yes', 'so', 'go', 'see', 'lab', 'dev', 'surf', 'tree'), and "
        "obscure, archaic, truncated-sounding, or reused-across-lines endings. Penalize "
        "PADDING and PROSE RUN-ON LINES hard: each line should read as a tight line of "
        "verse, not an essay sentence — a line that sprawls well past ~90 characters is a "
        "serious flaw even when its imagery is pretty. Prefer economy; every line must "
        "earn its place."
    ),
    polarity="positive",
)

# Deterministic quality-score penalties (multiplicative on a CORRECT poem's judged
# quality — see _line_length_penalty / _ending_penalty). The judge is unreliable at
# catching these (run 0ec8e2dc: run-on lines and 'hi' endings scored at the corpus
# mean), so they are charged deterministically too. All tunable.
LINE_CHAR_CAP = 90  # gold-reference poems never exceed ~90 chars/line; beyond this
# lines degenerate into prose run-ons (run 0ec8e2dc).
LEN_PENALTY = 0.5  # max quality fraction removed when every line runs ~2× the cap
W_HARD_ENDING = 0.5  # per-line quality fraction removed for a HARD filler ending
W_SOFT_ENDING = 0.2  # ditto for a SOFT (mode-collapse) ending — penalized less
PENALTY_FLOOR = 0.1  # a correct poem keeps at least this fraction of its quality

# Secondary bonuses are all quality-scaled (a weak poem can't farm them) and each
# capped at 0.15, so quality stays dominant. Diversity + conciseness apply only to
# the group's TOP occupied band, so even an all-"below" group keeps a gradient.
W_RHYME = 0.15  # rhyme-density bonus (any correct poem)
W_DIVERSITY = 0.15  # unique-ending bonus (anti mode-collapse)
# Conciseness = shorter generation + tool thrift, contributing equally (0.075 each).
# Tool-eff is graded (full at 0 calls → 0 at the cap), so the model weans off the tool.
W_CONCISE = 0.075  # shorter-generation bonus
W_TOOL_EFF = 0.075  # few-tool-calls bonus — the wean-off knob
LEN_BUDGET_BASE = 1500  # completion-char budget = BASE + PER_LINE * acrostic_len;
LEN_BUDGET_PER_LINE = 600  # at/under budget = full length-efficiency, decaying above

# ══════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════
_TRAILING_PUNCT = re.compile(r'[\s.!?,;:"\')}\]\-—…。！？，；：""' "》）】　]+$")
_MARKUP_WRAPPER_RE = re.compile(r"^[\*_`\\{}]+|[\*_`\\{}]+$")
_CJK_RE = re.compile(r"[一-鿿]")


def detect_language(target_word: str) -> str:
    return "zh" if any("一" <= ch <= "鿿" for ch in target_word) else "en"


def extract_answer(text: str) -> str | None:
    """Return the poem inside the final <answer>...</answer>, or None."""
    stripped = (text or "").rstrip()
    if not stripped.endswith("</answer>"):
        return None
    inner = stripped[: -len("</answer>")]
    idx = inner.rfind("<answer>")
    return inner[idx + len("<answer>") :].strip() if idx != -1 else None


def parse_poem_lines(text: str) -> list[str]:
    lines = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line or re.match(r"^(title|poem|verse)\s*:", line, re.IGNORECASE):
            continue
        if re.match(r"^[-=]{3,}$", line):
            continue
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            lines.append(line)
    return lines


def last_char(line: str, language: str) -> str:
    stripped = _TRAILING_PUNCT.sub("", line)
    if not stripped:
        return ""
    if language == "zh":
        return stripped[-1]
    for ch in reversed(stripped):
        if ch.isalpha():
            return ch.lower()
    return ""


def last_word(line: str) -> str:
    stripped = _TRAILING_PUNCT.sub("", line)
    words = stripped.split()
    if not words:
        return ""
    w = re.sub(r"^['\"(\[-]+|['\")\]-]+$", "", words[-1])
    return _MARKUP_WRAPPER_RE.sub("", w).lower()  # strip **bold** / _italic_ / `code`


def contains_hidden_word(poem: str, target: str, language: str) -> bool:
    if not target or not poem:
        return False
    if language == "zh":
        return target in poem
    return re.search(rf"\b{re.escape(target)}\b", poem, re.IGNORECASE) is not None


def final_poem(messages: Messages) -> str | None:
    """Poem from the <answer> block of the last non-empty assistant message."""
    text = ""
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    text = c
                    break
    else:
        text = extract_completion_text(messages)
    return extract_answer(text)


# ══════════════════════════════════════════════════════════════════════
# WORD VALIDITY
# ══════════════════════════════════════════════════════════════════════
_WEB2 = get_english_words_set(["web2"], lower=True)
# Rejects only true gibberish — deliberately LENIENT, since no freq/dict rule cleanly
# separates truncation-junk ("dis","tak") from rare real words ("origami"). The
# "natural vs forced ending" call is left to the rubric judge + partial credit.
# Length-aware validity:
#  - len >= 3: (in web2 OR CMU dict) AND minimally attested; OR common on its own.
#  - len == 2: dicts too noisy here, so require HIGH freq ("tv","iv","go" pass).
#  - len < 2: never valid.
_FREQ_THRESHOLD = 3e-6  # 3+ letters: common enough to pass without the dict
_DICT_FREQ_FLOOR = 3e-7  # 3+ letters: min attestation for a dictionary word
_SHORT_FREQ_THRESHOLD = 2e-5  # exactly 2 letters (dicts ignored — too noisy here)


def _in_dictionary(w: str) -> bool:
    return w in _WEB2 or bool(pronouncing.phones_for_word(w))


def is_valid_word(word: str) -> bool:
    w = (word or "").lower()
    n = len(w)
    if n < 2:
        return False
    f = word_frequency(w, "en")
    if n == 2:
        return f > _SHORT_FREQ_THRESHOLD
    return (_in_dictionary(w) and f > _DICT_FREQ_FLOOR) or f > _FREQ_THRESHOLD


# ══════════════════════════════════════════════════════════════════════
# HARD RULES
# ══════════════════════════════════════════════════════════════════════
# Minimum line size — kills the degenerate "acrostic spelled one word/char per line"
# hack. Legit poems clear these with wide margin.
MIN_ZH_LINE_CHARS = 4
MIN_EN_LINE_WORDS = 3

# Blacklisted line-endings: words almost always bent in just to land a letter. They
# pass is_valid_word (the non-word check misses them), so they are caught here instead
# — referenced by BOTH the feedback tool (which flags them) and quality scoring (which
# scales the judged quality down, _ending_penalty). Derived from sampling real rollouts
# and Sonnet-rating forced rate (run 0ec8e2dc). Two tiers:
#  HARD — interjections / bare verbs / dangling adverbs; weak endings in ANY context.
#  SOFT — content nouns the policy mode-collapsed to as the lazy default per hard letter
#         (l→hill, b→lab, v→dev, f→surf, e→tree). Fine when on-theme, so penalized less;
#         the length + diversity terms do the rest of the anti-collapse work.
_HARD_FORCED_ENDINGS = frozenset(
    {
        "hi",
        "ah",
        "oh",
        "eh",
        "ho",
        "hm",
        "hmm",
        "um",
        "uh",
        "er",
        "ahh",
        "ohh",
        "ooh",
        "aha",
        "hey",
        "yay",
        "lo",
        "ye",
        "ya",
        "huh",
        "aw",
        "ow",
        "hah",
        "ha",
        "oi",
        "yo",
        "yes",
        "far",
        "long",
        "see",
        "go",
    }
)
_SOFT_FORCED_ENDINGS = frozenset({"hill", "lab", "dev", "surf", "tree"})
_FORCED_ENDINGS = _HARD_FORCED_ENDINGS | _SOFT_FORCED_ENDINGS  # any blacklisted ending


def _partial_reward(misses: int, allowed: int) -> float:
    """Map a partial poem's miss count to a small graded reward: 1 miss → PARTIAL_HI,
    the miss limit → PARTIAL_LO, linear between. (allowed == 1 → always PARTIAL_HI.)"""
    if allowed <= 1:
        return PARTIAL_HI
    frac = (misses - 1) / (allowed - 1)  # 0 at 1 miss, 1 at the limit
    return PARTIAL_HI - frac * (PARTIAL_HI - PARTIAL_LO)


def check_hard_rules(poem: str | None, target: str) -> dict:
    """Graded gate. `correct` == True iff a perfect telestich. Otherwise, if the
    answer is present, the line count is right, it doesn't cheat, and MORE than
    MIN_CORRECT_FRAC of lines are correct (a correct line = a valid word ending in
    the required letter), the poem is `partial` and earns a small graded
    `partial_score` instead of being zeroed — this keeps a learning gradient.
    Cheating, a missing answer, a wrong line count, or ≤25% correct → gated (0)."""
    target = (target or "").strip()
    language = detect_language(target)
    lines = parse_poem_lines(poem or "")
    chars = list(target.lower()) if language == "en" else list(target)
    n = len(chars)
    cheated = contains_hidden_word(poem or "", target, language)

    def result(
        correct: bool, partial: bool, reason: str, misses: int = 0, score: float = 0.0
    ) -> dict:
        return {
            "correct": correct,
            "partial": partial,
            "partial_score": score,
            "misses": misses,
            "cheated": cheated,
            "language": language,
            "lines": lines,
            "reason": reason,
        }

    # Hard preconditions for ANY credit: a target, an answer poem, no cheating,
    # and the right number of lines.
    if n == 0:
        return result(False, False, "empty target word")
    if not lines:
        return result(False, False, "no <answer> poem found in completion")
    if cheated:
        return result(False, False, "hidden word written in the poem body")
    if len(lines) != n:
        return result(False, False, f"line count {len(lines)} != target length {n}")

    # Count "misses": lines whose ending isn't a valid word in the required letter
    # (or, for zh, a vertically-spelled one-char line / wrong char).
    miss_idx = []
    for i in range(n):
        if language == "zh":
            cjk = len(_CJK_RE.findall(lines[i]))
            ok = cjk >= MIN_ZH_LINE_CHARS and last_char(lines[i], language) == chars[i]
        else:
            ok = (
                last_char(lines[i], language) == chars[i]
                and is_valid_word(last_word(lines[i]))
                and len(lines[i].split()) >= MIN_EN_LINE_WORDS
            )
        if not ok:
            miss_idx.append(i)

    misses = len(miss_idx)
    if misses == 0:
        return result(True, False, "", 0)
    good = n - misses
    detail = "; ".join(
        f"line {i + 1} ('{last_word(lines[i]) or '∅'}'→needs '{chars[i]}')" for i in miss_idx
    )
    if good / n > MIN_CORRECT_FRAC:  # more than 25% of lines correct → partial credit
        max_partial = math.ceil((1 - MIN_CORRECT_FRAC) * n) - 1  # most misses still above the bar
        return result(
            False,
            True,
            f"partial ({good}/{n} endings correct): {detail}",
            misses,
            _partial_reward(misses, max_partial),
        )
    return result(
        False,
        False,
        f"{misses}/{n} endings off (only {good}/{n} correct, ≤25%): {detail}",
        misses,
    )


def _gate_category(reason: str) -> str:
    """Collapse a specific gate reason (e.g. "line 2 ends 'l', expected 'u'") into
    a short bucket so the group log shows counts-by-kind, not one entry per line."""
    r = reason.lower()
    if "no <answer>" in r or "no poem" in r:
        return "no-answer"
    if "empty target" in r:
        return "bad-target"
    if "line count" in r:
        return "wrong-line-count"
    if "hidden word" in r:
        return "cheated"
    if "endings off" in r:
        return "too-many-misses"
    return "other"


# ══════════════════════════════════════════════════════════════════════
# RHYME  (English rhyme density)
# ══════════════════════════════════════════════════════════════════════
def _rhyme_analysis(
    lines: list[str],
) -> tuple[float, list[list[int]], list[str], list[int]]:
    """(density, clusters, endings, idx). density = largest rhyme cluster size /
    #lines-with-pronunciations. clusters = connected groups of mutually-rhyming
    line indices; endings = ending word per line; idx = lines with pronunciations."""
    endings = [_MARKUP_WRAPPER_RE.sub("", last_word(line)) for line in lines]
    parts, idx = {}, []
    for i, w in enumerate(endings):
        phones = pronouncing.phones_for_word(w) if w else []
        if phones:
            parts[i] = [pronouncing.rhyming_part(p) for p in phones]
            idx.append(i)
    if len(idx) < 2:
        return 0.0, [], endings, idx
    adj = {i: set() for i in idx}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if endings[i] != endings[j] and any(x == y for x in parts[i] for y in parts[j]):
                adj[i].add(j)
                adj[j].add(i)
    visited, clusters = set(), []
    for i in idx:
        if i in visited:
            continue
        stack, comp = [i], []
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.append(x)
            stack.extend(adj[x])
        clusters.append(sorted(comp))
    largest = max((len(c) for c in clusters if len(c) >= 2), default=0)
    return largest / len(idx), clusters, endings, idx


def rhyme_score(lines: list[str]) -> float:
    return _rhyme_analysis(lines)[0]


def rhyme_detail(lines: list[str]) -> str:
    """Human-readable explanation of the rhyme score — which lines rhyme."""
    density, clusters, endings, idx = _rhyme_analysis(lines)
    rhyming = [c for c in clusters if len(c) >= 2]
    groups = (
        "; ".join("+".join(f"L{i + 1}:{endings[i]}" for i in c) for c in rhyming)
        or "no rhyming pairs"
    )
    unmatched = [f"L{i + 1}:{endings[i]}" for i in idx if not any(i in c for c in rhyming)]
    largest = max((len(c) for c in rhyming), default=0)
    tail = f"; unmatched {unmatched}" if unmatched else ""
    return f"largest rhyme {largest}/{len(idx)} → {density:.2f} | {groups}{tail}"


# ══════════════════════════════════════════════════════════════════════
# QUALITY-SCORE PENALTIES  (deterministic, multiplicative on a CORRECT poem's quality)
# ══════════════════════════════════════════════════════════════════════
def overlong_lines(lines: list[str]) -> list[int]:
    """Indices of lines past LINE_CHAR_CAP — the prose-run-on degeneracy."""
    return [i for i, ln in enumerate(lines) if len(ln) > LINE_CHAR_CAP]


def _line_length_penalty(lines: list[str]) -> float:
    """Quality multiplier in [PENALTY_FLOOR, 1]: scales down by the mean normalized
    overage of lines past LINE_CHAR_CAP. All lines within the cap → 1.0 (no penalty)."""
    if not lines:
        return 1.0
    overage = sum(max(0.0, (len(ln) - LINE_CHAR_CAP) / LINE_CHAR_CAP) for ln in lines)
    return max(PENALTY_FLOOR, 1.0 - LEN_PENALTY * overage / len(lines))


def blacklisted_endings(lines: list[str]) -> list[tuple[int, str, str]]:
    """[(line_idx, ending_word, 'hard'|'soft')] for lines ending on a blacklisted word."""
    out = []
    for i, ln in enumerate(lines):
        w = last_word(ln)
        if w in _HARD_FORCED_ENDINGS:
            out.append((i, w, "hard"))
        elif w in _SOFT_FORCED_ENDINGS:
            out.append((i, w, "soft"))
    return out


def _ending_penalty(lines: list[str]) -> float:
    """Quality multiplier in [PENALTY_FLOOR, 1]: scales down for blacklisted endings —
    HARD fillers cost W_HARD_ENDING per line, SOFT (mode-collapse) cost W_SOFT_ENDING."""
    if not lines:
        return 1.0
    charged = sum(
        W_HARD_ENDING if tier == "hard" else W_SOFT_ENDING
        for _, _, tier in blacklisted_endings(lines)
    )
    return max(PENALTY_FLOOR, 1.0 - charged / len(lines))


# ══════════════════════════════════════════════════════════════════════
# DIVERSITY
# ══════════════════════════════════════════════════════════════════════
def ending_reuse_detail(
    poems: list[str],
) -> tuple[list[float], list[list[tuple[str, int]]]]:
    """(scores, details). Per-poem reuse score in [0,1]: mean over the poem's
    ending words of the fraction of *other* poems that also use that word. 0 =
    all endings unique to this poem; 1 = every ending shared by every sibling.
    details[i] = [(ending_word, n_other_poems_sharing_it)] for logging."""
    n = len(poems)
    endings = []
    for p in poems:
        zh = detect_language(p) == "zh"
        toks = []
        for line in parse_poem_lines(p):
            tok = last_char(line, "zh") if zh else last_word(line)
            if tok:
                toks.append(tok)
        endings.append(toks)
    doc_freq: Counter = Counter()
    for ews in endings:
        for w in set(ews):
            doc_freq[w] += 1
    scores, details = [], []
    for ews in endings:
        details.append([(w, doc_freq[w] - 1) for w in ews])
        if not ews or n < 2:
            scores.append(0.0)
        else:
            scores.append(sum((doc_freq[w] - 1) / (n - 1) for w in ews) / len(ews))
    return scores, details


def ending_reuse_scores(poems: list[str]) -> list[float]:
    return ending_reuse_detail(poems)[0]


def duplicate_divisors(poems: list[str], threshold: float = 0.85) -> list[float]:
    """Cluster near-identical *whole poems*; divisor = cluster size."""
    cluster_of, reps = [], []
    for p in poems:
        for cid, rep in enumerate(reps):
            if SequenceMatcher(None, p, rep).ratio() > threshold:
                cluster_of.append(cid)
                break
        else:
            cluster_of.append(len(reps))
            reps.append(p)
    counts = Counter(cluster_of)
    return [float(counts[c]) for c in cluster_of]


# ══════════════════════════════════════════════════════════════════════
# PER-ROLLOUT RECORD
# ══════════════════════════════════════════════════════════════════════
@dataclass
class _Rollout:
    """One rollout's parsed state + scoring, carried through the pipeline so the
    final assembly and logging read straight off it (see compute_group_rewards)."""

    idx: int
    rollout_id: str
    completion_len: int  # chars of the whole completion (reasoning + answer)
    poem: str  # parsed <answer> poem ("" if none)
    poem_len: int
    lines: list[str]
    language: str
    n_tool_calls: int
    correct: bool  # perfect telestich → quality-judged
    reason: str  # failure description ("" when correct)
    partial: bool = False  # answer present, line count right, ≤ limit misses → graded
    partial_score: float = 0.0  # small graded reward for a partial poem
    misses: int = 0  # count of missed endings (wrong letter / non-word / short)
    quality: float = 0.0  # judged band score after the deterministic craft penalties
    len_pen: float = 1.0  # line-length penalty multiplier applied to quality
    end_pen: float = 1.0  # blacklisted-ending penalty multiplier applied to quality
    q: float = 0.0  # quality after the duplicate divisor (used everywhere)
    bucket: str = "gated"  # gated | below | mid | above
    reuse: float = 0.0  # ending-word reuse vs siblings (0=unique, 1=all shared)
    ending_detail: str = ""  # per-ending share counts, for the breakdown log
    len_eff: float = 0.0  # length efficiency in [0,1] (1 = within budget; logged)
    in_top_band: bool = False  # in the group's top occupied bucket → earns secondary bonuses
    components: dict = field(default_factory=dict)


def _bucket(q: float) -> str:
    """Map a quality score to its band bucket."""
    if q < ACCEPTABLE_EDGE:
        return "below"
    if q < GREAT_EDGE:
        return "mid"
    return "above"


def _map_correct_score(s: float) -> float:
    """Remap a CORRECT poem's raw rubric band score onto the reward ladder. With
    acceptable/great anchors scored at ACCEPTABLE_EDGE/GREAT_EDGE, the rubric returns
    normal..great as [0.4, 0.7] and above-great as (0.7, 1.0]; we only floor the
    below-normal band [0, 0.4) up to [MIN_CORRECT, 0.4) so the worst correct poem
    still scores MIN_CORRECT, not ~0."""
    if s < ACCEPTABLE_EDGE:
        return MIN_CORRECT + s * (ACCEPTABLE_EDGE - MIN_CORRECT) / ACCEPTABLE_EDGE
    return s


# ── ELO judge helpers (pure; see ELO_JUDGE) ──────────────────────────────
# Logistic map BT-rating → band score, calibrated so the acceptable anchor rating (0.0)
# → ACCEPTABLE_EDGE and the great anchor (ELO_GREAT_RATING) → GREAT_EDGE. Asymptotes to
# (0,1), so below-baseline poems keep a gradient instead of flooring to 0 (a hard band
# clamp would collapse them — the run sits almost entirely below baseline).
_ELO_A = math.log(1.0 / ACCEPTABLE_EDGE - 1.0)
_ELO_B = math.log(1.0 / GREAT_EDGE - 1.0)
_ELO_K = (_ELO_A - _ELO_B) / ELO_GREAT_RATING
_ELO_R0 = _ELO_A / _ELO_K


def _rating_to_band(r: float) -> float:
    """BT rating → band score in (0,1), calibrated at the two anchor ratings."""
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, _ELO_K * (r - _ELO_R0)))))


def _make_slices(n: int, size: int = ELO_SLICE, stride: int = ELO_STRIDE) -> list[list[int]]:
    """Overlapping poem-index slices over 0..n-1: wrap-around windows of `size` at `stride`,
    plus an extra slice for any index the stride left covered <2× — so every poem appears in
    ≥2 slices for cross-comparison. n ≤ size → one slice of all poems."""
    if n <= size:
        return [list(range(n))]
    slices = [[(s + j) % n for j in range(size)] for s in range(0, n, stride)]
    cnt = Counter(i for sl in slices for i in sl)
    under = [i for i in range(n) if cnt[i] < 2]
    while under:
        take = under[:size]
        under = under[size:]
        if len(take) < size:
            take += [i for i in range(n) if i not in take][: size - len(take)]
        slices.append(take)
    return slices


def _bt_fit(
    pairs: list[tuple[int, int]],
    fixed: dict[int, float],
    n_players: int,
    iters: int = ELO_BT_ITERS,
    lr: float = ELO_BT_LR,
) -> list[float]:
    """Bradley-Terry ratings by gradient ascent over (winner, loser) pairs pooled across
    slices; `fixed` players (the anchors) stay pinned at their given ratings."""
    ratings = [0.0] * n_players
    for pid, val in fixed.items():
        ratings[pid] = val
    norm = max(1.0, len(pairs) / max(1, n_players))
    for _ in range(iters):
        grad = [0.0] * n_players
        for win, lose in pairs:
            d = max(-30.0, min(30.0, ratings[win] - ratings[lose]))
            pe = 1.0 / (1.0 + math.exp(-d))
            grad[win] += 1.0 - pe
            grad[lose] -= 1.0 - pe
        for i in range(n_players):
            if i not in fixed:
                ratings[i] += lr * grad[i] / norm
    return ratings


def _first_ref(refs: Any) -> str | None:
    """First non-empty reference poem from a refs list, or None."""
    items = [r for r in (refs or []) if r and str(r).strip()]
    return items[0] if items else None


_TOOL_CALL_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def _count_tool_calls(messages: Messages) -> int:
    """Count structured and legacy text-form tool-call attempts.

    BaseEnv records OpenAI tool calls on assistant messages. The text fallback
    keeps older captured transcripts scoreable and includes rejected calls that
    were emitted directly in completion text.
    """

    count = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
            count += len(tool_calls)
        content = message.get("content")
        if isinstance(content, str):
            count += len(_TOOL_CALL_RE.findall(content))
    return count


# ══════════════════════════════════════════════════════════════════════
# FEEDBACK TOOL
# ══════════════════════════════════════════════════════════════════════
def _normalize_word(word: str) -> str:
    """Normalize the hidden word the model passes to the feedback tool: take the
    first token, strip quotes/punctuation, lowercase. Returns "" if nothing usable
    (English letters or CJK) remains — the caller then rejects the malformed call."""
    w = (word or "").strip().split()[0] if (word or "").strip() else ""
    w = w.strip("\"'“”‘’.,;:!?()[]{}").lower()
    return w if re.fullmatch(r"[a-z]+|[一-鿿]+", w) else ""


def _draft_poem_text(raw: str) -> str:
    """Pull the poem out of a draft the model passes to the feedback tool. It may
    arrive wrapped in <answer>…</answer> (possibly with reasoning around it) or as
    bare lines — normalize both so tag lines aren't counted as poem lines."""
    m = re.search(r"<answer>(.*?)</answer>", raw or "", re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return re.sub(r"</?answer\s*>", "", raw or "", flags=re.IGNORECASE).strip()


def _programmatic_feedback(poem: str, target: str) -> str | None:
    """STACKED, LINE-BY-LINE deterministic feedback on a draft: a whole-poem line-count
    note (if any), then ONE entry per offending line in line order, with all of that
    line's problems joined together — wrong last letter, filler/blacklisted ending,
    non-word / too-short ending, over-long (run-on) line, and the hidden word leaked
    into the line — so the model can fix everything in a single revision. Returns None
    for a clean draft. Reward gating lives in check_hard_rules; this is FORMATIVE only."""
    target = (target or "").strip()
    language = detect_language(target)
    chars = list(target.lower()) if language == "en" else list(target)
    n = len(chars)
    lines = parse_poem_lines(poem or "")
    # blocking — can't give per-line feedback without a target word or any poem
    if n == 0:
        return "No target word in context — cannot give feedback."
    if not lines:
        return "No poem found. Put the poem inside <answer>...</answer>, one line per letter."

    a = len(lines)
    en = language == "en"
    notes: list[str] = []  # whole-poem notes (line count)
    line_msgs: list[str] = []  # one entry per offending line, in line order

    # whole-poem: line count
    if a != n:
        verb = f"add {n - a} line(s)" if a < n else f"remove {a - n} line(s)"
        notes.append(
            f"Line count: you have {a} line(s) but '{target}' needs {n} (one per letter) — {verb}."
        )

    # per line: stack ALL of that line's issues into one entry, in line order. The
    # wrong-letter problem supersedes the ending-word checks for the same line (no point
    # telling it to swap a filler when the letter is wrong anyway); over-long and the
    # leaked hidden word are independent and always added.
    for i in range(a):
        req = chars[i] if i < n else None  # required last letter (None for extra lines)
        got = last_char(lines[i], language)
        lw = last_word(lines[i])
        issues: list[str] = []
        if req is not None and got != req:
            issues.append(f"must end in '{req}', but \"{lw or lines[i]}\" ends in '{got or '∅'}'")
        elif en:
            need = f" ending in '{req}'" if req is not None else ""
            if lw in _FORCED_ENDINGS:
                issues.append(
                    f'"{lw}" is a forced filler word — use another, meaningful word{need}'
                )
            elif not is_valid_word(lw):
                issues.append(f'ends on non-word "{lw}"')
            elif len(lines[i].split()) < MIN_EN_LINE_WORDS:
                issues.append(
                    f"only {len(lines[i].split())} word(s); a poem line needs "
                    f"at least {MIN_EN_LINE_WORDS}"
                )
        if len(lines[i]) > LINE_CHAR_CAP:
            issues.append(
                f"runs long at {len(lines[i])} chars (over {LINE_CHAR_CAP}) — "
                f"cut to a tight line of verse"
            )
        if contains_hidden_word(lines[i], target, language):
            issues.append(
                f'contains the hidden word "{target}" — keep it hidden in the last letters only'
            )
        if issues:
            line_msgs.append(f"   - line {i + 1}: " + "; ".join(issues))

    if not notes and not line_msgs:
        return None  # clean draft — nothing to fix
    return "Revise your draft — fix all of these:\n" + "\n".join(notes + line_msgs)


# ══════════════════════════════════════════════════════════════════════
# ENV CLASS
# ══════════════════════════════════════════════════════════════════════
class TelestichEnv(BaseEnv):
    system_prompt = """\
A telestich is a poem whose lines, read by their LAST letters top to bottom, \
spell a hidden word the user gives you. Use the word the user names as a single \
lowercase word, ignoring any quotes or capitalization.

Example request:
"a poem about stress, last letters spell stress".

Example output:

Letters: s, t, r, e, s, s → 6 lines, one per letter. 's' repeats on lines 1, 5, \
6 — rhyme those three on one sound for a bonus; pick the hardest letters first.
Draft endings: distress, taut, roar, sea, caress, excess (the s-lines \
distress/caress/excess rhyme on "-ess"); never put "stress" in the poem itself.
Quick checks pass — 6 lines, each ends in a real word — but rather than eyeball \
every last letter, I send the draft to the `feedback` tool (passing the poem AND \
word="stress" so it can check the endings). It replies: "line 4 ends in 'a', needs \
'e'." I swap that ending to "peace" and finalize:

<answer>
The piling deadlines crush my chest, distress
And every muscle in my neck pulls taut
I watch the storm clouds gather, hear them roar
And wish that I could rest and be at peace
I crave a quiet calm, a soft caress
To free my racing mind from this excess
</answer>

Once you can reliably get every ending right on the first try, skip the tool and \
one-shot it for the small efficiency bonus.

How you're scored — a chain, each step gating the next:
1. CORRECTNESS is the gate: the last letters must spell the word, every line must \
end in a real word, and the line count must match. Miss this and you get nothing \
(the hidden word written in the poem, a wrong line count, or no answer all score \
zero) — except a near-complete attempt (right line count, only an ending or two \
off) earns small partial credit.
2. QUALITY scores correct poems and is the bulk of the reward: craft — vivid \
concrete imagery, COHERENCE (the lines must connect into one real poem, not a list), \
fit to the requested theme/voice — sets how high you go. Two things are penalized \
hard here: PROSE RUN-ON LINES (keep each line a tight line of verse, well under ~90 \
characters — don't write essay-length lines) and FILLER / FORCED endings (a word \
tacked on just to land the letter, e.g. "hi", "yes", "go", "tree" — choose an ending \
that genuinely fits the line's sense).
3. QUALITY then unlocks SECONDARY bonuses (smaller, each up to ~15%): rhyme, \
variety of ending words, and conciseness — which rewards a shorter, un-padded \
answer AND fewer feedback calls EQUALLY, so don't ramble and wean off the tool \
once you can nail the poem on your own.

Tips:
- Do the quick checks you can (right number of lines, every line ends in a real \
word), then SEND your draft to the `feedback` tool — pass both your poem and the \
hidden `word` it should spell — to confirm the endings and flag any over-long lines \
or filler endings; don't try to verify every last letter in your head, that's \
error-prone. The tool only checks CORRECTNESS (right endings, line count, no run-ons \
or filler words) — it does NOT tell you whether the poem is any good, so a "passed" \
result is not a quality signal: judge coherence and voice yourself before finalizing. \
Revise from its feedback, then finalize. Use it sparingly (≤3 calls — usually one verify pass is \
enough); once you can reliably nail it first try, skip it for the efficiency bonus.
- Rhyme only works between lines that share an ending letter, so rhyme the lines \
whose letters REPEAT in the word; match vowel SOUNDS, not spelling, and don't force it.
- Keep your thinking SHORT — a few notes, then the poem. Do NOT grind through \
candidate words letter-by-letter in your head (e.g. listing "mule? -> e, pule? -> \
e, ..."): it bloats the response, gets truncated, and is exactly what the \
`feedback` tool does for you reliably. Draft, check endings with the tool (or your \
own eye once you're good), revise, answer. Never write the hidden word in the \
poem; write in English; output only the poem in <answer></answer>, plain text, \
then stop."""

    def __init__(
        self,
        *,
        judge_base_url: str,
        judge_auth: ModelAuth = InjectedAuth("judge"),
        judge_timeout: float = 90.0,
        max_tool_calls: int = 3,
        max_turns: int | None = None,
        system_prompt: str | None = None,
        train_dataset_path: str = "train.jsonl",
        eval_dataset_path: str = "eval.jsonl",
    ) -> None:
        super().__init__(
            max_turns=max_tool_calls + 1 if max_turns is None else max_turns,
            max_tool_calls=max_tool_calls,
        )
        self._system_prompt = self.system_prompt if system_prompt is None else system_prompt
        self._judge = Judge(
            model=JUDGE_MODEL,
            base_url=judge_base_url,
            auth=judge_auth,
            timeout=judge_timeout,
        )
        self._max_tool_calls = max_tool_calls
        self._tool_calls: dict[str, int] = {}
        # Canonical split filenames, resolved against the runtime dataset
        # base_dir (the mirrored upload prefix at training time).
        self._dataset_paths = {
            "train": train_dataset_path,
            "eval": eval_dataset_path,
        }

    async def create_dataset(
        self,
        split: DatasetSplit,
        base_dir: Path,
        *,
        max_examples: int | None = None,
    ) -> JsonlDataset[JsonRow]:
        return JsonlDataset(
            resolve_dataset_path(base_dir, self._dataset_paths[split]),
            row_to_example=self._example_from_row,
            max_examples=max_examples,
        )

    def _example_from_row(self, row: JsonRow) -> Example[JsonRow]:
        prompt = row.get("prompt", "")
        if not isinstance(prompt, str):
            raise TypeError("TelestichEnv dataset 'prompt' must be a string")
        messages: Messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: JsonRow = {
            "prompt_messages": messages,
            "prompt": prompt,
            "ground_truth": row.get("ground_truth", ""),
            "acceptable_refs": row.get("acceptable_refs", []),
            "great_refs": row.get("great_refs", []),
        }
        return Example(
            id=canonical_example_id(payload),
            payload=payload,
        )

    async def list_tools(self) -> list[Tool]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "feedback",
                    "description": (
                        "Feedback on a DRAFT telestich (up to 3 calls): pass your draft ('poem') "
                        "AND the hidden word it should spell ('word'); it checks every line ending "
                        "against that word — so you don't have to verify each last letter yourself "
                        "— and flags over-long (prose run-on) lines and filler endings, both of "
                        "which lower your score. Recommended flow: write a full draft, send it "
                        "here "
                        "to check the endings and tighten, then put your revised best in <answer>. "
                        "Use it sparingly (often one pass is enough); a correct poem written with "
                        "no "
                        "calls earns a small efficiency bonus, so wean off once you can reliably "
                        "nail "
                        "it on the first try."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "poem": {
                                "type": "string",
                                "description": "your full draft poem, one line per letter",
                            },
                            "word": {
                                "type": "string",
                                "description": "the hidden word your poem should spell "
                                "(the exact target word from the request)",
                            },
                        },
                        "required": ["poem", "word"],
                    },
                    "strict": False,
                },
            }
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name != "feedback":
            return f"Error: unknown tool '{tool_name}'."
        used = self._tool_calls.get(rollout_id, 0)
        if used >= self._max_tool_calls:
            return "No feedback calls remaining — submit your best poem in <answer>...</answer>."
        raw = (tool_args.get("poem") or "").strip()
        # The model supplies the hidden word to validate against (no extraction / no
        # ground_truth dependency — works identically in training and the playground).
        target = _normalize_word(tool_args.get("word") or "")
        if not raw or not target:  # malformed call — don't spend a budgeted call on it
            return (
                "Pass BOTH your full draft poem ('poem') and the hidden word it "
                "should spell ('word') to get feedback."
            )
        self._tool_calls[rollout_id] = used + 1
        # If this used the last budgeted call, every feedback response below tells the
        # model to stop and answer (mirrors the out-of-calls message above).
        last_note = (
            "\n\nThat was your last feedback call — put your best poem in "
            "<answer>...</answer> now. (Both fewer calls AND a shorter, "
            "un-padded answer score higher on conciseness.)"
            if self._tool_calls[rollout_id] >= self._max_tool_calls
            else ""
        )
        poem = _draft_poem_text(raw)
        prog = _programmatic_feedback(poem, target)
        if prog is not None:
            return prog + last_note  # structure/quality issues — fix these first
        # Clean telestich: right structure, real non-filler endings, no run-on lines.
        # Deterministic checks are all the feedback we give (no LLM critic).
        good = (
            "Correctness check passed: the endings spell the word, every line ends "
            "in a real non-filler word, and no line runs long. (This checks the "
            "mechanics only, not how good the poem is.)"
        )
        if not last_note:
            good += (
                " Put it in <answer> now rather than calling again — both fewer "
                "feedback calls and a shorter, un-padded answer score higher."
            )
        return good + last_note

    @asynccontextmanager
    async def rollout_context(
        self,
        rollout_id: str,
        example: Example[JsonRow],
    ) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            self._tool_calls.pop(rollout_id, None)

    async def compute_reward(
        self,
        rollout: BaseRollout,
    ) -> None:
        """Defer scoring until all sibling poems are available."""

        del rollout
        return None

    async def _quality(
        self,
        prompt: str,
        poems: list[str],
        anchors: list[RankingAnchor],
    ) -> list[float]:
        """Quality via the MULTI-ANCHOR rubric judge, returned in `poems` order. With
        ELO_JUDGE (default) each call ranks a small SLICE of poems against both anchors and
        a Bradley-Terry fit (anchors pinned as fixed-rating reference players) yields one
        calibrated score per poem — see _quality_elo. ELO_JUDGE=False restores the disjoint
        JUDGE_BATCH ranker: each call ranks ≤JUDGE_BATCH poems against the blind anchors
        (run 0ec8e2dc: full-group 15/36 false-better → batch-of-3 0/36), scores in `poems`
        order."""
        if not poems:
            return []
        if ELO_JUDGE and anchors:
            return await self._quality_elo(prompt, poems, anchors)

        async def _score_batch(batch: list[str]) -> list[float]:
            res = await evaluate_rubric_ranking(
                rubric=QUALITY_RUBRIC,
                question=prompt,
                responses=batch,
                judge=self._judge,
                anchors=anchors,
            )
            return list(res.scores)

        batches = [poems[i : i + JUDGE_BATCH] for i in range(0, len(poems), JUDGE_BATCH)]
        results = await asyncio.gather(*(_score_batch(b) for b in batches))
        return [s for batch_scores in results for s in batch_scores]

    async def _quality_elo(
        self,
        prompt: str,
        poems: list[str],
        anchors: list[RankingAnchor],
    ) -> list[float]:
        """Slice-and-Bradley-Terry quality scorer (see ELO_JUDGE / _quality). Each judge
        call ranks an ELO_SLICE-sized slice of poems against BOTH anchors (in every call);
        poems sit in overlapping slices (_make_slices), so the pooled pairwise outcomes feed
        a BT fit with the anchors pinned at fixed ratings (_bt_fit). Ratings map back to band
        scores (_rating_to_band). Slices run concurrently and a judge failure fails the
        group score. Returns band scores in `poems` order."""
        n = len(poems)
        n_anchors = len(anchors)
        anchor_ids = list(range(n, n + n_anchors))
        # anchor band edge → fixed BT rating (acceptable edge → 0.0, great edge → GREAT_RATING)
        span = GREAT_EDGE - ACCEPTABLE_EDGE
        fixed = {
            n + a: (anchor.score - ACCEPTABLE_EDGE) / span * ELO_GREAT_RATING
            for a, anchor in enumerate(anchors)
        }

        async def _slice_pairs(sl: list[int]) -> list[tuple[int, int]]:
            res = await evaluate_rubric_ranking(
                rubric=QUALITY_RUBRIC,
                question=prompt,
                responses=[poems[i] for i in sl] + [anchor.response for anchor in anchors],
                judge=self._judge,
            )
            sc = res.scores
            players = list(sl) + anchor_ids  # local item index → global player id
            return [
                (players[a], players[b])
                for a in range(len(players))
                for b in range(len(players))
                if a != b and sc[a] > sc[b] + 1e-9
            ]

        slices = _make_slices(n)
        results = await asyncio.gather(*(_slice_pairs(sl) for sl in slices))
        pairs = [pr for sub in results for pr in sub]
        ratings = _bt_fit(pairs, fixed, n + n_anchors)
        return [_rating_to_band(ratings[i]) for i in range(n)]

    async def _score_quality(
        self,
        prompt: str,
        correct: list[_Rollout],
        acceptable: str | None,
        great: str | None,
    ) -> None:
        """Rank the CORRECT poems against the acceptable + great anchors; write each
        record's quality (remapped onto the reward ladder), dup-adjusted `q`, and
        `bucket`. Partial poems are handled separately (graded partial_score), not here."""
        anchors: list[RankingAnchor] = []
        if acceptable:
            anchors.append(RankingAnchor(acceptable, ACCEPTABLE_EDGE, "baseline"))
        if great:
            anchors.append(RankingAnchor(great, GREAT_EDGE, "great"))
        scores = await self._quality(prompt, [r.poem for r in correct], anchors)
        dup = duplicate_divisors([r.poem for r in correct])
        for local, r in enumerate(correct):
            ladder = _map_correct_score(scores[local])  # judged band → reward ladder
            # deterministic craft penalties the judge under-charges (run-on lines, filler
            # endings) scale the quality down multiplicatively before bucketing.
            r.len_pen = _line_length_penalty(r.lines)
            r.end_pen = _ending_penalty(r.lines)
            r.quality = ladder * r.len_pen * r.end_pen
            r.q = r.quality / dup[local]  # near-duplicate whole poems shared down
            r.bucket = _bucket(r.q)

    def _tool_eff(self, r: _Rollout) -> float:
        """Graded few-tool-calls bonus: full W_TOOL_EFF·q at 0 calls, scaling linearly
        to 0 at the call cap (so fewer calls → higher conciseness, and over-emitting
        rejected calls floors at 0). Quality-scaled like the other secondaries."""
        factor = max(0.0, 1.0 - r.n_tool_calls / self._max_tool_calls)
        return W_TOOL_EFF * r.q * factor

    def _apply_secondary(self, rolls: list[_Rollout], correct: list[_Rollout], budget: int) -> None:
        """Compute every rollout's reward components onto its record. Gated poems →
        all-zero. Partial poems → graded partial_score quality, no bonuses. Correct poems:
        rhyme (any), plus diversity and conciseness (folds in the tool-efficiency
        bonus) applied only to the group's TOP occupied band (anti-collapse)."""
        if correct:
            reuse, reuse_detail = ending_reuse_detail([r.poem for r in correct])
            for local, r in enumerate(correct):
                r.reuse = reuse[local]
                r.ending_detail = ", ".join(
                    f"{w}(shared×{c})" if c else f"{w}(unique)" for w, c in reuse_detail[local]
                )

        # the group's top occupied bucket among correct poems — secondary bonuses
        # apply only here, so even an all-"below" group still gets a gradient.
        top_band = next(
            (b for b in ("above", "mid", "below") if any(r.bucket == b for r in correct)),
            None,
        )

        for r in rolls:
            # length efficiency in [0,1]: 1 = answered within budget, decaying above
            r.len_eff = math.exp(-max(0.0, r.completion_len / budget - 1.0))

            if not r.correct and not r.partial:
                r.components = {
                    "quality": 0.0,
                    "rhyme": 0.0,
                    "diversity": 0.0,
                    "conciseness": 0.0,
                }
                continue
            if r.partial:  # small graded reward (r.q == partial_score); no judging/bonuses
                r.components = {
                    "quality": round(r.q, 4),
                    "rhyme": 0.0,
                    "diversity": 0.0,
                    "conciseness": 0.0,
                }
                continue
            rhyme_bonus = W_RHYME * r.q * rhyme_score(r.lines)
            # rhyme applies to any correct poem; diversity + conciseness go to the
            # group's top band only. Conciseness = shorter generation + a HEAVY
            # tool-efficiency (no-tool) bonus, so the model weans off the tool.
            r.in_top_band = r.bucket == top_band
            if r.in_top_band:
                div_bonus = W_DIVERSITY * r.q * (1.0 - r.reuse)
                concise = W_CONCISE * r.q * r.len_eff + self._tool_eff(r)
            else:
                div_bonus = concise = 0.0
            r.components = {
                "quality": round(r.q, 4),
                "rhyme": round(rhyme_bonus, 4),
                "diversity": round(div_bonus, 4),
                "conciseness": round(concise, 4),
            }

    def _fmt_rollout(self, r: _Rollout, budget: int) -> str:
        """One path-revealing log line per rollout."""
        total = sum(r.components.values())
        meta = (
            f"completion_len={r.completion_len}/{budget} poem_len={r.poem_len} "
            f"lines={len(r.lines)} tools={r.n_tool_calls}"
        )
        if not r.correct and not r.partial:
            return f"[TelestichEnv] reward={total:+.3f} GATED ({r.reason}) | {meta}"
        tag = r.bucket.upper() + (f" PARTIAL({r.misses}miss)" if r.partial else "")
        pen = (
            ""
            if (r.len_pen == 1.0 and r.end_pen == 1.0)
            else f" pen[len×{r.len_pen:.2f} end×{r.end_pen:.2f}]"
        )
        return (
            f"[TelestichEnv] reward={total:+.3f} {tag} "
            f"q={r.quality:.3f}->{r.q:.3f}{pen} reuse={r.reuse:.2f} "
            f"len_eff={r.len_eff:.2f} | {meta} | {r.components}"
        )

    def _explain_rollout(self, r: _Rollout, budget: int) -> str:
        """Multi-line WHY-breakdown for a poem that passed correctness — spells
        out how each component reached its value (which lines rhyme, which
        endings are shared, why conciseness was charged)."""
        c = r.components
        if r.in_top_band:
            div_line = (
                f"  diversity = {c['diversity']:+.4f} = W_DIVERSITY({W_DIVERSITY}) × "
                f"q({r.q:.3f}) × (1 − reuse {r.reuse:.2f}); endings: {r.ending_detail}"
            )
        else:
            div_line = (
                f"  diversity = {c['diversity']:+.4f} (none — bucket '{r.bucket}' is "
                f"not the group's top band)"
            )
        if r.in_top_band:
            len_part = W_CONCISE * r.q * r.len_eff
            tool_eff = self._tool_eff(r)
            conc_line = (
                f"  conciseness = {c['conciseness']:+.4f} = length {len_part:+.4f} "
                f"[W_CONCISE({W_CONCISE})×q({r.q:.3f})×len_eff({r.len_eff:.2f})] + tool-eff "
                f"{tool_eff:+.4f} [W_TOOL_EFF({W_TOOL_EFF})×q×factor, "
                f"{r.n_tool_calls}/{self._max_tool_calls} calls]"
            )
        else:
            conc_line = (
                f"  conciseness = +0.0000 (none — bucket '{r.bucket}' is not the group's top band)"
            )
        return (
            f"[TelestichEnv][why] poem {r.rollout_id} — {r.bucket.upper()} bucket, "
            f"reward={sum(c.values()):+.3f}\n"
            f"  quality   = {c['quality']:+.4f}  (ladder score {r.quality:.3f} → q {r.q:.3f})\n"
            f"  rhyme     = {c['rhyme']:+.4f} = W_RHYME({W_RHYME}) × q({r.q:.3f}) × "
            f"rhyme_score({rhyme_score(r.lines):.2f}) | {rhyme_detail(r.lines)}\n"
            f"{div_line}\n"
            f"{conc_line}"
        )

    async def compute_group_rewards(
        self,
        rollouts: Sequence[BaseRollout],
    ) -> Mapping[str, dict[str, float]]:
        """Score a complete sibling group and key rewards by rollout ID."""

        if not rollouts:
            return {}

        task = rollouts[0].example_args
        target = str(task.get("ground_truth") or "")
        prompt = str(task.get("prompt") or "")
        acceptable = _first_ref(task.get("acceptable_refs"))
        great = _first_ref(task.get("great_refs"))
        language = detect_language(target)
        budget = LEN_BUDGET_BASE + LEN_BUDGET_PER_LINE * max(1, len(target.strip()))

        # ── Stage 0: parse each rollout once into a record ──
        rolls = [
            self._build_rollout(index, rollout, target) for index, rollout in enumerate(rollouts)
        ]
        correct = [r for r in rolls if r.correct]
        partial = [r for r in rolls if r.partial]
        n_gated = len(rolls) - len(correct) - len(partial)
        logger.info(
            f"[TelestichEnv] group target={target!r} ({language}) — "
            f"{len(correct)} correct, {len(partial)} partial, {n_gated} gated (of {len(rolls)}; "
            f"anchors: baseline={'Y' if acceptable else 'N'}, great={'Y' if great else 'N'})"
        )
        if n_gated:
            # counts-by-kind, worst-offender first, so failures read at a glance
            cats = Counter(
                _gate_category(r.reason) for r in rolls if not r.correct and not r.partial
            )
            breakdown = ", ".join(f"{cat} {ct}" for cat, ct in cats.most_common())
            logger.info(f"[TelestichEnv]   stage1 gated {n_gated}: {breakdown}")

        # ── Stage 1+2: quality. Only CORRECT poems are judged/ranked; partial poems
        #    get a small graded score (no judge call). ──
        for r in partial:
            r.quality = r.partial_score
            r.q = r.partial_score
            r.bucket = _bucket(r.q)
        if correct:
            await self._score_quality(prompt, correct, acceptable, great)
            occ = Counter(r.bucket for r in correct)
            logger.info(
                f"[TelestichEnv] stage2 quality (correct only): buckets "
                f"below={occ['below']} mid={occ['mid']} above={occ['above']}"
            )

        # ── Stage 3: secondary terms (rhyme, diversity, conciseness) onto records ──
        self._apply_secondary(rolls, correct, budget)

        # ── Stage 4: assemble + log ──
        # Path A (flow): one summary line per rollout (gated + valid alike).
        # Path B (why): a full component breakdown for poems that passed the gate.
        out: dict[str, dict[str, float]] = {}
        for r in rolls:
            with bind_rollout_context(r.rollout_id):
                logger.info(self._fmt_rollout(r, budget))
                if r.correct:  # full breakdown only for judged poems
                    logger.info(self._explain_rollout(r, budget))
            out[r.rollout_id] = r.components
        return out

    def _build_rollout(
        self,
        idx: int,
        rollout: BaseRollout,
        target: str,
    ) -> _Rollout:
        """Stage 0: parse a rollout's completion once into a record."""
        completion = extract_completion_text(rollout.messages) or ""
        poem = final_poem(rollout.messages) or ""
        chk = check_hard_rules(poem, target)
        return _Rollout(
            idx=idx,
            rollout_id=rollout.rollout_id,
            completion_len=len(completion),
            poem=poem,
            poem_len=len(poem),
            lines=chk["lines"],
            language=chk["language"],
            n_tool_calls=_count_tool_calls(rollout.messages),
            correct=chk["correct"],
            reason=chk["reason"],
            partial=chk["partial"],
            partial_score=chk["partial_score"],
            misses=chk["misses"],
        )


# ── Runnable entrypoint ──────────────────────────────────────────────────────

MODEL = os.environ.get("TELESTICH_MODEL", "Qwen/Qwen3.5-4B")
VALIDATE_MODEL = os.environ.get("TELESTICH_VALIDATE_MODEL", "gpt-5.4-mini")
RUN_NAME = os.environ.get("TELESTICH_RUN_NAME", "telestich")
RUNTIME_DEPENDENCIES = ["english_words", "openai", "pronouncing", "wordfreq"]
DATASET_PATH = Path(__file__).parent / "telestich_dataset.jsonl"
DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"
TRAINING_ARGS = {"model": MODEL}
SPLIT_SEED = 42


def _rows() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    return [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]


def generate_data(*, force: bool) -> dict[str, Path]:
    dataset_files = {
        "train.jsonl": TRAIN_FILE,
        "eval.jsonl": EVAL_FILE,
    }
    if TRAIN_FILE.exists() and EVAL_FILE.exists() and not force:
        print(f"data: using existing {TRAIN_FILE.name} / {EVAL_FILE.name}")
        return dataset_files

    train_rows, eval_rows = _split_rows()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(TRAIN_FILE, train_rows)
    _write_jsonl(EVAL_FILE, eval_rows)
    print(f"data: wrote {len(train_rows)} train / {len(eval_rows)} eval examples")
    return dataset_files


def _split_rows() -> tuple[list[dict], list[dict]]:
    """Hold out ~10% eval at random; TRAIN keeps curriculum order (simpler first)."""

    examples = _rows()
    if len(examples) < 2:
        raise SystemExit(f"Need >=2 examples, got {len(examples)}.")
    n_eval = max(1, len(examples) // 10)
    eval_idx = set(random.Random(SPLIT_SEED).sample(range(len(examples)), n_eval))
    eval_rows = [e for i, e in enumerate(examples) if i in eval_idx]
    train_rows = [e for i, e in enumerate(examples) if i not in eval_idx]
    return train_rows, eval_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )


def validate(env: TelestichEnv, uploaded_assets: Any) -> Any:
    from castform import validate_environment

    report = asyncio.run(
        validate_environment(
            env,
            model=VALIDATE_MODEL,
            split="eval",
            base_dir=DATA_DIR,
            remote_assets=uploaded_assets,
            max_context_tokens=2048,
        )
    )
    _print_validation(report)
    return report


def launch(uploaded_assets: Any, *, assume_yes: bool) -> str | None:
    from castform import config
    from castform.platform.client import TrainerClient

    if not assume_yes:
        reply = input("launch training on GPUs? this spends credits. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("launch: cancelled")
            return None

    with TrainerClient() as trainer:
        run_id = trainer.launch_training_run(
            name=RUN_NAME,
            launcher_args=TRAINING_ARGS,
            **dataclasses.asdict(uploaded_assets),
        )
    print(f"launch: started {run_id}")
    print(f"view: {config.web_app_url()}/train/{run_id}")
    return run_id


def _print_validation(report: Any) -> None:
    for location in ("local", "remote"):
        outcomes = getattr(report, location)
        if outcomes is None:
            continue
        errors = getattr(report, f"{location}_errors")
        for rollout_id, outcome in outcomes.items():
            if rollout_id in errors:
                print(f"❌ {location} {rollout_id}: {errors[rollout_id]}")
            else:
                print(
                    f"✅ {location} {rollout_id}: "
                    f"{outcome.termination_reason} {dict(outcome.rewards)}"
                )
        for rollout_id, error in errors.items():
            if rollout_id not in outcomes:
                print(f"❌ {location} {rollout_id}: {error}")
    print("✅ validation passed" if report.ok else "❌ validation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=("data", "validate", "launch"),
        default="validate",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the launch confirmation",
    )
    args = parser.parse_args(argv)
    total_stages = {"data": 1, "validate": 4, "launch": 5}[args.action]

    print(f"[stage 1/{total_stages}] generating data")
    dataset_files = generate_data(force=args.force)
    if args.action == "data":
        return 0

    from castform import config

    ensure_session()
    constructor_args = {"judge_base_url": config.llm_url()}
    print(f"[stage 2/{total_stages}] bundling environment")
    bundled_environment = dump_bundle(
        TelestichEnv,
        constructor_args=constructor_args,
        pip_dependencies=RUNTIME_DEPENDENCIES,
    )
    print(f"[stage 3/{total_stages}] uploading environment and dataset")
    uploaded_assets = upload_assets(
        bundle=bundled_environment,
        dataset_files=dataset_files,
        run_name=RUN_NAME,
    )
    print(f"[stage 4/{total_stages}] validating environment")
    report = validate(TelestichEnv(**constructor_args), uploaded_assets)
    if not report.ok:
        return 1
    if args.action == "launch":
        print(f"[stage 5/{total_stages}] launching training")
        launch(uploaded_assets, assume_yes=args.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
