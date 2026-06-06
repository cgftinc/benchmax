"""TelestichEnv — reward env for telestich (last-letter acrostic) poems.

A telestich hides a target word in the last letter of each line. Reward per
rollout in a GRPO group (see ``compute_group_reward``):

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
"""

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import pronouncing
from english_words import get_english_words_set
from wordfreq import word_frequency

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.logging import rollout_context
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.rubrics.rubric import Rubric, evaluate_rubric_ranking

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
        "through-line of sense — HEAVILY penalize interchangeable lines, list-like "
        "poems, or sentences that don't actually parse.\n"
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
    type="positive",
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
        f"line {i + 1} ('{last_word(lines[i]) or '∅'}'→needs '{chars[i]}')"
        for i in miss_idx
    )
    if good / n > MIN_CORRECT_FRAC:  # more than 25% of lines correct → partial credit
        max_partial = (
            math.ceil((1 - MIN_CORRECT_FRAC) * n) - 1
        )  # most misses still above the bar
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
            if endings[i] != endings[j] and any(
                x == y for x in parts[i] for y in parts[j]
            ):
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
    unmatched = [
        f"L{i + 1}:{endings[i]}" for i in idx if not any(i in c for c in rhyming)
    ]
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
    final assembly and logging read straight off it (see compute_group_reward)."""

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
    in_top_band: bool = (
        False  # in the group's top occupied bucket → earns secondary bonuses
    )
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
    band_edges=[ACCEPTABLE_EDGE, GREAT_EDGE] the rubric already returns
    normal..great as [0.4, 0.7] and above-great as (0.7, 1.0]; we only floor the
    below-normal band [0, 0.4) up to [MIN_CORRECT, 0.4) so the worst correct poem
    still scores MIN_CORRECT, not ~0."""
    if s < ACCEPTABLE_EDGE:
        return MIN_CORRECT + s * (ACCEPTABLE_EDGE - MIN_CORRECT) / ACCEPTABLE_EDGE
    return s


def _first_ref(refs: Any) -> str | None:
    """First non-empty reference poem from a refs list, or None."""
    items = [r for r in (refs or []) if r and str(r).strip()]
    return items[0] if items else None


_TOOL_CALL_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def _count_tool_calls(completion: str) -> int:
    """Count tool-call attempts in the completion (includes rejected over-budget
    ones, which the env's own counter caps out — this is what flags waste)."""
    return len(_TOOL_CALL_RE.findall(completion or ""))


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
            f"Line count: you have {a} line(s) but '{target}' needs {n} "
            f"(one per letter) — {verb}."
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
            issues.append(
                f"must end in '{req}', but \"{lw or lines[i]}\" ends in '{got or '∅'}'"
            )
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
                f'contains the hidden word "{target}" — keep it hidden in the last '
                f"letters only"
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
error-prone. Revise from its feedback, then finalize. Use it sparingly (≤3 calls — usually one verify pass is \
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
        judge_timeout: float = 90.0,
        max_tool_calls: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._judge_base_url = judge_base_url
        self._judge_timeout = judge_timeout
        self._max_tool_calls = max_tool_calls
        self._tool_calls: dict[str, int] = {}

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        prompt = example.get("prompt", "")
        return make_example(
            prompt_messages=[{"role": "user", "content": prompt}],
            task={
                "prompt": prompt,
                "ground_truth": example.get("ground_truth", ""),
                "acceptable_refs": example.get("acceptable_refs", []),
                "great_refs": example.get("great_refs", []),
            },
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="feedback",
                description=(
                    "Feedback on a DRAFT telestich (up to 3 calls): pass your draft ('poem') "
                    "AND the hidden word it should spell ('word'); it checks every line ending "
                    "against that word — so you don't have to verify each last letter yourself "
                    "— and flags over-long (prose run-on) lines and filler endings, both of "
                    "which lower your score. Recommended flow: write a full draft, send it here "
                    "to check the endings and tighten, then put your revised best in <answer>. "
                    "Use it sparingly (often one pass is enough); a correct poem written with no "
                    "calls earns a small efficiency bonus, so wean off once you can reliably nail "
                    "it on the first try."
                ),
                input_schema={
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
            )
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
            "Looks good: the endings spell the word, every line ends in a real "
            "non-filler word, and no line runs long."
        )
        if not last_note:
            good += (
                " Put it in <answer> now rather than calling again — both fewer "
                "feedback calls and a shorter, un-padded answer score higher."
            )
        return good + last_note

    async def release_rollout(self, rollout_id: str) -> None:
        self._tool_calls.pop(rollout_id, None)  # avoid unbounded growth over a run

    async def compute_reward(
        self, rollout_id, messages, task, **kwargs
    ) -> dict[str, float]:
        return {}  # all logic is group-level (compute_group_reward)

    async def _quality(
        self,
        prompt: str,
        poems: list[str],
        anchors: list[str],
        band_edges: list[float],
        anchor_labels: list[str] | None = None,
    ) -> list[float]:
        """Quality via the shared MULTI-ANCHOR rubric judge. BATCHED: each call ranks
        at most JUDGE_BATCH poems against the anchors (acceptable floor + great bar)
        inserted blind. Ranking the whole group at once lets near-identical sibling
        poems contaminate each other's placement so the judge over-rates them above the
        anchors (audited on run 0ec8e2dc: full-group 15/36 false-better → batch-of-3
        0/36). Batches run concurrently; scores are returned in `poems` order. On judge
        error a batch returns 0s. `anchor_labels` only name the anchors in the debug log."""

        async def _score_batch(batch: list[str]) -> list[float]:
            res = await evaluate_rubric_ranking(
                rubric=QUALITY_RUBRIC,
                question=prompt,
                responses=batch,
                model_name=JUDGE_MODEL,
                base_url=self._judge_base_url,
                timeout=self._judge_timeout,
                anchors=anchors or None,
                band_edges=band_edges or None,
                anchor_labels=anchor_labels or None,
            )
            return res["scores"]

        batches = [
            poems[i : i + JUDGE_BATCH] for i in range(0, len(poems), JUDGE_BATCH)
        ]
        results = await asyncio.gather(*(_score_batch(b) for b in batches))
        return [s for batch_scores in results for s in batch_scores]

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
        anchors, edges, labels = [], [], []
        if acceptable:
            anchors.append(acceptable)
            edges.append(ACCEPTABLE_EDGE)
            labels.append("baseline")
        if great:
            anchors.append(great)
            edges.append(GREAT_EDGE)
            labels.append("great")
        scores = await self._quality(
            prompt, [r.poem for r in correct], anchors, edges, labels
        )
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

    def _apply_secondary(
        self, rolls: list[_Rollout], correct: list[_Rollout], budget: int
    ) -> None:
        """Compute every rollout's reward components onto its record. Gated poems →
        all-zero. Partial poems → graded partial_score quality, no bonuses. Correct poems:
        rhyme (any), plus diversity and conciseness (folds in the tool-efficiency
        bonus) applied only to the group's TOP occupied band (anti-collapse)."""
        if correct:
            reuse, reuse_detail = ending_reuse_detail([r.poem for r in correct])
            for local, r in enumerate(correct):
                r.reuse = reuse[local]
                r.ending_detail = ", ".join(
                    f"{w}(shared×{c})" if c else f"{w}(unique)"
                    for w, c in reuse_detail[local]
                )

        # the group's top occupied bucket among correct poems — secondary bonuses
        # apply only here, so even an all-"below" group still gets a gradient.
        top_band = next(
            (
                b
                for b in ("above", "mid", "below")
                if any(r.bucket == b for r in correct)
            ),
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
            if (
                r.partial
            ):  # small graded reward (r.q == partial_score); no judging/bonuses
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
                f"  conciseness = +0.0000 (none — bucket '{r.bucket}' is "
                f"not the group's top band)"
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

    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        task = tasks[0] or {}
        target = str(task.get("ground_truth") or "")
        prompt = str(task.get("prompt") or "")
        acceptable = _first_ref(task.get("acceptable_refs"))
        great = _first_ref(task.get("great_refs"))
        language = detect_language(target)
        budget = LEN_BUDGET_BASE + LEN_BUDGET_PER_LINE * max(1, len(target.strip()))

        # ── Stage 0: parse each rollout once into a record ──
        rolls = [
            self._build_rollout(i, rollout_ids[i], messages_list[i], target)
            for i in range(len(rollout_ids))
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
                _gate_category(r.reason)
                for r in rolls
                if not r.correct and not r.partial
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
        out = []
        for r in rolls:
            with rollout_context(r.rollout_id):
                logger.info(self._fmt_rollout(r, budget))
                if r.correct:  # full breakdown only for judged poems
                    logger.info(self._explain_rollout(r, budget))
            out.append(r.components)
        return out

    def _build_rollout(
        self, idx: int, rollout_id: str, messages: Messages, target: str
    ) -> _Rollout:
        """Stage 0: parse a rollout's completion once into a record."""
        completion = extract_completion_text(messages) or ""
        poem = final_poem(messages) or ""
        chk = check_hard_rules(poem, target)
        return _Rollout(
            idx=idx,
            rollout_id=rollout_id,
            completion_len=len(completion),
            poem=poem,
            poem_len=len(poem),
            lines=chk["lines"],
            language=chk["language"],
            n_tool_calls=_count_tool_calls(completion),
            correct=chk["correct"],
            reason=chk["reason"],
            partial=chk["partial"],
            partial_score=chk["partial_score"],
            misses=chk["misses"],
        )
