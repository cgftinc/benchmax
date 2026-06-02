"""TelestichEnv — reward env for telestich (last-letter acrostic) poems.

A telestich hides a target word in the last letter (or Chinese character) of
each line. Reward, per rollout in a GRPO group (see ``compute_group_reward``):

  1. HARD RULES (deterministic, no LLM): the poem must be a valid telestich —
     acrostic spells the target, every line ends on a real word, right line
     count — and must NOT write the hidden word in the body. Fail → reward 0,
     no judge called.
  2. QUALITY (one judge call per group, via benchmax.rubrics): the shared
     multi-anchor `evaluate_rubric_ranking` ranks the group's valid poems with
     BOTH reference poems inserted blind — acceptable as a floor, great as the
     bar — placing each poem in a band: below acceptable -> [0,0.1) ("below"),
     acceptable..great -> [0.1,0.5] ("mid"), above great -> [0.5,1.0] ("above").
  3. BONUSES (deterministic, all POSITIVE and quality-scaled, so the total reward
     is >= 0): ending-word diversity (anti mode-collapse, mid+above buckets), a
     rhyme/length form bonus, and a conciseness bonus awarded only to the top
     occupied band — rewarding a good poem that reached its answer quickly (short
     overall rollout, plus a small tool-call-thrift part).

The final reward = quality + diversity + form + conciseness
(the component values sum to it; every component is >= 0).
"""

import logging
import math
import random
import re
from collections import Counter, defaultdict
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

# Quality is scored by the shared MULTI-ANCHOR rubric judge (benchmax.rubrics):
# one judge call ranks the group's valid poems with BOTH of the example's
# reference poems inserted blind — `acceptable` as a floor, `great` as the bar.
# A poem's score falls in one of three bands by where it ranks vs the anchors:
#     below acceptable   -> [0, 0.1)    "below" bucket  (degenerate / sub-par)
#     acceptable..great   -> [0.1, 0.5]  "mid" bucket
#     above great         -> [0.5, 1.0]  "above" bucket
# The floor anchor gives quality an absolute zero-point: an all-bad group ranks
# below acceptable -> everyone near 0 (pure relative ranking can't do this).
ACCEPTABLE_EDGE = 0.1    # score earned by tying the acceptable (floor) anchor
GREAT_EDGE = 0.5         # score earned by tying the great (bar) anchor
QUALITY_GATE = ACCEPTABLE_EDGE   # secondary BONUSES require quality >= this
                                 # (i.e. mid/above buckets); below it earns no bonus
QUALITY_RUBRIC = Rubric(
    title="poetic quality",
    description=(
        "Judge the poem's craft only (the acrostic is already verified): coherence "
        "(reads as one deliberate poem, not interchangeable lines), concrete vivid "
        "imagery, faithfulness to the requested theme/voice/tone, natural un-forced "
        "line endings (not bent just to hit a letter), no nonsense lines, no template "
        "lock. Prefer economy — every line should earn its place; penalize padding, "
        "filler, and prose run-on lines that read like an essay rather than a poem."
    ),
    type="positive",
)

# Secondary terms are all SCALED BY QUALITY, so a weak poem can't farm them.
W_FORM = 0.15        # rhyme density (en) / line-length uniformity (zh)
W_DIVERSITY = 0.50   # unique-ending bonus (anti mode-collapse); mid+above buckets only

# Conciseness — a POSITIVE bonus (like diversity: quality-scaled), awarded only
# to the TOP occupied band (mid/above) so we reward a GOOD poem that also reached
# its answer quickly. Two parts: overall rollout length (the bulk) and tool-call
# thrift (a small part). A long-winded or tool-heavy top poem just earns less of
# it; gated / below-bucket poems earn none — so every component stays >= 0.
W_CONCISE = 0.15            # max conciseness bonus (quality-scaled)
CONCISE_TOOL_WEIGHT = 0.2   # tool thrift is a small part; rollout length is the rest
CALL_DECAY = 0.35           # tool efficiency = exp(-CALL_DECAY * n_tool_calls)
LEN_BUDGET_BASE = 1500      # completion-char budget = BASE + PER_LINE * acrostic_len;
LEN_BUDGET_PER_LINE = 600   # at/under budget = full length-efficiency, decaying above

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
    return inner[idx + len("<answer>"):].strip() if idx != -1 else None


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
# WORD VALIDITY + WORD BANK
# ══════════════════════════════════════════════════════════════════════
_WEB2 = get_english_words_set(["web2"], lower=True)
_FREQ_THRESHOLD = 1e-7
_MIN_WORD_LEN = 2


def is_valid_word(word: str) -> bool:
    w = (word or "").lower()
    if len(w) < _MIN_WORD_LEN:
        return False
    return w in _WEB2 or word_frequency(w, "en") > _FREQ_THRESHOLD


_ENDING_INDEX: dict[str, list[tuple[str, float]]] | None = None


def _build_ending_index() -> dict[str, list[tuple[str, float]]]:
    """letter -> [(word, freq)] for real, poem-usable words ending in that letter.

    Filters out function words and proper-noun/abbreviation noise by keeping
    only dictionary words >= 3 letters (so 'i'/'a' fillers and 'to'/'of'/'hi'
    fillers don't dominate the bank — the source of the ski/hi crutch).
    """
    index: dict[str, list[tuple[str, float]]] = defaultdict(list)
    bad = re.compile(r"[\s.\-]")
    for w in _WEB2:
        if len(w) < 3 or not w[-1].isalpha() or bad.search(w):
            continue
        freq = word_frequency(w, "en")
        if freq > _FREQ_THRESHOLD:
            index[w[-1].lower()].append((w, freq))
    for letter in index:
        seen, deduped = set(), []
        for w, f in sorted(index[letter], key=lambda x: -x[1]):
            if w not in seen:
                seen.add(w)
                deduped.append((w, f))
        index[letter] = deduped
    return dict(index)


def word_bank(letter: str, k: int = 30) -> list[str]:
    """k frequency-weighted real words ending in `letter` (a-z)."""
    global _ENDING_INDEX
    if _ENDING_INDEX is None:
        _ENDING_INDEX = _build_ending_index()
    letter = (letter or "").lower().strip()
    if len(letter) != 1 or not letter.isalpha():
        return []
    pool = _ENDING_INDEX.get(letter, [])[:200]
    if len(pool) <= k:
        return [w for w, _ in pool]
    words, weights = [w for w, _ in pool], [f for _, f in pool]
    out: list[str] = []
    while len(out) < k and words:
        i = random.choices(range(len(words)), weights=weights, k=1)[0]
        out.append(words.pop(i))
        weights.pop(i)
    return out


# ══════════════════════════════════════════════════════════════════════
# HARD RULES
# ══════════════════════════════════════════════════════════════════════
# A telestich line must be an actual poetic line, not the bare acrostic spelled
# vertically. These floors kill the degenerate "one char/word per line" hack
# (e.g. 挚/友/如/镜) with a wide margin — in the great-ref poems the SHORTEST
# real line is 11 CJK chars (zh) / 6 words (en), so legit poems never trip these.
MIN_ZH_LINE_CHARS = 4
MIN_EN_LINE_WORDS = 3


def check_hard_rules(poem: str | None, target: str) -> dict:
    """Deterministic gate. correct == True iff valid telestich AND not cheating.
    `reason` names the first rule that failed (empty string when correct)."""
    target = (target or "").strip()
    language = detect_language(target)
    lines = parse_poem_lines(poem or "")
    chars = list(target.lower()) if language == "en" else list(target)
    n = len(chars)

    cheated = contains_hidden_word(poem or "", target, language)

    def result(correct: bool, reason: str) -> dict:
        return {"correct": correct, "cheated": cheated, "language": language,
                "lines": lines, "reason": reason}

    if n == 0:
        return result(False, "empty target word")
    if not lines:
        return result(False, "no poem found in completion")
    if len(lines) != n:
        return result(False, f"line count {len(lines)} != target length {n}")
    for i in range(n):
        got = last_char(lines[i], language)
        if got != chars[i]:
            return result(False, f"line {i + 1} ends '{got}', expected '{chars[i]}'")
        if language == "zh":
            # reject the acrostic spelled vertically (one char per line)
            cjk = len(_CJK_RE.findall(lines[i]))
            if cjk < MIN_ZH_LINE_CHARS:
                return result(False, f"line {i + 1} too short ({cjk} chars) — not a poem line")
        else:
            if not is_valid_word(last_word(lines[i])):
                return result(False, f"line {i + 1} ends on non-word '{last_word(lines[i])}'")
            if len(lines[i].split()) < MIN_EN_LINE_WORDS:
                return result(False, f"line {i + 1} too short — not a poem line")
    if cheated:
        return result(False, "hidden word written in the poem body")
    return result(True, "")


# ══════════════════════════════════════════════════════════════════════
# FORM  (rhyme density for EN, line-length uniformity for ZH)
# ══════════════════════════════════════════════════════════════════════
def _rhyme_analysis(lines: list[str]) -> tuple[float, list[list[int]], list[str], list[int]]:
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


def _english_rhyme_density(lines: list[str]) -> float:
    return _rhyme_analysis(lines)[0]


def _mandarin_length_uniformity(lines: list[str]) -> float:
    lengths = [len(_CJK_RE.findall(line)) for line in lines]
    nz = [n for n in lengths if n > 0]
    if len(nz) < 2:
        return 0.0
    modal = Counter(nz).most_common(1)[0][0]
    return sum(1 for n in lengths if n == modal) / len(nz)


def form_score(lines: list[str], language: str) -> float:
    return _mandarin_length_uniformity(lines) if language == "zh" else _english_rhyme_density(lines)


def form_detail(lines: list[str], language: str) -> str:
    """Human-readable explanation of the form score — which lines rhyme (en) or
    the line-length profile (zh)."""
    if language == "zh":
        lengths = [len(_CJK_RE.findall(line)) for line in lines]
        nz = [n for n in lengths if n > 0]
        modal = Counter(nz).most_common(1)[0][0] if nz else 0
        match = sum(1 for n in nz if n == modal)
        return f"CJK lengths {lengths}, modal {modal} → {match}/{len(nz)} lines uniform"
    density, clusters, endings, idx = _rhyme_analysis(lines)
    rhyming = [c for c in clusters if len(c) >= 2]
    groups = "; ".join(
        "+".join(f"L{i + 1}:{endings[i]}" for i in c) for c in rhyming
    ) or "no rhyming pairs"
    unmatched = [f"L{i + 1}:{endings[i]}" for i in idx
                 if not any(i in c for c in rhyming)]
    largest = max((len(c) for c in rhyming), default=0)
    tail = f"; unmatched {unmatched}" if unmatched else ""
    return f"largest rhyme {largest}/{len(idx)} → {density:.2f} | {groups}{tail}"


# ══════════════════════════════════════════════════════════════════════
# DIVERSITY
# ══════════════════════════════════════════════════════════════════════
def ending_reuse_detail(poems: list[str]) -> tuple[list[float], list[list[tuple[str, int]]]]:
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
    completion_len: int          # chars of the whole completion (reasoning + answer)
    poem: str                    # parsed <answer> poem ("" if none)
    poem_len: int
    lines: list[str]
    language: str
    n_tool_calls: int
    valid: bool                  # passed the hard-rules gate
    reason: str                  # first failed rule ("" when valid)
    quality: float = 0.0         # raw rubric band score
    q: float = 0.0               # quality after the duplicate divisor (used everywhere)
    bucket: str = "gated"        # gated | below | mid | above
    reuse: float = 0.0           # ending-word reuse vs siblings (0=unique, 1=all shared)
    ending_detail: str = ""      # per-ending share counts, for the breakdown log
    len_eff: float = 0.0         # conciseness efficiency factors in [0,1] (logged)
    tool_eff: float = 0.0
    components: dict = field(default_factory=dict)


def _bucket(q: float) -> str:
    """Map a quality score to its band bucket."""
    if q < ACCEPTABLE_EDGE:
        return "below"
    if q < GREAT_EDGE:
        return "mid"
    return "above"


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
# ENV CLASS
# ══════════════════════════════════════════════════════════════════════
class TelestichEnv(BaseEnv):
    system_prompt = """\
A telestich is a poem where the last letter (or character, for Chinese) of \
each line, read top to bottom, spells out a hidden word. The user will tell \
you what word to hide.

Extracting the target word from the request:
- The target may appear plain (marry), in quotes ("marry" or 'marry'), in \
ALL CAPS (MARRY), or Capitalized (Marry). Normalize it to lowercase \
(English) or as-is (Chinese) before spelling it out.
- Strip surrounding quotes and punctuation. The hidden word is always a \
single word or short Chinese phrase — never a sentence.
- Phrases like "X at the end of every line" refer to the last LETTER of \
each line spelling X (a telestich), not to repeating the word X at each \
line end.

Process (keep your thinking SHORT — a few lines of notes, then the poem):
1. Spell out the target word letter by letter (or character by character). \
Count — that's how many lines.
2. You have 2 tool calls total. Use word_bank only for the hardest letters \
where you can't readily think of ending words. (For Chinese poems, the tool is \
not available — rely on your own vocabulary.)
3. Pick ONE ending word per line and commit. Build each line as a natural \
phrase around it.
4. Output the poem in <answer></answer> tags. Plain text only. Stop after \
</answer>.

Be concise. Reaching a correct answer quickly is rewarded, and a rollout that \
rambles until it runs out of room and never reaches <answer> earns nothing. So \
do NOT enumerate long candidate lists, do NOT second-guess or re-derive, and do \
NOT restate the rules. Choose your ending words in a line or two and write the \
poem.

Rules:
1. Exactly as many lines as letters/characters in the target word.
2. Every ending word must be a real word — no invented words, no standalone \
letters tacked on.
3. Each line is a complete, meaningful phrase — keep lines tight and \
image-dense (a poetic line, not a long prose sentence). The poem should be \
coherent and match the requested theme.
4. Write the poem in the same language as the request.
5. Do NOT include the hidden target word anywhere in the poem itself — the \
whole point is that it's hidden in the ending letters.
6. Form preference (extra credit when correctness is perfect):
   - English: rhyme line endings with each other. The score rewards every \
line whose ending word rhymes with at least one other line — arrangement \
doesn't matter. So you can pair them (line 1 with 2, 3 with 4), interleave \
them (1 with 3, 2 with 4), have all lines share one rhyme, or any mix. \
Pick ending words whose vowel sounds match, not just whose letters match \
(`tree/free` rhymes, `tree/dry` doesn't). Don't force it if it breaks \
coherence.
   - Chinese: keep all lines the same length (count CJK characters per \
line; aim for uniform 5- or 7-character lines, or whatever length fits the \
hidden word naturally).

English example:

User: breakup poem where the last letters spell brave

B-R-A-V-E = 5 letters, 5 lines. Topic: breakup.

<answer>
My chest still holds the dull familiar throb
of mornings spent pretending not to remember
that love like ours was never just an idea
now I just sit alone and stare at the tv
and wonder if you ever meant to keep your promise
</answer>

中文例子:

User: 写一首关于思念的藏尾诗，尾字拼出"月光"

月-光 = 2个字，2行。主题：思念。

<answer>
独坐窗前望明月
故人远去似流光
</answer>"""

    def __init__(self, *, judge_base_url: str, judge_api_key: str = "",
                 judge_timeout: float = 90.0, max_tool_calls: int = 2,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._judge_base_url = judge_base_url
        # Empty → judge bearer resolves via the platform act-as seam
        # (resolve_judge_key). Don't bake a literal key into constructor_args.
        self._judge_api_key = judge_api_key
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
        return [ToolDefinition(
            name="word_bank",
            description="Returns ~30 real English words ending in the given letter (a-z). Up to 2 calls.",
            input_schema={"type": "object",
                          "properties": {"letter": {"type": "string", "description": "one letter a-z"}},
                          "required": ["letter"]},
        )]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name != "word_bank":
            return f"Error: unknown tool '{tool_name}'."
        used = self._tool_calls.get(rollout_id, 0)
        if used >= self._max_tool_calls:
            return "No tool calls remaining."
        self._tool_calls[rollout_id] = used + 1
        words = word_bank(tool_args.get("letter", ""))
        return ", ".join(words) if words else "No common words found for that letter."

    async def release_rollout(self, rollout_id: str) -> None:
        self._tool_calls.pop(rollout_id, None)  # avoid unbounded growth over a run

    async def compute_reward(self, rollout_id, messages, task, **kwargs) -> dict[str, float]:
        return {}  # all logic is group-level (compute_group_reward)

    async def _quality(self, prompt: str, poems: list[str],
                       anchors: list[str], band_edges: list[float]) -> list[float]:
        """Quality via the shared MULTI-ANCHOR rubric judge: one call ranks `poems`
        with the anchors (acceptable floor + great bar) inserted blind, returning
        band scores in [0,1] aligned with `poems`. On judge error the rubric
        returns 0s. With no anchors it degrades to pure relative ranking."""
        res = await evaluate_rubric_ranking(
            rubric=QUALITY_RUBRIC,
            question=prompt,
            responses=poems,
            model_name=JUDGE_MODEL,
            base_url=self._judge_base_url,
            api_key=self._judge_api_key,
            timeout=self._judge_timeout,
            anchors=anchors or None,
            band_edges=band_edges or None,
        )
        return res["scores"]

    async def _score_quality(self, prompt: str, valid: list[_Rollout],
                             acceptable: str | None, great: str | None) -> None:
        """Rank the valid poems against the acceptable + great anchors; write each
        record's band `quality`, duplicate-adjusted `q`, and `bucket`."""
        anchors, edges = [], []
        if acceptable:
            anchors.append(acceptable)
            edges.append(ACCEPTABLE_EDGE)
        if great:
            anchors.append(great)
            edges.append(GREAT_EDGE)
        scores = await self._quality(prompt, [r.poem for r in valid], anchors, edges)
        dup = duplicate_divisors([r.poem for r in valid])
        for local, r in enumerate(valid):
            r.quality = scores[local]
            r.q = r.quality / dup[local]    # near-duplicate whole poems shared down
            r.bucket = _bucket(r.q)

    def _apply_secondary(self, rolls: list[_Rollout], valid: list[_Rollout],
                         budget: int) -> None:
        """Compute every rollout's reward components onto its record:
        form + diversity (quality-scaled bonuses, valid poems) and conciseness
        (a hard penalty on every rollout)."""
        if valid:
            reuse, reuse_detail = ending_reuse_detail([r.poem for r in valid])
            for local, r in enumerate(valid):
                r.reuse = reuse[local]
                r.ending_detail = ", ".join(
                    f"{w}(shared×{c})" if c else f"{w}(unique)" for w, c in reuse_detail[local]
                )

        # conciseness bonus targets only the TOP occupied band (above, else mid);
        # if every poem is below the floor there is no band worth rewarding.
        top_bucket = next((b for b in ("above", "mid")
                           if any(r.bucket == b for r in valid)), None)

        for r in rolls:
            # efficiency factors in [0,1]: 1 = answered within budget / no tools
            r.len_eff = math.exp(-max(0.0, r.completion_len / budget - 1.0))
            r.tool_eff = math.exp(-CALL_DECAY * r.n_tool_calls)

            if not r.valid:
                r.components = {"quality": 0.0, "form": 0.0, "diversity": 0.0,
                                "conciseness": 0.0}
                continue
            form_bonus = W_FORM * r.q * form_score(r.lines, r.language)
            # diversity is gated to the mid+above buckets (quality >= floor)
            div_bonus = (W_DIVERSITY * r.q * (1.0 - r.reuse)
                         if r.q >= QUALITY_GATE else 0.0)
            # conciseness: top-band poems only, quality-scaled, reward reaching the
            # answer quickly (short rollout, plus a small tool-thrift part)
            if r.bucket == top_bucket:
                eff = (1.0 - CONCISE_TOOL_WEIGHT) * r.len_eff + CONCISE_TOOL_WEIGHT * r.tool_eff
                concise = W_CONCISE * r.q * eff
            else:
                concise = 0.0
            r.components = {"quality": round(r.q, 4),
                            "form": round(form_bonus, 4),
                            "diversity": round(div_bonus, 4),
                            "conciseness": round(concise, 4)}

    def _fmt_rollout(self, r: _Rollout, budget: int) -> str:
        """One path-revealing log line per rollout."""
        total = sum(r.components.values())
        meta = (f"completion_len={r.completion_len}/{budget} poem_len={r.poem_len} "
                f"lines={len(r.lines)} tools={r.n_tool_calls}")
        if not r.valid:
            return f"[TelestichEnv] reward={total:+.3f} GATED ({r.reason}) | {meta}"
        return (f"[TelestichEnv] reward={total:+.3f} {r.bucket.upper()} "
                f"q={r.quality:.3f}->{r.q:.3f} reuse={r.reuse:.2f} "
                f"len_eff={r.len_eff:.2f} | {meta} | {r.components}")

    def _explain_rollout(self, r: _Rollout, budget: int) -> str:
        """Multi-line WHY-breakdown for a poem that passed correctness — spells
        out how each component reached its value (which lines rhyme, which
        endings are shared, why conciseness was charged)."""
        c = r.components
        if r.q >= QUALITY_GATE:
            div_line = (f"  diversity = {c['diversity']:+.4f} = W_DIVERSITY({W_DIVERSITY}) × "
                        f"q({r.q:.3f}) × (1 − reuse {r.reuse:.2f}); endings: {r.ending_detail}")
        else:
            div_line = (f"  diversity = {c['diversity']:+.4f} (none — q {r.q:.3f} below the "
                        f"{QUALITY_GATE} floor, bucket={r.bucket})")
        if c["conciseness"]:
            conc_line = (f"  conciseness = {c['conciseness']:+.4f} = W_CONCISE({W_CONCISE}) × "
                         f"q({r.q:.3f}) × [{1 - CONCISE_TOOL_WEIGHT:g}×len_eff({r.len_eff:.2f}) "
                         f"+ {CONCISE_TOOL_WEIGHT:g}×tool_eff({r.tool_eff:.2f})]; "
                         f"completion {r.completion_len}/{budget}, {r.n_tool_calls} tools")
        else:
            conc_line = (f"  conciseness = +0.0000 (none — not in the top band "
                         f"'{r.bucket}', conciseness rewards only the best poems)")
        return (
            f"[TelestichEnv][why] poem {r.rollout_id} — {r.bucket.upper()} bucket, "
            f"reward={sum(c.values()):+.3f}\n"
            f"  quality   = {c['quality']:+.4f}  (judge band {r.quality:.3f}"
            f"{f', ÷dup→{r.q:.3f}' if abs(r.q - r.quality) > 1e-9 else ''})\n"
            f"  form      = {c['form']:+.4f} = W_FORM({W_FORM}) × q({r.q:.3f}) × "
            f"form_score({form_score(r.lines, r.language):.2f}) | {form_detail(r.lines, r.language)}\n"
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
        rolls = [self._build_rollout(i, rollout_ids[i], messages_list[i], target)
                 for i in range(len(rollout_ids))]
        valid = [r for r in rolls if r.valid]
        gated_reasons = Counter(r.reason for r in rolls if not r.valid)
        logger.info(
            f"[TelestichEnv] group target={target!r} ({language}) n={len(rolls)} "
            f"anchors=(acc={'Y' if acceptable else 'N'},great={'Y' if great else 'N'}) "
            f"-> stage1 gate {len(valid)}/{len(rolls)} valid; gated={dict(gated_reasons)}")

        # ── Stage 1+2: quality via the multi-anchor rubric → band score + bucket ──
        if valid:
            await self._score_quality(prompt, valid, acceptable, great)
            occ = Counter(r.bucket for r in valid)
            logger.info(f"[TelestichEnv] stage2 quality: buckets "
                        f"below={occ['below']} mid={occ['mid']} above={occ['above']}")

        # ── Stage 3: secondary terms (form, diversity, conciseness) onto records ──
        self._apply_secondary(rolls, valid, budget)

        # ── Stage 4: assemble + log ──
        # Path A (flow): one summary line per rollout (gated + valid alike).
        # Path B (why): a full component breakdown for poems that passed the gate.
        out = []
        for r in rolls:
            with rollout_context(r.rollout_id):
                logger.info(self._fmt_rollout(r, budget))
                if r.valid:
                    logger.info(self._explain_rollout(r, budget))
            out.append(r.components)
        return out

    def _build_rollout(self, idx: int, rollout_id: str, messages: Messages,
                       target: str) -> _Rollout:
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
            valid=chk["correct"],
            reason=chk["reason"],
        )
