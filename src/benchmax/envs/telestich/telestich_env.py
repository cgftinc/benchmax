"""TelestichEnv — reward env for telestich (last-letter acrostic) poems.

A telestich is a poem where the last letter (or character, for Chinese) of
each line, read top to bottom, spells out a hidden target word. This env
rewards completions on acrostic correctness, judge-rated quality,
conciseness, and rhyme/form.

A runnable script that generates a dataset, bundles this env, and launches a
training run lives in ``example.py`` next to this module.
"""

import asyncio
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, List

import pronouncing
from english_words import get_english_words_set
from openai import AsyncOpenAI
from wordfreq import top_n_list, word_frequency

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.logging import rollout_context
from benchmax.envs.reward_helpers import clip01, extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition

logger = logging.getLogger(__name__)

# Judge model for the quality rubric.
JUDGE_MODEL = "gpt-5.4"



# ── Word-validity & letter-bank helpers ──
_WEB2_WORDS = get_english_words_set(["web2"], lower=True)
_WORD_FREQ_THRESHOLD = 1e-7
_MIN_WORD_LEN = 2


def _is_valid_word(word):
    """Check if a word is a real English word (not made up or a single letter)."""
    w = word.lower()
    if len(w) < _MIN_WORD_LEN:
        return False
    return w in _WEB2_WORDS or word_frequency(w, "en") > _WORD_FREQ_THRESHOLD


def _build_ending_index():
    """Build a map of letter -> list of common words ending in that letter."""
    index = defaultdict(list)
    bad_word_re = re.compile(r"[\s.\-]")
    for w in _WEB2_WORDS:
        if len(w) < _MIN_WORD_LEN or not w[-1].isalpha() or bad_word_re.search(w):
            continue
        freq = word_frequency(w, "en")
        if freq > _WORD_FREQ_THRESHOLD:
            index[w[-1].lower()].append((w, freq))
    # Also add high-freq words not in dict
    for w in top_n_list("en", 50000):
        if len(w) < _MIN_WORD_LEN or not w[-1].isalpha() or bad_word_re.search(w):
            continue
        if w not in _WEB2_WORDS:
            freq = word_frequency(w, "en")
            if freq > _WORD_FREQ_THRESHOLD:
                index[w[-1].lower()].append((w, freq))
    # Sort each letter by frequency descending, deduplicate
    for letter in index:
        seen = set()
        deduped = []
        for w, f in sorted(index[letter], key=lambda x: -x[1]):
            if w not in seen:
                seen.add(w)
                deduped.append((w, f))
        index[letter] = deduped
    return dict(index)


_ENDING_INDEX = None


def _get_ending_index():
    global _ENDING_INDEX
    if _ENDING_INDEX is None:
        _ENDING_INDEX = _build_ending_index()
    return _ENDING_INDEX


def _word_bank(letter: str) -> list[str]:
    """Return 30 random common words ending with the given letter.

    Samples from the top 200 candidates, weighted by frequency so that
    more common words appear more often. Repeated calls return different results.
    """
    letter = letter.lower().strip()
    if len(letter) != 1 or not letter.isalpha():
        return []
    entries = _get_ending_index().get(letter, [])
    pool = entries[:200]
    if len(pool) <= 30:
        return [w for w, _ in pool]
    weights = [f for _, f in pool]
    chosen = set()
    result = []
    # Weighted sampling without replacement
    candidates = list(pool)
    cand_weights = list(weights)
    while len(result) < 30 and candidates:
        picked = random.choices(candidates, weights=cand_weights, k=1)[0]
        w, _ = picked
        if w not in chosen:
            chosen.add(w)
            result.append(w)
            idx = candidates.index(picked)
            candidates.pop(idx)
            cand_weights.pop(idx)
    return result


# ── Judge prompt (v7: compact, classification-only; scoring is deterministic) ──
QUALITY_JUDGE_PROMPT = """\
You evaluate a telestich poem (a poem whose lines' last letters spell a hidden target word).

**User request**: {prompt}
**Poem**:
{poem_text}

---

You ONLY identify problems and rate two axes. Python computes the score from your output. Do NOT emit a score or verdict.

### Part 1 — Problems (empty list if none)

For each check, list offending lines with a short reason. Judge the PATTERN, not specific phrases.

**broken_line** — a line where the LAST word is obviously forced to hit a letter:
- Line ends on a bare interjection (`hi`, `hello`, `oh`, `ah`, `yeah`) used as filler. EXCEPTION: do NOT fire on quoted speech like `say hello` or `greeted with 'hi'`.
- Line ends on a tiny grammatical word (`a`, `the`, `for`, `in`, `to`, `we`) leaving the sentence incomplete.
- Line ends on a content word that couldn't plausibly appear in the scene the rest of the line sets up. Bar is HIGH.

**nonsense_line** — a line is grammatical but there is no coherent reading; you would need to invent context to parse it.

**repetition** — two lines are identical or near-identical, OR the same word appears as the final word of two different lines.

**template_locked** — most lines share an obvious syntactic template, making the poem feel like a fill-in-the-blank exercise rather than authored verse. Fire if the SAME pattern repeats on 3+ lines (out of however many the poem has):
- Same opening word/phrase ("The X...", "We X...", "My X...", "She X...")
- Same simile pattern ("X like a Y", "as X as Y") on 3+ lines
- Identical subject-verb-object scaffolding with only the final word swapped (e.g. "The mountain peaks are sharp like a ski / The ancient forest chimes like a bell / The morning sun warms the green tea")
The bar is HIGH — a single repeated opener like one stray "And..." is fine. Fire only when the pattern is OBVIOUS and PERVASIVE — the kind of thing a reader notices in one glance. Do NOT fire just because a few lines share a subject (e.g. two lines starting with "The"); you need 3+ lines locked into the same scaffolding.

**prompt_alignment_fail** — the poem does NOT do what the prompt asks. Check all three:
- Theme: does the poem actually touch the subject the prompt names (mortality, scam, graduation, war, etc.)? If entirely absent, fire.
- Narrator: if the prompt specifies a voice (child, corporate executive, archaic bard), does the speaker stay consistent? A child-narrator who mentions their mortgage = fire.
- Subtext: if the prompt says the hidden word / theme should stay hidden, does the poem lexically state the hidden concept? Target `tyrant` + poem says `autocrat` = fire.

### Part 2 — Axes (1 to 5, higher is better)

- **specificity**: 1 = abstract platitudes, 5 = many named concrete objects and actions
- **coherence**: 1 = lines feel interchangeable between unrelated poems, 5 = reads as one deliberate poem

### Output (strict JSON, no markdown)

{{
  "problems": {{
    "broken_line": [],
    "nonsense_line": [],
    "repetition": [],
    "template_locked": [],
    "prompt_alignment_fail": []
  }},
  "axes": {{
    "specificity": 1-5,
    "coherence": 1-5
  }},
  "rationale": "one-sentence summary"
}}"""


# ── Judge scoring (deterministic; separate from LLM output) ──
_FAILURE_PENALTIES = {
    "broken_line": 0.25,
    "nonsense_line": 0.25,
    "repetition": 0.30,
    "template_locked": 0.30,
    "prompt_alignment_fail": 0.35,
}
_AXIS_BASE = 0.30
_AXIS_RANGE = 0.60  # 5/5 → 0.30 + (10/10)*0.60 = 0.90
_AXIS_MAX_SUM = 10  # two axes × max 5
_SCORE_FLOOR = 0.10
_SCORE_CEIL = 1.00


def _score_judge_output(problems: dict, axes: dict) -> dict:
    """Compute (score, verdict, breakdown) from v7 judge JSON fields.

    problems: dict of 4 categories (broken_line, nonsense_line, repetition,
              prompt_alignment_fail) → list of strings (empty if none).
    axes: dict with specificity and coherence (1-5 each).
    """
    ax_sum = int(axes.get("specificity", 3) or 3) + int(axes.get("coherence", 3) or 3)
    ax_sum = max(2, min(_AXIS_MAX_SUM, ax_sum))
    axis_contribution = _AXIS_BASE + (ax_sum / _AXIS_MAX_SUM) * _AXIS_RANGE

    per_cat = {}
    total_penalty = 0.0
    for cat, weight in _FAILURE_PENALTIES.items():
        hits = problems.get(cat) or []
        n = 1 if hits else 0  # per-category cap = 1 to prevent over-penalizing
        per_cat[cat] = n * weight
        total_penalty += n * weight

    raw = axis_contribution - total_penalty
    score = max(0.10, min(1.00, raw))

    if score >= 0.80:
        verdict = "GOOD"
    elif score >= 0.40:
        verdict = "OK"
    else:
        verdict = "BAD"

    return {
        "score": round(score, 3),
        "verdict": verdict,
        "axis_sum": ax_sum,
        "axis_contribution": round(axis_contribution, 3),
        "total_penalty": round(total_penalty, 3),
        "per_category_penalty": per_cat,
    }


# ── Poem parsing helpers (line splitting, letter/word extraction) ──
_TRAILING_PUNCT = re.compile(r'[\s.!?,;:"\')}\]\-—…。！？，；：""' "》）】\u3000]+$")


def _detect_language(target_word):
    for ch in target_word:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _parse_poem_lines(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(title|poem|verse)\s*:", line, re.IGNORECASE):
            continue
        if re.match(r"^[-=]{3,}$", line):
            continue
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            lines.append(line)
    return lines


def _get_last_char(line, language):
    stripped = _TRAILING_PUNCT.sub("", line)
    if not stripped:
        return ""
    if language == "zh":
        return stripped[-1]
    for ch in reversed(stripped):
        if ch.isalpha():
            return ch.lower()
    return ""


def _get_last_word(line):
    stripped = _TRAILING_PUNCT.sub("", line)
    words = stripped.split()
    if not words:
        return ""
    return re.sub(r"^['\"\(\[-]+|['\"\)\]-]+$", "", words[-1]).lower()


def _contains_hidden_word(poem_text: str, target_word: str, language: str) -> bool:
    if not target_word or not poem_text:
        return False
    if language == "zh":
        return target_word in poem_text
    return (
        re.search(rf"\b{re.escape(target_word)}\b", poem_text, re.IGNORECASE)
        is not None
    )


def _extract_answer_block(text):
    stripped = text.rstrip()
    close = "</answer>"
    if not stripped.endswith(close):
        return None
    inner = stripped[: -len(close)]
    idx = inner.rfind("<answer>")
    if idx == -1:
        return None
    return inner[idx + len("<answer>") :].strip()


# ── Group-level similarity (cross-rollout divisors) ──
def _cluster_similarity_divisors(
    texts: List[str], threshold: float = 0.85
) -> List[float]:
    """Cluster texts by string similarity; return each text's cluster size as its divisor."""
    cluster_ids = [-1] * len(texts)
    cluster_reps: List[str] = []
    for i, text in enumerate(texts):
        for cid, rep in enumerate(cluster_reps):
            if SequenceMatcher(None, text, rep).ratio() > threshold:
                cluster_ids[i] = cid
                break
        else:
            cluster_ids[i] = len(cluster_reps)
            cluster_reps.append(text)
    counts = Counter(cluster_ids)
    return [float(counts[cid]) for cid in cluster_ids]


# ── Judge output logging ──
def _log_judge_breakdown(rollout_id: str, judge: dict) -> None:
    """Emit a multi-line log for v7 judge output."""
    verdict = judge.get("verdict", "?")
    score = judge.get("score", 0.0)
    penalty = judge.get("total_penalty", 0.0)
    axes = judge.get("axes", {}) or {}
    s = axes.get("specificity", "?")
    c = axes.get("coherence", "?")
    logger.info(f"[TelestichEnv] judge: verdict={verdict} score={score:.2f} "
        f"penalty={penalty:.2f} axes(s/c)={s}/{c}",
    )
    rationale = str(judge.get("rationale", "")).strip()
    if rationale:
        logger.info(f"  rationale: {rationale[:240]}")
    for cat, hits in (judge.get("problems") or {}).items():
        if not hits:
            continue
        logger.info(f"  {cat}:")
        for h in hits:
            logger.info(f"    {str(h)[:240]}")


_MARKUP_WRAPPER_RE = re.compile(r"^[\*_`\\{}]+|[\*_`\\{}]+$")


def _ending_sequence(completion_text: str) -> str:
    """Whitespace-joined sequence of per-line ending tokens.

    English: last word of each line, stripped of markdown/LaTeX wrappers.
    Mandarin: last character of each line.

    Used by compute_group_reward to detect rollouts that share the same
    ending-word pattern across siblings (template mode-collapse within a
    group). SequenceMatcher on these joined strings gives ratio=1.0 for
    identical sequences and decays smoothly for partial overlap.
    """
    poem = _extract_answer_block(completion_text)
    if poem is None:
        poem = completion_text
    lines = _parse_poem_lines(poem)
    if not lines:
        return ""
    language = _detect_language(poem)
    tokens: list[str] = []
    for line in lines:
        if language == "zh":
            ch = _get_last_char(line, "zh")
            if ch:
                tokens.append(ch)
        else:
            w = _MARKUP_WRAPPER_RE.sub("", _get_last_word(line))
            if w:
                tokens.append(w)
    return " ".join(tokens)


_CJK_RE = re.compile(r"[一-鿿]")


def _english_rhyme_density(lines: list[str]) -> tuple[float | None, dict]:
    """Max-cluster rhyme score: largest set of mutually-rhyming line endings,
    divided by scoreable line count. Pushes the poem toward a committed
    monorhyme rather than scattered pairs.

    Identical-word "rhymes" (same ending word repeated) are excluded.
    """
    endings: list[str] = []
    for line in lines:
        w = _MARKUP_WRAPPER_RE.sub("", _get_last_word(line))
        endings.append(w)
    scoreable_idx: list[int] = []
    parts_by_idx: dict[int, list[str]] = {}
    for i, w in enumerate(endings):
        if not w:
            continue
        phones = pronouncing.phones_for_word(w)
        if not phones:
            continue
        parts_by_idx[i] = [pronouncing.rhyming_part(p) for p in phones]
        scoreable_idx.append(i)
    if len(scoreable_idx) < 2:
        return None, {"endings": endings, "clusters": [], "scoreable": len(scoreable_idx)}

    # Build mutual-rhyme adjacency (skip identical-word pairs).
    adj: dict[int, set[int]] = {i: set() for i in scoreable_idx}
    for a in range(len(scoreable_idx)):
        for b in range(a + 1, len(scoreable_idx)):
            i, j = scoreable_idx[a], scoreable_idx[b]
            if endings[i] == endings[j]:
                continue
            if any(x == y for x in parts_by_idx[i] for y in parts_by_idx[j]):
                adj[i].add(j)
                adj[j].add(i)

    # Connected components on the rhyme graph.
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for i in scoreable_idx:
        if i in visited:
            continue
        stack = [i]
        comp: list[int] = []
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.append(x)
            stack.extend(adj[x])
        clusters.append(sorted(comp))

    # Score = size of the largest cluster with ≥2 members / scoreable.
    cluster_sizes = [len(c) for c in clusters if len(c) >= 2]
    largest = max(cluster_sizes) if cluster_sizes else 0
    score = largest / len(scoreable_idx)
    cluster_words = [
        [endings[i] for i in c] for c in clusters if len(c) >= 2
    ]
    return score, {
        "endings": endings,
        "clusters": cluster_words,
        "largest": largest,
        "scoreable": len(scoreable_idx),
    }


def _mandarin_length_uniformity(lines: list[str]) -> tuple[float | None, dict]:
    lengths = [len(_CJK_RE.findall(line)) for line in lines]
    nonzero = [n for n in lengths if n > 0]
    if len(nonzero) < 2:
        return None, {"lengths": lengths, "modal": None}
    counts = Counter(nonzero)
    modal, _ = counts.most_common(1)[0]
    matching = sum(1 for n in lengths if n == modal)
    score = matching / len(nonzero)
    return score, {"lengths": lengths, "modal": modal}


def score_rhyme(lines: list[str], language: str) -> tuple[float | None, dict]:
    """Form-quality score in [0, 1].

    English: fraction of line endings that rhyme (CMU perfect rhyme) with >=1
    other line. OOV lines excluded from numerator and denominator.
    Mandarin: fraction of lines whose CJK char-count equals the modal length.
    Returns (None, info) when fewer than 2 scoreable lines.
    """
    if language == "zh":
        return _mandarin_length_uniformity(lines)
    return _english_rhyme_density(lines)


_TOOL_CALL_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def _count_tool_calls(completion: Any) -> int:
    """Count `<tool_call>` occurrences in the completion text.

    Counts all attempts, including those rejected for exceeding the cap —
    extras are exactly the inefficiency we want to penalize.
    """
    text = extract_completion_text(completion)
    return len(_TOOL_CALL_RE.findall(text or ""))


# ── TelestichEnv ──
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

Process:
1. Spell out the target word letter by letter (or character by character). \
Count — that's how many lines.
2. You have 2 tool calls total. Use word_bank for letters where \
you struggle to think of ending words — prioritize the hardest letters. \
(For Chinese poems, the tool is not available — rely on your own vocabulary.)
3. Pick one ending word per line. Build each line as a natural phrase around \
that ending word.
4. Verify each ending word's last letter/character, confirm they spell the target.
5. Output the poem in <answer></answer> tags. Plain text only. Stop after \
</answer>.

Be concise — total output length is penalized, so don't ramble. Use your \
tool calls wisely on the letters you find hardest.

Rules:
1. Exactly as many lines as letters/characters in the target word.
2. Every ending word must be a real word — no invented words, no standalone \
letters tacked on.
3. Each line is a complete, meaningful phrase. The poem should be coherent \
and match the requested theme.
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
R, A, E are easy. B and V are the hardest — use both tool calls on those.

<tool_call>
{"name": "word_bank", "arguments": {"letter": "b"}}
</tool_call>
<tool_call>
{"name": "word_bank", "arguments": {"letter": "v"}}
</tool_call>

B: throb, club, bob, web, absorb, bomb, climb, sub, rob, ...
(1 tool call remaining)
V: tv, iv, gov, hiv, dev, mtv, suv, rev, improv, ...
(0 tool calls remaining)

Picking: throb, remember, idea, tv, promise
throb→B, remember→R, idea→A, tv→V, promise→E = BRAVE ✓

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

    def __init__(
        self,
        *,
        judge_base_url: str,
        judge_api_key: str,
        judge_timeout: float = 60.0,
        w_quality: float = 1.0,
        w_conciseness: float = 0.3,
        w_rhyme_en: float = 0.5,
        w_rhyme_zh: float = 0.15,
        winner_bar: float = 0.80,
        winner_eps: float = 0.15,
        **kwargs: Any,
    ) -> None:
        self._judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=judge_api_key,
            max_retries=3,
        )
        self._judge_timeout = judge_timeout
        self._w_quality = w_quality
        self._w_conciseness = w_conciseness
        self._w_rhyme_en = w_rhyme_en
        self._w_rhyme_zh = w_rhyme_zh
        self._winner_bar = winner_bar
        self._winner_eps = winner_eps
        self._tool_calls: dict[str, int] = {}
        self._max_tool_calls = 2

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        prompt_text = example.get("prompt", "")
        return make_example(
            prompt_messages=[{"role": "user", "content": prompt_text}],
            task={
                "prompt": prompt_text,
                "ground_truth": example.get("ground_truth", ""),
            },
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="word_bank",
                description="Returns 30 random common English words that end with the given letter (a-z only; not available for Chinese characters). You have 2 tool calls total — use them for letters where you struggle to think of ending words.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "letter": {
                            "type": "string",
                            "description": "A single letter (a-z)",
                        },
                    },
                    "required": ["letter"],
                },
            ),
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        if tool_name == "word_bank":
            count = self._tool_calls.get(rollout_id, 0)
            if count >= self._max_tool_calls:
                return "No tool calls remaining."
            self._tool_calls[rollout_id] = count + 1
            remaining = self._max_tool_calls - (count + 1)

            letter = tool_args.get("letter", "")
            words = _word_bank(letter)
            if not words:
                result = f"No common words found ending in '{letter}'."
            else:
                result = ", ".join(words)
            result += (
                f"\n({remaining} tool call{'s' if remaining != 1 else ''} remaining)"
            )
            return result
        return f"Error: Tool '{tool_name}' not found."

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        # All reward logic lives in _compute_group_reward (so we can divide by
        # the cluster-duplication divisor). Return empty so the trainer doesn't
        # double-write quality/conciseness keys.
        return {}

    async def compute_group_reward(
        self,
        rollout_ids: list[str],
        messages_list: list[Messages],
        tasks: list[dict[str, Any] | None],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        # Logs emitted from this body fan out to every rid in the group by
        # default (env_service wraps the call in ``group_context``). Per-rid
        # log lines are wrapped in ``rollout_context(rid)`` below so each
        # rollout's env_log only sees its own efficiency/divisor line, not
        # the other N-1 rollouts'.
        n = len(rollout_ids)

        # Per-rollout rewards, computed concurrently.
        per_rewards = await asyncio.gather(
            *[
                self._compute_single_reward(rid, msgs, t, **kwargs)
                for rid, msgs, t in zip(rollout_ids, messages_list, tasks)
            ]
        )

        # Text-similarity divisor (full-poem near-duplicates).
        cluster_texts: list[str] = []
        ending_texts: list[str] = []
        languages: list[str] = []
        for messages in messages_list:
            text = extract_completion_text(messages)
            poem_for_lang = _extract_answer_block(text) or text
            cluster_texts.append(poem_for_lang)
            ending_texts.append(_ending_sequence(text))
            languages.append(_detect_language(poem_for_lang))
        text_divisors = _cluster_similarity_divisors(cluster_texts, threshold=0.85)
        # Ending-word divisor (sibling rollouts that share the same
        # per-line ending token sequence — template mode-collapse).
        # Skipped for Mandarin: the acrostic forces every correct rollout
        # to share the same ending characters, so this divisor would
        # mechanically punish correctness.
        ending_divisors_raw = _cluster_similarity_divisors(ending_texts, threshold=0.85)
        ending_divisors = [
            1.0 if lang == "zh" else d
            for lang, d in zip(languages, ending_divisors_raw)
        ]
        # Combine by max — a rollout is "duplicated" by whichever signal
        # fires more strongly. Using max instead of multiplying avoids
        # double-penalizing rollouts that happen to collide on both.
        divisors = [max(t, e) for t, e in zip(text_divisors, ending_divisors)]

        # Group-relative efficiency bonus (length + tool calls), anchored
        # on the winners' averages within this group. Non-winners get 0
        # so sub-standard-but-short rollouts aren't rewarded.
        raw_q = [
            (r.get("quality", 0.0) / self._w_quality) if self._w_quality else 0.0
            for r in per_rewards
        ]
        lengths = [len(extract_completion_text(m)) for m in messages_list]
        tool_counts = [_count_tool_calls(m) for m in messages_list]
        max_q = max(raw_q) if raw_q else 0.0
        bar = max(self._winner_bar, max_q - self._winner_eps)
        winners = [i for i, q in enumerate(raw_q) if q >= bar]

        efficiencies = [0.0] * n
        if winners:
            L_anchor = sum(lengths[i] for i in winners) / len(winners)
            C_anchor = sum(tool_counts[i] for i in winners) / len(winners)
            for i in winners:
                len_ineff = (
                    1 - math.exp(-max(0.0, lengths[i] / L_anchor - 1.0))
                    if L_anchor > 0 else 0.0
                )
                if C_anchor > 0:
                    call_ineff = 1 - math.exp(
                        -max(0.0, tool_counts[i] / C_anchor - 1.0)
                    )
                else:
                    # All winners used zero tool calls — any use by this
                    # winner is pure excess.
                    call_ineff = 1.0 if tool_counts[i] > 0 else 0.0
                efficiencies[i] = 1.0 - 0.5 * (len_ineff + call_ineff)
            # Group-level summary: fans out to every rid in the group.
            logger.info(f"[TelestichEnv] group efficiency: bar={bar:.3f} "
                f"winners={len(winners)}/{n} L_anchor={L_anchor:.0f} "
                f"C_anchor={C_anchor:.2f}",
            )

        adjusted: list[dict[str, float]] = []
        for i, (rid, rewards, div, t_div, e_div) in enumerate(
            zip(rollout_ids, per_rewards, divisors, text_divisors, ending_divisors)
        ):
            rewards = dict(rewards)
            rewards["conciseness"] = self._w_conciseness * efficiencies[i]
            with rollout_context(rid):
                logger.info(f"[TelestichEnv] efficiency={efficiencies[i]:.3f} "
                    f"(len={lengths[i]} calls={tool_counts[i]} "
                    f"winner={i in winners})",
                )
                if div > 1.0:
                    logger.info(f"[TelestichEnv] duplication divisor={div} "
                        f"(text={t_div}, ending={e_div}) applied to {rewards}",
                    )
            adjusted.append({k: v / div for k, v in rewards.items()})
        return adjusted

    async def _compute_single_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, float]:
        zeros = {"quality": 0.0, "conciseness": 0.0, "rhyme": 0.0}
        try:
            # Extract only the final assistant message for answer extraction
            if isinstance(messages, list):
                final_text = ""
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, str) and content.strip():
                            final_text = content
                            break
            else:
                final_text = extract_completion_text(messages)
            if not final_text.strip():
                return zeros

            t = task or {}
            prompt = str(t.get("prompt", ""))
            poem_text = _extract_answer_block(final_text)
            if poem_text is None:
                logger.info("[TelestichEnv] No <answer> block found")
                return zeros
            lines = _parse_poem_lines(poem_text)

            # 1. Get target word from task.ground_truth
            target_word = str(t.get("ground_truth") or "").strip()
            if not target_word:
                return zeros

            language = _detect_language(target_word)

            # 2. Correctness: letter match + valid word combined
            target_chars = (
                list(target_word.lower()) if language == "en" else list(target_word)
            )
            n = len(target_chars)
            if not lines or n == 0:
                correctness = 0.0
            else:
                compare_n = min(len(lines), n)
                misses = 0
                for i in range(compare_n):
                    letter_ok = _get_last_char(lines[i], language) == target_chars[i]
                    word_ok = language == "zh" or _is_valid_word(
                        _get_last_word(lines[i])
                    )
                    if not (letter_ok and word_ok):
                        misses += 1
                misses += max(0, n - len(lines))
                misses += max(0, len(lines) - n)

                if misses == 0:
                    correctness = 1.0
                elif misses >= n:
                    correctness = 0.0
                else:
                    correctness = 0.5 * (n - misses) / (n - 1) if n > 1 else 0.0

            # 3. Judge quality (Strategy 1: per-line categories + whole-poem
            # axes → score in [0, 1]). Only runs if correctness >= 0.5.
            if correctness >= 0.5:
                judge = await self._judge_quality(prompt, poem_text)
                judge_score = judge["score"]
                _log_judge_breakdown(rollout_id, judge)
            else:
                judge = {"score": 0.0, "reasoning": "correctness below gate"}
                judge_score = 0.0
                logger.info(f"[TelestichEnv] judge: skipped (correctness={correctness:.3f} < 0.5)",
                )

            # 4. Quality: multiply correctness and judge score
            quality = correctness * judge_score
            logger.info(f"[TelestichEnv] quality breakdown: correctness={correctness:.3f} × "
                f"judge_score={judge_score:.3f} → quality={quality:.3f}",
            )

            # 5. Cheating penalty: hidden word appearing in the poem body is
            # constraint gaming (the poem just writes the target word out),
            # not a craft flaw. Zero the quality entirely.
            cheated = _contains_hidden_word(poem_text, target_word, language)
            if cheated:
                quality = 0.0
                logger.info(f"[TelestichEnv] cheating penalty: hidden word '{target_word}' in poem body → quality=0",
                )

            # 6. Rhyme/form bonus, hard-gated on perfect acrostic + no cheat.
            # English: CMU perfect-rhyme density. Mandarin: char-count uniformity.
            rhyme = 0.0
            if correctness == 1.0 and not cheated:
                rhyme_raw, rhyme_info = score_rhyme(lines, language)
                if rhyme_raw is not None:
                    w_rhyme = (
                        self._w_rhyme_en if language == "en" else self._w_rhyme_zh
                    )
                    rhyme = w_rhyme * rhyme_raw
                    logger.info(f"[TelestichEnv] rhyme: lang={language} raw={rhyme_raw:.3f} "
                        f"weighted={rhyme:.3f} info={str(rhyme_info)[:240]}",
                    )
                else:
                    logger.info("[TelestichEnv] rhyme: skipped (too few scoreable lines)",
                    )
            else:
                logger.info(f"[TelestichEnv] rhyme: skipped "
                    f"(correctness={correctness:.3f}, cheated={cheated})",
                )

            # 7. Conciseness is computed at the group level (see
            # compute_group_reward) so it can anchor on winner lengths /
            # tool-call counts within the same GRPO group.
            rewards = {
                "quality": self._w_quality * clip01(quality),
                "conciseness": 0.0,
                "rhyme": rhyme,
            }
            logger.info(f"[TelestichEnv] per-rollout rewards={rewards} (conciseness filled at group level)")
            return rewards

        except Exception as e:
            logger.info(f"[TelestichEnv] compute_reward error: {e}")
            print(f"[TelestichEnv] compute_reward error: {e}")
            return zeros

    async def _judge_quality(self, prompt: str, poem_text: str) -> dict:
        """Judge poem quality via v7 rubric: LLM classifies problems + rates axes;
        Python deterministically computes the score.

        Returns dict:
          score: float in [0.1, 1.0] — final quality reward
          verdict: 'GOOD' | 'OK' | 'BAD'
          problems: dict of 4 categories (broken_line, nonsense_line, repetition,
                    prompt_alignment_fail) → list of hit strings
          axes: dict of specificity / coherence (1–5 each)
          rationale, total_penalty, axis_contribution, per_category_penalty
        """
        judge_prompt = QUALITY_JUDGE_PROMPT.format(prompt=prompt, poem_text=poem_text)
        zeros = {
            "score": 0.0,
            "verdict": "BAD",
            "problems": {k: [] for k in _FAILURE_PENALTIES},
            "axes": {"specificity": 0, "coherence": 0},
            "rationale": "",
            "total_penalty": 0.0,
            "axis_contribution": 0.0,
            "per_category_penalty": {},
        }
        try:
            resp = await self._judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                timeout=self._judge_timeout,
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content if resp.choices else None
            content = (raw or "").strip()
            if not content:
                return zeros
            content = re.sub(
                r"^```(?:json)?|```$", "", content, flags=re.MULTILINE
            ).strip()

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if not m:
                    return zeros
                try:
                    result = json.loads(m.group())
                except Exception:
                    return zeros

            problems = result.get("problems") or {}
            axes = result.get("axes") or {}
            rationale = str(result.get("rationale", "")).strip()

            scored = _score_judge_output(problems, axes)

            return {
                "score": scored["score"],
                "verdict": scored["verdict"],
                "problems": {k: problems.get(k) or [] for k in _FAILURE_PENALTIES},
                "axes": {
                    "specificity": int(axes.get("specificity", 3) or 3),
                    "coherence": int(axes.get("coherence", 3) or 3),
                },
                "rationale": rationale,
                "total_penalty": scored["total_penalty"],
                "axis_contribution": scored["axis_contribution"],
                "per_category_penalty": scored["per_category_penalty"],
            }

        except Exception as e:
            print(f"[TelestichEnv] _judge_quality error: {e}")
            return zeros
