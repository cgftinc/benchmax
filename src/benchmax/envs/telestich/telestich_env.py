"""TelestichEnv — reward env for telestich (last-letter acrostic) poems.

A telestich hides a target word in the last letter (or Chinese character) of
each line. Reward, per rollout in a GRPO group (see ``compute_group_reward``):

  1. HARD RULES (deterministic, no LLM): the poem must be a valid telestich —
     acrostic spells the target, every line ends on a real word, right line
     count — and must NOT write the hidden word in the body. Fail → reward 0,
     no judge called.
  2. QUALITY (one gpt-listwise call per group): gpt ranks the group's valid
     poems together with the example's `acceptable` + `great` reference poems.
     A poem's rank relative to those two anchors maps to a banded score:
        below-acceptable [.05,.30] | mid [.30,.80] | above-great [.80,1.0].
  3. ADJUSTMENTS (deterministic): discount reused ending words (anti
     mode-collapse), add a rhyme/length form bonus, and add a winner-anchored
     conciseness bonus (only top performers; shorter/fewer-tool-calls scores
     higher) — all logged as components.

The final reward = quality + form + conciseness − reuse_penalty
(the component values sum to it).
"""

import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

import pronouncing
from english_words import get_english_words_set
from openai import AsyncOpenAI
from wordfreq import word_frequency

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.logging import rollout_context
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
JUDGE_MODEL = "gpt-5.4"
JUDGE_MAX_TOKENS = 2500

# Score bands (non-overlapping, ascending). A poem's gpt-listwise rank relative
# to the acceptable/great anchors places it in one band; position within the
# band is set by the rank.
BAND_BELOW = (0.05, 0.30)   # ranked below the acceptable anchor
BAND_MID = (0.30, 0.80)     # between acceptable and great anchors
BAND_ABOVE = (0.80, 1.00)   # above the great anchor

W_REUSE = 0.50   # weight of the ending-word reuse discount
W_FORM = 0.15    # weight of the rhyme/length form bonus
W_CONCISE = 0.15   # conciseness bonus — small + quality-scaled, so it only breaks
                   # ties among similar-quality top performers, never lifts a weak poem
WINNER_BAR = 0.50  # min band-quality to count as a "top performer"
WINNER_EPS = 0.15  # ...and within this of the group's best quality

_TOOL_CALL_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def _count_tool_calls(text: str) -> int:
    return len(_TOOL_CALL_RE.findall(text or ""))

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
def check_hard_rules(poem: str | None, target: str) -> dict:
    """Deterministic gate. correct == True iff valid telestich AND not cheating."""
    target = (target or "").strip()
    language = detect_language(target)
    lines = parse_poem_lines(poem or "")
    chars = list(target.lower()) if language == "en" else list(target)
    n = len(chars)

    cheated = contains_hidden_word(poem or "", target, language)
    if not lines or n == 0 or len(lines) != n:
        return {"correct": False, "cheated": cheated, "language": language, "lines": lines}
    for i in range(n):
        letter_ok = last_char(lines[i], language) == chars[i]
        word_ok = language == "zh" or is_valid_word(last_word(lines[i]))
        if not (letter_ok and word_ok):
            return {"correct": False, "cheated": cheated, "language": language, "lines": lines}
    return {"correct": not cheated, "cheated": cheated, "language": language, "lines": lines}


# ══════════════════════════════════════════════════════════════════════
# FORM  (rhyme density for EN, line-length uniformity for ZH)
# ══════════════════════════════════════════════════════════════════════
def _english_rhyme_density(lines: list[str]) -> float:
    endings = [_MARKUP_WRAPPER_RE.sub("", last_word(line)) for line in lines]
    parts, idx = {}, []
    for i, w in enumerate(endings):
        phones = pronouncing.phones_for_word(w) if w else []
        if phones:
            parts[i] = [pronouncing.rhyming_part(p) for p in phones]
            idx.append(i)
    if len(idx) < 2:
        return 0.0
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
        clusters.append(comp)
    largest = max((len(c) for c in clusters if len(c) >= 2), default=0)
    return largest / len(idx)


def _mandarin_length_uniformity(lines: list[str]) -> float:
    lengths = [len(_CJK_RE.findall(line)) for line in lines]
    nz = [n for n in lengths if n > 0]
    if len(nz) < 2:
        return 0.0
    modal = Counter(nz).most_common(1)[0][0]
    return sum(1 for n in lengths if n == modal) / len(nz)


def form_score(lines: list[str], language: str) -> float:
    return _mandarin_length_uniformity(lines) if language == "zh" else _english_rhyme_density(lines)


# ══════════════════════════════════════════════════════════════════════
# DIVERSITY
# ══════════════════════════════════════════════════════════════════════
def ending_reuse_scores(poems: list[str]) -> list[float]:
    """Per-poem reuse score in [0,1]: mean over the poem's ending words of the
    fraction of *other* poems that also use that word. 0 = all endings unique
    to this poem; 1 = every ending it uses is shared by every sibling.
    Discourages the whole group leaning on the same crutch words (ski/hi/free).
    """
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
    scores = []
    for ews in endings:
        if not ews or n < 2:
            scores.append(0.0)
        else:
            scores.append(sum((doc_freq[w] - 1) / (n - 1) for w in ews) / len(ews))
    return scores


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
# JUDGE  (gpt-listwise over the group + reference anchors -> banded score)
# ══════════════════════════════════════════════════════════════════════
_LISTWISE_PROMPT = """\
A user requested a poem:
{prompt}

Here are {n} candidate poems (all already valid telestichs — judge ONLY poetic
quality and faithfulness to the request, not the acrostic):
{block}

Rank them best-to-worst on coherence, concrete imagery, theme/voice match, and
natural (un-forced) line endings. Output strict JSON only:
{{"order": ["<id best>", ..., "<id worst>"]}}"""


def _parse_order(raw: str, ids: list[str]) -> list[str]:
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    raw_order = []
    if m:
        try:
            raw_order = [x for x in (json.loads(m.group()).get("order") or []) if x in ids]
        except Exception:
            raw_order = []
    seen, order = set(), []
    for x in raw_order:  # dedup (judge may repeat an id), keep first occurrence
        if x not in seen:
            seen.add(x)
            order.append(x)
    return order + [i for i in ids if i not in seen]  # append any missing


def bands_from_ranking(order: list[str], acc_ids: list[str], great_ids: list[str],
                       cand_ids: list[str]) -> dict[str, float]:
    """Map each candidate's listwise rank to a banded score, using the mean
    rank-position of the acceptable / great anchors as band boundaries."""
    n = len(order)
    q = {pid: 1 - i / (n - 1) for i, pid in enumerate(order)} if n > 1 else {pid: 0.5 for pid in order}
    qa = sum(q[i] for i in acc_ids) / len(acc_ids) if acc_ids else 0.40
    qg = sum(q[i] for i in great_ids) / len(great_ids) if great_ids else 0.80
    # Keep anchors ordered and off the extremes so all three bands stay
    # reachable even if the judge ranks an anchor at the very top/bottom.
    qa = min(qa, 0.80)
    qg = min(max(qg, qa + 0.10), 0.95)
    out = {}
    for c in cand_ids:
        qc = q[c]
        if qc >= qg:
            f = (qc - qg) / (1 - qg) if qg < 1 else 1.0
            lo, hi = BAND_ABOVE
        elif qc >= qa:
            f = (qc - qa) / (qg - qa)
            lo, hi = BAND_MID
        else:
            f = qc / qa if qa > 0 else 0.0
            lo, hi = BAND_BELOW
        out[c] = round(lo + (hi - lo) * max(0.0, min(1.0, f)), 4)
    return out


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

Be concise — total output length is penalized among top poems, so don't \
ramble. Use your tool calls only on the letters you find hardest.

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

    def __init__(self, *, judge_base_url: str, judge_api_key: str,
                 judge_timeout: float = 90.0, max_tool_calls: int = 2,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._judge = AsyncOpenAI(base_url=judge_base_url, api_key=judge_api_key, max_retries=3)
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

    async def _listwise_quality(self, prompt, poems, acc_refs, great_refs) -> dict[int, float]:
        """One gpt call: rank poems + anchors, map to banded quality per poem idx."""
        items = [(f"c{i}", p) for i, p in enumerate(poems)]
        acc_ids = [f"a{i}" for i in range(len(acc_refs))]
        great_ids = [f"g{i}" for i in range(len(great_refs))]
        items += list(zip(acc_ids, acc_refs)) + list(zip(great_ids, great_refs))
        random.shuffle(items)
        block = "\n\n".join(f"[{pid}]\n{poem}" for pid, poem in items)
        ids = [pid for pid, _ in items]
        try:
            resp = await self._judge.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": _LISTWISE_PROMPT.format(
                    prompt=prompt, n=len(items), block=block)}],
                temperature=0, max_tokens=JUDGE_MAX_TOKENS, timeout=self._judge_timeout,
            )
        except Exception as e:
            # On judge failure give every valid poem the SAME neutral quality
            # (no free high reward, no shuffle-derived noise). The group's
            # gradient that step comes from reuse/form only.
            logger.info(f"[TelestichEnv] judge error: {e}; neutral fallback")
            return {i: BAND_BELOW[1] for i in range(len(poems))}
        order = _parse_order(resp.choices[0].message.content or "", ids)
        cand_ids = [f"c{i}" for i in range(len(poems))]
        banded = bands_from_ranking(order, acc_ids, great_ids, cand_ids)
        return {i: banded[f"c{i}"] for i in range(len(poems))}

    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        task = tasks[0] or {}
        target = str(task.get("ground_truth") or "")
        prompt = str(task.get("prompt") or "")
        acc_refs = list(task.get("acceptable_refs") or [])
        great_refs = list(task.get("great_refs") or [])
        n = len(rollout_ids)

        # ── Stage 1: hard rules (deterministic) ──
        poems, gate, form = [], [], []
        for messages in messages_list:
            poem = final_poem(messages) or ""
            chk = check_hard_rules(poem, target)
            poems.append(poem)
            gate.append(chk["correct"])
            form.append(form_score(chk["lines"], chk["language"]) if chk["correct"] else 0.0)

        valid = [i for i in range(n) if gate[i]]

        # ── Stage 2: quality via one gpt-listwise call over valid poems + anchors ──
        quality = [0.0] * n
        if valid:
            q_by_local = await self._listwise_quality(
                prompt, [poems[i] for i in valid], acc_refs, great_refs)
            for local, i in enumerate(valid):
                quality[i] = q_by_local[local]

        # ── Stage 3: adjustments (deterministic) ──
        reuse = [0.0] * n
        if valid:
            r = ending_reuse_scores([poems[i] for i in valid])
            for local, i in enumerate(valid):
                reuse[i] = r[local]
        dup = [1.0] * n
        if valid:
            d = duplicate_divisors([poems[i] for i in valid])
            for local, i in enumerate(valid):
                dup[i] = d[local]

        # Winner-anchored conciseness: only "top performers" — within WINNER_EPS
        # of the group's best quality AND above WINNER_BAR — earn it; among them,
        # shorter output / fewer tool calls scores higher. The threshold stops a
        # short degenerate poem from gaming length (non-winners get 0).
        lengths = [len(extract_completion_text(m) or "") for m in messages_list]
        tool_counts = [_count_tool_calls(extract_completion_text(m)) for m in messages_list]
        qmax = max((quality[i] for i in valid), default=0.0)
        bar = max(WINNER_BAR, qmax - WINNER_EPS)
        winners = [i for i in valid if quality[i] >= bar]
        conciseness = [0.0] * n
        if winners:
            len_anchor = sum(lengths[i] for i in winners) / len(winners)
            call_anchor = sum(tool_counts[i] for i in winners) / len(winners)
            for i in winners:
                len_ineff = 1 - math.exp(-max(0.0, lengths[i] / len_anchor - 1.0)) if len_anchor else 0.0
                if call_anchor:
                    call_ineff = 1 - math.exp(-max(0.0, tool_counts[i] / call_anchor - 1.0))
                else:
                    call_ineff = 1.0 if tool_counts[i] else 0.0
                # scaled by quality so a weak "winner" earns little — keeps it a
                # tiebreaker among genuinely good poems, not a lever to lift junk
                conciseness[i] = W_CONCISE * quality[i] * (1.0 - 0.5 * (len_ineff + call_ineff))

        out = []
        for i in range(n):
            if not gate[i]:
                out.append({"quality": 0.0, "reuse_penalty": 0.0, "form": 0.0, "conciseness": 0.0})
                continue
            q = quality[i] / dup[i]                       # near-duplicate whole poems shared down
            reuse_pen = W_REUSE * reuse[i] * q            # discount reused ending words
            form_bonus = W_FORM * form[i]                 # rhyme / length bonus
            comp = {"quality": round(q, 4),
                    "reuse_penalty": round(-reuse_pen, 4),
                    "form": round(form_bonus, 4),
                    "conciseness": round(conciseness[i], 4)}  # shorter top-performers
            with rollout_context(rollout_ids[i]):
                logger.info(f"[TelestichEnv] reward={sum(comp.values()):.3f} {comp} "
                            f"(reuse={reuse[i]:.2f} dup={dup[i]:.0f} winner={i in winners})")
            out.append(comp)
        return out
