# Telestich Environment

`TelestichEnv` rewards a model for writing **telestich** poems — poems where the
last letter of each line, read top to bottom, spells out a hidden target word
given in the prompt. English-only.

## Installation

```bash
pip install "benchmax[telestich]"
```

Includes:
- `english_words`, `wordfreq` — real-word validity checks
- `pronouncing` — CMU rhyme scoring for the rhyme bonus

The judge uses an OpenAI-compatible endpoint (`openai`). The `feedback` tool is
fully deterministic — no LLM.

## Usage

```python
from benchmax.envs.telestich.telestich_env import TelestichEnv

env = TelestichEnv(judge_base_url=..., judge_api_key=...)
```

Each dataset example is
`{"prompt": str, "ground_truth": <hidden word>, "acceptable_refs": [poem, ...],
"great_refs": [poem, ...]}`. The two reference sets are the quality **anchors**
the judge ranks against (generated offline).

## Tools

- **`feedback(poem, word)`** — deterministic formative feedback on a draft (≤3
  calls/rollout). The model passes its draft **and** the hidden word it should
  spell; the tool **stacks every issue in one pass** — line count, per-line wrong
  letters (over the shared range even when the count is off), filler/blacklisted
  endings, non-word / too-short endings, **prose run-on lines** (over `LINE_CHAR_CAP`
  chars), and the hidden word leaked into a line — so the model can fix them all in
  one revision. It never sets reward and never reveals the hidden word.

## Reward

`compute_group_reward` scores the whole GRPO group, writing each rollout's
components onto a `_Rollout` record. Reward = the sum of the logged components:
`quality + rhyme + diversity + conciseness` (every component ≥ 0).

0. **Parse** each rollout once: completion length, the `<answer>` poem, lines,
   tool-call count.
1. **Hard rules** (deterministic, no LLM). A perfect telestich — acrostic spells
   the target, exact line count, every line ends on a real word, each line is a
   real poem line (≥ `MIN_EN_LINE_WORDS` words), no hidden word in the body — is
   **CORRECT** and goes to the judge. A near-miss (answer present, right line
   count, no cheat, **>`MIN_CORRECT_FRAC`** of lines correct) is **PARTIAL** — a
   small graded reward (`PARTIAL_HI`→`PARTIAL_LO`) so there's always a gradient.
   Cheating / no answer / wrong line count / ≤25% correct → **0**.
2. **Quality** — the shared **multi-anchor** rubric judge
   (`benchmax.rubrics.evaluate_rubric_ranking`) ranks the CORRECT poems against
   **both** references inserted blind (`acceptable` as a floor, `great` as the bar),
   in **batches of ≤`JUDGE_BATCH` poems** per call (ranking the whole group at once
   lets near-identical siblings contaminate each other's placement and the judge
   over-rates them — see the run-0ec8e2dc audit). The band score maps onto a reward
   ladder:
   - below acceptable → `[0.1, 0.4)` — **below** bucket (floored at `MIN_CORRECT`)
   - acceptable…great → `[0.4, 0.7]` — **mid** bucket
   - above great → `(0.7, 1.0]` — **above** bucket

   The judged score is then scaled down by two **deterministic** quality penalties
   the judge under-charges: `_line_length_penalty` (lines past `LINE_CHAR_CAP` ≈ 90
   chars — the prose-run-on degeneracy) and `_ending_penalty` (blacklisted line
   endings — `_HARD_FORCED_ENDINGS` interjections/fillers scaled by `W_HARD_ENDING`,
   `_SOFT_FORCED_ENDINGS` mode-collapse nouns scaled less by `W_SOFT_ENDING`).
   Near-duplicate whole poems are divided down before bucketing.
3. **Secondary bonuses** (deterministic, all **quality-scaled** so a weak poem
   can't farm them):
   - **rhyme** = `W_RHYME · q · rhyme_score` — CMU rhyme density; awarded to any
     correct poem.
   - **diversity** = `W_DIVERSITY · q · (1 − reuse)` — rewards line-ending words
     *not* shared with sibling rollouts; fights ending-word mode collapse.
   - **conciseness** = `W_CONCISE · q · len_eff + W_TOOL_EFF · q · tool_eff` —
     shorter generation **and** fewer feedback calls, contributing equally.
     `len_eff = exp(−max(0, completion_len/budget − 1))`;
     `tool_eff` is graded (full at 0 calls → 0 at the call cap), so the model
     weans off the tool as it gets reliable.

   Diversity and conciseness apply only to the group's **top occupied band**, so
   even an all-`below` group keeps a gradient (anti-collapse).

The `acceptable` (competent-but-plain) and `great` (excellent) references are
generated offline and cached per example; missing anchors degrade gracefully
(against none it's pure relative ranking). Every rollout logs one path-revealing
line, and correct poems get a full per-component breakdown.

## Example

[`example.py`](example.py) generates a dataset, bundles this env, and launches a
training run. Run it from the benchmax project root:

```bash
uv run --extra telestich python -m benchmax.envs.telestich.example
```

Auth uses the device-auth session (a browser login is opened automatically if
`~/.castform` has no valid session) — no API key needed.

`telestich_dataset.jsonl` (next to the script) is the curated English seed dataset
(curriculum-ordered). Set `TELESTICH_FULL_RUN=1` for a real run on the full set;
the default is a 2-example smoke that exercises generate → bundle → upload →
launch.
