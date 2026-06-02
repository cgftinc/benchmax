# Telestich Environment

`TelestichEnv` rewards a model for writing **telestich** poems — poems where
the last letter (or character, for Chinese) of each line, read top to bottom,
spells out a hidden target word given in the prompt.

## Installation

```bash
pip install "benchmax[telestich]"
```

Includes:
- `english_words`, `wordfreq` — real-word validity checks and the per-letter
  ending-word bank
- `pronouncing` — CMU rhyme scoring for the English form bonus

The judge uses an OpenAI-compatible endpoint (`openai`, a core dependency).

## Usage

```python
from benchmax.envs.telestich.telestich_env import TelestichEnv

env = TelestichEnv(judge_base_url=..., judge_api_key=...)
```

Each dataset example is
`{"prompt": str, "ground_truth": <hidden word>, "acceptable_refs": [poem, ...],
"great_refs": [poem, ...]}`. The two reference sets are the quality **anchors**
the judge ranks against (generated offline; see below).

## Tools

- **`word_bank(letter)`** — returns ~30 real, poem-usable English words ending
  in `letter` (≥3 letters, frequency-weighted; function words / single letters
  filtered out). Capped at 2 calls per rollout; not offered for Chinese.

## Reward

`compute_group_reward` scores the whole GRPO group through four stages, writing
each rollout's components onto a `_Rollout` record. The reward is the sum of the
logged components: `quality + form + diversity + conciseness`.

0. **Parse** each rollout once into a record: completion length, the `<answer>`
   poem, poem length, lines, language, tool-call count.
1. **Hard rules** (deterministic, no LLM). A rollout must be a valid telestich —
   acrostic spells the target, exact line count, every line ends on a real word
   (en), and **each line is a real poem line** (≥ `MIN_EN_LINE_WORDS` words /
   ≥ `MIN_ZH_LINE_CHARS` CJK chars, which kills the "acrostic spelled vertically"
   hack) — and must not write the hidden word in the body. Fail → `quality = 0`,
   the judge is never called (it still receives the conciseness penalty).
2. **Quality** — the shared **multi-anchor** rubric judge
   (`benchmax.rubrics.evaluate_rubric_ranking` with `anchors` + `band_edges`)
   ranks the group's valid poems in one call with **both** references inserted
   blind: `acceptable` as a floor, `great` as the bar. A poem's band score:
   - below acceptable → `[0, 0.1)` — the **below** bucket (degenerate / sub-par)
   - acceptable…great → `[0.1, 0.5]` — the **mid** bucket
   - above great → `[0.5, 1.0]` — the **above** bucket

   The floor anchor gives quality an **absolute zero-point**: an all-bad group
   ranks below acceptable → everyone near 0 (pure relative ranking can't do
   that). Near-duplicate whole poems are divided down before bucketing.
3. **Secondary terms** (deterministic; all bonuses are **quality-scaled** so a
   weak poem can't farm them):
   - **form** = `W_FORM · q · form_score` — English CMU rhyme density / Chinese
     line-length uniformity.
   - **diversity** = `W_DIVERSITY · q · (1 − reuse)` — rewards line-ending words
     *not* shared with sibling rollouts; fights ending-word mode collapse (every
     `i`-line on "ski"). Applied only to the **mid + above** buckets (quality ≥
     `QUALITY_GATE`).
   - **conciseness** = `W_CONCISE · q · (0.8·len_eff + 0.2·tool_eff)` — a
     **positive** bonus awarded only to the **top occupied band** (above, else
     mid), rewarding a good poem that reached its answer quickly. `len_eff =
     exp(−max(0, completion_len/budget − 1))` (1 at/under the budget
     `LEN_BUDGET_BASE + LEN_BUDGET_PER_LINE × acrostic_len`, decaying above);
     `tool_eff = exp(−CALL_DECAY · n_tool_calls)` (tool thrift, a small part). A
     long-winded or tool-heavy top poem simply earns less of it.

All four components are **≥ 0**, so every reward is non-negative: a gated or
below-band rollout earns `0`, and the differences GRPO learns from come from the
positive bonuses among the poems that clear the bar.

The `acceptable` (competent-but-plain) and `great` (excellent) references are
generated offline and cached per example; an example missing one anchor is
scored against whichever remain, and against none it degrades to relative
ranking. Every rollout logs one path-revealing line (bucket, lengths, tool
count, and each component).

## Example

[`example.py`](example.py) generates a dataset, bundles this env, and launches a
training run. Run it from the benchmax project root:

```bash
CASTFORM_API_KEY=sk_... CASTFORM_LLM_API_KEY=sk_... \
    uv run --extra telestich python -m benchmax.envs.telestich.example
```

`telestich_dataset.jsonl` (next to the script) is the seed dataset; the script
generates more via the platform LLM to reach the target example count.
