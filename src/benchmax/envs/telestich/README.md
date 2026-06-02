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

`compute_group_reward` scores the whole GRPO group in three stages. The reward
is the sum of the logged components (`quality + form + conciseness −
reuse_penalty`):

1. **Hard rules** (deterministic, no LLM). A rollout must be a valid telestich
   — acrostic spells the target, every line ends on a real word, exact line
   count — and must not write the hidden word in the body. Fail → reward 0,
   the judge is never called.
2. **Quality** — the shared rubric judge (`benchmax.rubrics.evaluate_rubric_ranking`)
   ranks the group's valid poems in one call, with the example's **great** poem
   inserted blind as `ground_truth`. Each poem's score in `[0,1]` is anchored to
   that reference's rank: above the great bar → `[0.5, 1.0]`, below → `[0, 0.3]`.
   Judging is relative, not absolute. (Single great anchor today; the
   `acceptable_refs` are kept in the dataset for a future multi-anchor mode —
   set `ANCHOR_KIND` to switch which reference is the bar.)
3. **Adjustments** (deterministic):
   - **reuse_penalty** — discounts quality for line-ending words shared with
     sibling rollouts (per ending word, within the group); fights ending-word
     mode collapse (e.g. every rollout ending an `i`-line on "ski").
   - **form** — small bonus: English CMU rhyme density / Chinese line-length
     uniformity.
   - **conciseness** — winner-anchored efficiency: only "top performers"
     (within `WINNER_EPS` of the group's best quality and above `WINNER_BAR`)
     earn it; among them, shorter output and fewer tool calls score higher.
     The quality threshold keeps short degenerate outputs from gaming length.
   Near-duplicate whole poems within a group are also divided down.

The `acceptable` (competent-but-plain) and `great` (excellent) reference poems
are generated offline and cached per example. Examples missing an anchor fall
back to default band positions.

## Example

[`example.py`](example.py) generates a dataset, bundles this env, and launches a
training run. Run it from the benchmax project root:

```bash
CASTFORM_API_KEY=sk_... CASTFORM_LLM_API_KEY=sk_... \
    uv run --extra telestich python -m benchmax.envs.telestich.example
```

`telestich_dataset.jsonl` (next to the script) is the seed dataset; the script
generates more via the platform LLM to reach the target example count.
