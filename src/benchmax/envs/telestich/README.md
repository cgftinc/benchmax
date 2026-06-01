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

Each dataset example is `{"prompt": str, "ground_truth": <hidden word>}`.

## Tools

- **`word_bank(letter)`** — returns 30 common English words ending in `letter`
  (frequency-weighted). Capped at 2 calls per rollout; not offered for Chinese.

## Reward

`compute_group_reward` scores each rollout and applies group-relative
adjustments:

- **quality** = acrostic *correctness* × *judge score*. Correctness checks
  every line's last letter against the target and that each ending is a real
  word. The judge (`gpt-5.4`) classifies problems (broken/nonsense/repetition/
  template/alignment) and rates specificity + coherence; Python computes the
  score deterministically. Writing the hidden word in the poem body zeros
  quality (constraint gaming).
- **rhyme** — form bonus, gated on perfect correctness. English: CMU
  perfect-rhyme density. Chinese: CJK char-count uniformity.
- **conciseness** — group-relative efficiency bonus anchored on the winning
  rollouts' length and tool-call counts.

Near-duplicate rollouts within a group (by full-poem text or by shared
line-ending sequence) are divided down to discourage mode collapse.

## Example

[`example.py`](example.py) generates a dataset, bundles this env, and launches a
training run. Run it from the benchmax project root:

```bash
CASTFORM_API_KEY=sk_... CASTFORM_LLM_API_KEY=sk_... \
    uv run --extra telestich python -m benchmax.envs.telestich.example
```

`telestich_dataset.jsonl` (next to the script) is the seed dataset; the script
generates more via the platform LLM to reach the target example count.
