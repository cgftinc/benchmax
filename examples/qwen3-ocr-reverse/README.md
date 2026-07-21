# qwen3-ocr-reverse

Hard-mode OCR: the model transcribes a rendered invoice-style page in
reversed reading order — bottom line first, characters right-to-left within
each line. Pages are small synthetic PNGs (data URIs) with deliberately varied
geometry so every example produces a different vision-feature grid. Reward is
deterministic, normalized to [0,1]: Hungarian-matched segment similarity,
segment-count agreement, and ordering.

Purpose: a single-turn multimodal task with uploaded image datasets — the
counterpart to geo3k's runtime-resolved remote images — plus a transcription
contract the base model cannot satisfy without learning.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/qwen3-ocr-reverse
uv run python main.py             # data (synthetic pages) → validate (no GPU)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```
