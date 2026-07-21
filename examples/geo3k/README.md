# geo3k

Visual geometry problems from the public `chenhegu/geo3k_imgurl` dataset. The
model reads a geometry diagram, may call the `zoom` tool to magnify a small or
unclear region (the crop returns as an image inside the tool response), and
answers with `\boxed{...}`. Reward is boxed-answer correctness.

Purpose: a real multimodal RL task exercising runtime dataset resolution
(HuggingFace at trainer time), remote-URL diagram images, and optional
mid-rollout tool images.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/geo3k
uv run python main.py             # data (HF prefetch) → validate (no GPU)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```
