# dominant-color

A multi-turn vision memory game. Each rollout shows the model three
checkered-tile images, one at a time — about 55% of each image's tiles share
one dominant color. Image 1 arrives in the prompt; the rest are revealed with
the `see_next_image` tool, so every rollout pushes mid-rollout images through
tool responses. The reward is all-or-nothing on reporting the dominant colors
in the order seen.

Purpose: the minimal deterministic exercise of tool-returned images in
training — multi-turn vision, cross-turn memory, and image-bearing tool
responses, with no network or external dataset.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/dominant-color
uv run python main.py             # data → validate (no GPU)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```
