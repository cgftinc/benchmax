# dominant-color

a multi-turn vision sanity-check environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) that tests image-bearing tool responses and cross-turn memory.

## example task

the first image rides in the prompt; the model calls `see_next_image` to reveal the rest, then answers from memory:

```
user: [image 1: noisy checkered tile, mostly red] This is image 1 of 3.
tool: see_next_image() → [image 2: mostly teal]
tool: see_next_image() → [image 3: mostly purple]
assistant: \boxed{red, teal, purple}
```

colors come from a fixed sixteen-name palette, and the answer must list every dominant color in the order seen.

## launch training

```bash
cd examples/dominant-color
uv run python main.py launch

# if iterating on the env, validate first
uv run python main.py validate
```

the dataset is generated deterministically at runtime, so launch only uploads the environment bundle, validates it, then asks for confirmation before spending credits (pass `--yes` to skip).

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end.

## environment

`DominantColorEnv` shows three checkered images in sequence. roughly 55% of each image uses one dominant color; the first image is in the prompt and `see_next_image` returns the remaining images from tool calls.

```python
class DominantColorEnv(BaseEnv):
    async def create_dataset(...):
        return deterministic_image_sequences(...)

    async def list_tools(...):
        return [see_next_image]

    async def run_tool(...):
        return next_image_content_parts(...)

    async def compute_reward(...):
        return {"correctness": exact_order_match(...)}
```

the reward is all-or-nothing on returning every dominant color in the order shown.
