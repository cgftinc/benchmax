# geo3k

a multimodal geometry environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) using the public [`chenhegu/geo3k_imgurl`](https://huggingface.co/datasets/chenhegu/geo3k_imgurl) dataset and an image-returning zoom tool.

## example task

each row pairs a geometry diagram with a question, and only the final boxed answer is scored:

```
user: [diagram: circle with an inscribed triangle] Find x. Round to the nearest tenth if necessary.
tool: zoom(x0=0.4, y0=0.5, x1=0.9, y1=1.0) → [magnified crop of the labeled angle]
assistant: ... \boxed{12.5}
```

## launch training

```bash
cd examples/geo3k
uv run python main.py launch

# if iterating on the env, validate first
uv run python main.py validate
```

launch caches the hugging face dataset, uploads the environment bundle, validates it, then asks for confirmation before training with at most 256 training and 32 evaluation examples (pass `--yes` to skip).

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end.

## environment

```python
class Geo3KEnv(BaseEnv):
    async def create_dataset(...):
        return Geo3KDataset(...)

    async def list_tools(...):
        return [zoom]

    async def run_tool(...):
        return cropped_image_content_parts(...)

    async def compute_reward(...):
        return {"correctness": boxed_answer_matches(...)}
```

the model reads a geometry diagram, may call `zoom` with normalized crop coordinates, and answers with `\boxed{...}`. zoom is optional and does not directly affect reward.
