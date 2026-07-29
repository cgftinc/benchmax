# math

a small sanity-check environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) that teaches a model to solve arithmetic expressions with four tools.

## validate the environment

```bash
cd examples/math
uv run python main.py validate
```

this downloads the public [`dawidmt/arithmetic50`](https://huggingface.co/datasets/dawidmt/arithmetic50) test split, uses the first 40 examples for training and the remaining 10 for evaluation, bundles and uploads the environment and dataset, then validates them locally and in a hosted sandbox. it does not launch training.

use this command while iterating on the environment. validation runs the first evaluation example with a small model context so it stays fast.

## launch training

```bash
uv run python main.py launch
```

launch follows the same data, upload, and validation path, then asks for confirmation before starting training with the assets that were just validated. pass `--yes` to skip only the launch confirmation.

## environment

`MathEnv` extends [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md), limits each rollout to three model turns, and exposes the four basic arithmetic operations.

```python
class MathEnv(BaseEnv):
    max_turns = 3

    async def create_dataset(...):
        return JsonlDataset(...)

    async def list_tools(...):
        return [add, subtract, multiply, divide]

    async def run_tool(...):
        ...

    async def compute_reward(...):
        return {"correctness": ...}
```

the correctness reward requires the model to call at least one tool and return the right numeric answer. ten percent of successful tool responses also include 1,000 irrelevant characters, preserving the original math sanity check that makes the model identify the result instead of copying the entire tool response.

## extensions

see [`extensions/`](extensions/) for runnable `MathEnv` subclasses that demonstrate group scoring and stress trainer recovery and error handling.
