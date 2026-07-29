# math

a small sanity-check environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) that teaches a model to solve arithmetic expressions with four tools.

## example task

each row is an arithmetic expression; the model must use the tools and return the final number in answer tags:

```
user: What is 31 + 4 * 12?
tool: multiply(a=4, b=12) → 48
tool: add(a=31, b=48) → 79
assistant: <answer>79</answer>
```

## launch training

```bash
cd examples/math
uv run python main.py launch

# if iterating on the env, validate first
uv run python main.py validate
```

launch downloads the public [`dawidmt/arithmetic50`](https://huggingface.co/datasets/dawidmt/arithmetic50) test split (40 training and 10 evaluation examples), uploads the environment and dataset, validates them, then asks for confirmation before spending credits (pass `--yes` to skip).

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end.

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
