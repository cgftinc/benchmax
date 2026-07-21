# math

Small mixed-operator arithmetic (the public `dawidmt/arithmetic50` split:
`×`/`÷` symbols and order-of-operations traps that reliably confuse small
models) solved through add/sub/mul/div tools, with deliberately padded tool
output. Recreates the trainer's historical "mathenv" fixture on the current
BenchMax API, including sentinel failure injection for exercising every
BaseEnv failure path.

Purpose: the fastest end-to-end smoke of the whole training loop — tools,
uploaded JSONL datasets, group scoring (`math_group_env.MathGroupEnv`), and
failure-path fixtures. The trainer's pipeline e2e tests build on this example.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/math
uv run python main.py             # data (HF download) → validate (no GPU)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```
