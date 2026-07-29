# qwen3-ocr-reverse

a single-turn multimodal environment built with [`BaseEnv`](../../packages/benchmax/src/benchmax/envs/base/README.md) that trains hard-mode OCR on small synthetic invoice pages.

## example task

the prompt asks the model to transcribe the rendered page reading right to left and bottom to top:

```
image text:
INVOICE 4821
ACME CO 2026-03-14
TOTAL 96.20

expected answer:
02.69 LATOT
41-30-6202 OC EMCA
1284 ECIOVNI
```

the bottom line comes first and the characters within every line are reversed.

## launch training

```bash
cd examples/qwen3-ocr-reverse
uv run python main.py launch

# if iterating on the env, validate first
uv run python main.py validate
```

launch generates 128 training and 16 evaluation pages, uploads the environment and dataset, validates them, then asks for confirmation before spending credits (pass `--yes` to skip).

validate stops after the checks: it runs sample rollouts with a standard model, locally and in a hosted sandbox, just to confirm the environment runs end to end.

## environment

```python
class Qwen3OCREnv(BaseEnv):
    max_turns = 1

    async def create_dataset(...):
        return JsonlDataset(...)

    async def list_tools(...):
        return []

    async def compute_reward(...):
        return infinity_doc_reward(...)
```

each answer transcribes the rendered page in reversed reading order: bottom line first and characters right-to-left within each line. the deterministic reward combines matched-segment similarity, segment-count agreement, and ordering.
