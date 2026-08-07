# pii masking sft example

the canonical end-to-end castform SFT example: map a bounded slice of a public
PII-masking corpus into a validated `benchmax-sft-v1` dataset, then (explicitly)
upload it and launch a LoRA finetune of Qwen3.5-4B.

deterministic on purpose: it reads the literal `train` split of the pinned
revision in source order, inspects exactly `--rows` records (default 256, hard
maximum 4096) with no shuffle or filtering, and fails rather than emitting fewer.
the corpus is streamed — the 4.61 GB download never happens.

## prepare (free, local)

from the workspace root:

```bash
uv run --group sft-example python -m pii_masking.main --rows 256 --output train.jsonl
```

this streams the source, validates every mapped row through `benchmax.sft.SftDataset`,
and writes canonical JSONL locally. nothing is uploaded.

## launch (paid)

> **cost warning:** `--launch` uploads the dataset and starts a real GPU training run
> that spends credits. **cancellation warning:** stopping a run guarantees only the
> last successfully uploaded periodic checkpoint — work since that checkpoint is
> discarded, and a final checkpoint is not promised on cancellation.

```bash
uv run --group sft-example python -m pii_masking.main --rows 256 --output train.jsonl --launch --run-name pii-masking-sft
```

the model (Qwen3.5-4B), LoRA policy, and GPU topology are platform-owned; the
example passes only the default `SftTrainingConfig` choices.

## source data and attribution

- source: [ai4privacy/pii-masking-openpii-1m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m)
  by **Ai4Privacy / Ai Suisse SA**
- license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — commercial use
  and redistribution permitted with attribution
- pinned revision: `ecfdc547f4a0955600cfe6ab98ba2a162207fcc0`
- this example transforms the source's `source_text` / `masked_text` records into
  chat-format training examples; source rows are never vendored into this repository,
  and CI never downloads them (tests use original synthetic fixtures only)

to update the pinned revision deliberately: pick the new commit hash on the dataset's
hub page, update `SOURCE_REVISION` in `main.py`, re-run prepare, and review the
resulting dataset before launching. do not drop the pin — unpinned streams make the
example non-reproducible.

do **not** substitute `ai4privacy/pii-masking-300k`: its custom license restricts
commercial use and derivatives, and it is not compatible with this workflow without
recorded written permission.

## dependencies

`datasets==5.0.1` lives only in the workspace `sft-example` dependency group (hence
`uv run --group sft-example`); it is not a runtime dependency of `benchmax` or
`castform`.
