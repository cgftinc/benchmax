---
name: train-sft
description: Build a validated benchmax-sft-v1 dataset from existing labeled conversations, upload it, and explicitly launch a supervised finetuning run — no environment, rewards, or rollouts.
---

# Train with SFT

Use this when the user **already has the completions they want the model to
imitate** — support transcripts, labeled chat data, tool-call traces, input→output
pairs. There is no environment, tool loop, reward, or validate stage: the RL loop
in this project's other skills does not apply. If the task needs the model to
*discover* good behavior against a scorer, use the RL skills instead.

## The whole flow

```python
from benchmax.sft import SftDataset, SftDatasetError
from castform.platform import SftTrainingConfig, TrainerClient, upload_sft_assets

train = SftDataset.from_jsonl("train.jsonl")      # or SftDataset.from_rows(rows)
uploaded = upload_sft_assets(dataset=train, run_name="support-sft")
run_id = TrainerClient().launch_sft_run(
    assets=uploaded,
    name="support-sft",
    config=SftTrainingConfig(num_epochs=1, learning_rate=1e-5, seed=42),
)
```

Construction is **all-or-nothing**: `SftDataset` either satisfies the whole
`benchmax-sft-v1` contract or raises `SftDatasetError` with every issue, ordered
and line-aware. Fix the data the diagnostics point at — never pre-filter rows
silently or patch around individual issues without telling the user.

## Row contract (one JSON object per line)

```json
{
  "messages": [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4", "weight": 1}
  ],
  "tools": [],
  "metadata": {"id": "optional producer identity"}
}
```

- `system`/`user`: exactly `role` + non-empty string `content`.
- `assistant`: optional string-or-null `content`, optional non-empty
  `tool_calls`, optional integer `weight` `0 | 1` (omitted means `1`). Each
  assistant turn needs content or a tool call; each row needs at least one
  assistant turn with effective weight `1`.
- `tool` results: exactly `role` + string `content` + non-empty `tool_call_id`;
  every tool call gets exactly one result, in declaration order, before the
  next non-tool message. Tool definitions are OpenAI function shapes;
  `function.arguments` must decode as a JSON object.
- Everything else is rejected: images/audio/multimodal parts, fractional
  weights, legacy prompt/completion keys, unknown fields, duplicate JSON keys,
  rows over 1 MiB, more than 1024 messages.

**Masking:** set `weight: 0` on assistant turns that are context, not target
(earlier drafts, retrieved answers, another model's output). Only weight-1
turns contribute to the loss.

## What the user actually chooses

| arg | accepted | default |
|---|---|---|
| `num_epochs` | 1–100 | 1 |
| `learning_rate` | (0, 0.1] | 1e-5 |
| `max_context_tokens` | 256–8192, or 32768 / 65536 / 131072 (long context — must be enabled on the platform) | 8192 |
| `save_interval` (steps) | 1–10000 | 20 |
| `seed` | 0–2147483647 | 42 |
| `lr_decay_style` | `"constant"` or `"cosine"` | unset — keeps the platform default |
| `min_lr` | ≥ 0 and below `learning_rate` | unset |
| `warmup_ratio` | 0–0.5 of total steps | unset |
| `adam_beta2` | 0.9–0.999 | unset |
| `grad_clip` | (0, 10] | unset |
| `lora_rank` | 32 or 64 | unset — keeps the platform's fixed policy |
| `global_batch_size` | 4–64, multiple of 4 at `max_context_tokens` ≤ 8192; 1–64 at a long-context rung (the platform picks the topology, so it decides divisibility) | unset — 4 |
| `eval_interval` (steps) | 1–10000; only with an eval set | unset — derived from `save_interval` |

Every arg whose default reads `unset` is optional: leave it out and the
platform's own value applies, so an untouched config behaves exactly as before
these knobs existed. Do not set one just to restate a default.

`lora_rank` picks the adapter's rank; alpha is always derived as 2x and is not
a knob. Rank 128 is not accepted — it trains, but serving cannot load it.

Model (`Qwen/Qwen3.5-4B`) and GPU topology are platform-owned — do not invent
knobs for them. A row that renders past `max_context_tokens` tokens fails the
run's preflight; trim long rows up front.

## Held-out eval (optional)

Pass a second dataset to score during training:

```python
assets = upload_sft_assets(dataset=train, eval_dataset=held_out, run_name="...")
```

The eval set is scored on the live weights between training steps and plotted
as `eval/loss` beside `train/loss`. Rules worth knowing before you offer it:

- **At most 2048 rows**, under the same per-row limits as training rows. The
  bound is a compute bound, not a taste one — eval runs on the training GPUs
  and pauses training while it does.
- **It is never billed.** Eval forward passes cost the user nothing, which is
  also why the size and cadence are capped rather than left open.
- **Cadence defaults to `save_interval`**, so by default every eval lands on a
  checkpoint. Set `eval_interval` only to make it sparser; a much denser
  cadence is rejected at launch.
- An eval set is part of the data identity: changing it produces a different
  upload prefix, and a resumed run must keep the one it started with.

Skip it for small or exploratory runs — a held-out split costs training rows,
and `train/loss` alone answers "is this learning at all".

## Cost and consent

`launch_sft_run` spends GPU credits. Ask the human before calling it, every
time — preparing and uploading the dataset first is free and fine. Steps per
epoch ≈ rows / 4 (tiny datasets pad by repeating their first rows, so 1–3-row
datasets train on repeats; prefer at least a few dozen rows). Stopping a run
keeps only checkpoints already uploaded; work since the last one is lost.

If launch fails with "SFT launch is not enabled", the platform gate is off for
this account — surface that to the user rather than retrying.

## Monitor

Same as any run (`view-progress` skill): `castform runs status <id>`, and
`castform runs scalars <id>` — watch `train/loss` fall, and `eval/loss` too
when the run has an eval set. There are no rollouts or reward curves for SFT
runs. The run page shows loss, dataset prefix, and config; runs with an eval
set also get an eval tab, chart-only.

## Reference

The canonical worked example (streaming a pinned public corpus, bounded
mapping, offline tests, explicit paid `--launch` gate) ships in the benchmax
repo under `examples/sft/pii_masking/`.
