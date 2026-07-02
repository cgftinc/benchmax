---
name: launch-run
description: Launch a castform GPU training run with `castform launch` (validate → upload → launch) and set launcher args correctly. Use only after verify-environment is green — this spends GPU.
---

# Launch a run

> The four-step path: `castform setup → data → validate → launch`. This skill is
> the **launch** step (4) — only after `validate` is green. See `GETTING_STARTED.md`.

## Fast path

`castform launch` runs the full flow: pre-flight `validate` → upload env+datasets
→ launch. **This spends real GPU credits** — only launch after `castform validate`
is green. It warns that the run incurs GPU cost and prompts to continue (pass
`--yes` to skip the prompt for non-interactive use).

```bash
castform launch --name my-run --set model=Qwen/Qwen3.5-4B
```

It prints the run URL and a `castform runs status …` command to track it. Then go
to the **view-progress** skill.

The defaults are sensible — for a first run you usually set only `model`. Tune the
rest only when you have a reason (see Going deeper).

## Going deeper

### Discover the accepted args (don't guess)

The accepted args and their **live defaults / ranges / soft-caps** are defined by
the server — list them at runtime:

```bash
castform launch --list-args
```

`--list-args` is a **live server fetch** (so is the SDK's `list_launch_args()`) —
the arg set is **not** enumerable offline. Set args with `--set key=value`; the
CLI validates each against the live schema and **rejects unknown keys** (and
server-only fields), so you can't silently send a wrong name.

### Knobs worth tuning

Server-side source of truth: `platform-service/src/lib/trainer-args.ts`. These are
the launch-arg defaults (a per-model config may override; `--list-args` is
authoritative at runtime):

| `--set` key | default | what it does |
|---|---|---|
| `model` | `Qwen/Qwen3.5-4B` | model id; selects the trainer config (e.g. also `Qwen/Qwen3.5-35B-A3B`). |
| `learning_rate` | `1e-5` | Adam learning rate (slime `--lr`). |
| `num_epochs` | `5` | passes over the train set (small by default for fast first-run feedback). |
| `group_size` | `9` | rollouts per prompt for GRPO; drives **both** train and eval rollout counts. |
| `lora_rank` | `128` | LoRA adapter rank (trainable-parameter count). |
| `lora_alpha` | `256` | LoRA scaling factor; convention is `2 × lora_rank`. |
| `max_rollout_len` | model-derived | total tokens across the WHOLE rollout (all turns), not a per-response cap; `> 16384` risks OOM. **`max_response_len` is not a thing** — the server rejects it. Over-budget rollouts are truncated and dropped from the loss, so set it generously. |
| `max_turns` | `4` (trainer default) | max turns per rollout; set it explicitly if your env is multi-turn (the trainer ignores the env's `recommended_max_turns`). |

**Tool calls cap at 8 at launch — and you can't raise them.** Unlike `castform validate`
(which takes `--max-tool-calls`), launch exposes only `max_turns` as a `--set` knob;
`max_tool_calls` is fixed at 8. A multi-turn / search env that needs more is silently
truncated in training — keep `MAX_SEARCH_CALLS` ≤ 8 (see design-environment's Tools / turns).

Server-controlled fields — `save`, `load`, `global_batch_size`, the eval mirrors —
are **not settable**: the launch handler fills them in and rejects caller input
that carries them. (`rollout_batch_size` is derived too, not a launch arg.)

### Run types

`--type simple` (default) is the GPU training pool. `--type simple-cpu` is a
CPU-only smoke pool (cheap) for exercising the launch lifecycle without GPU.
(`simple-r5` from older docs is not implemented.)
