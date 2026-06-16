---
name: launch-run
description: Launch a castform GPU training run with `castform launch` (validate → upload → launch) and set launcher args correctly. Use only after verify-environment is green — this spends GPU.
---

# Launch a run

`castform launch` runs the full flow: pre-flight `validate` → upload env+datasets
→ launch. **This spends real GPU** — only launch after `castform validate` is
green. It prompts for confirmation; pass `--yes` for non-interactive.

```bash
castform launch --name my-run --set model=Qwen/Qwen3.5-4B
```

It prints the run URL and a `castform runs status …` command to track it.

## Launcher args — discover, don't guess

The accepted args (and their defaults / ranges / soft-caps) are defined by the
server. List them:

```bash
castform launch --list-args
```

Set args with `--set key=value`. The CLI validates each against the live schema
and **rejects unknown keys**, so you can't silently send a wrong name.

Key ones:
- **`model`** — e.g. `Qwen/Qwen3.5-4B` (smaller, cheaper) or `Qwen/Qwen3.5-35B-A3B`.
- **`max_rollout_len`** — total tokens generated across the WHOLE rollout (all
  turns), not a per-response cap. This is the real knob; **`max_response_len` is
  not a thing** (the server rejects it). A rollout that hits the budget is
  truncated and dropped from the loss — set it generously.
- **`max_turns`** — defaults to **4**. If your env is multi-turn, set this
  explicitly: `--set max_turns=N`. The trainer does not read the env's
  `recommended_max_turns`.
- **`group_size`** — rollouts per prompt for GRPO. Defaults are model-tuned
  (4B→9); omit it to take the default unless you have a reason.

## Run types

`--type simple` is the GPU training pool. `--type simple-cpu` is a CPU-only smoke
pool (cheap) for testing the launch lifecycle without GPU. (`simple-r5` from older
docs is not implemented.)

After launch, go to the **view-progress** skill.
