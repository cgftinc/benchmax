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

The defaults are sensible for a first smoke run — for a serious RAG run, also set
the rollout budget and think about epochs. If eval starts falling while train keeps
rising, the best checkpoint is likely before the final step.

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
| `num_epochs` | `5` | passes over the train set. For RAG, 3 is often a safer first serious run than training deep into an overfit tail; watch eval and keep the best checkpoint, not just the final. |
| `group_size` | `9` | rollouts per prompt for GRPO; drives **both** train and eval rollout counts. |
| `lora_rank` | `128` | LoRA adapter rank (trainable-parameter count). |
| `lora_alpha` | `256` | LoRA scaling factor; convention is `2 × lora_rank`. |
| `max_rollout_len` | model-derived | total tokens across the WHOLE rollout (all turns), not a per-response cap; `> 16384` risks OOM. **`max_response_len` is not a thing** — the server rejects it. Over-budget rollouts are truncated and dropped from the loss, so set it generously while keeping search output compact. |
| `max_turns` | `4` (trainer default) | max turns per rollout; set it explicitly if your env is multi-turn (the trainer ignores the env's `recommended_max_turns`). |

**Tool calls cap at 8 at launch — and you can't raise them.** Unlike `castform validate`
(which takes `--max-tool-calls`), launch exposes only `max_turns` as a `--set` knob;
`max_tool_calls` is fixed at 8. A multi-turn / search env that needs more is silently
truncated in training — keep `MAX_SEARCH_CALLS` ≤ 8 (see design-environment's Tools / turns).

For search/RAG, budget the transcript, not just the final answer. Tool outputs count
against `max_rollout_len` across all turns: `MAX_SEARCH_CALLS × per-search output`
can push a run into truncation or OOM even when each individual search result is
reasonable. Prefer a smaller default search budget, trimmed per-chunk bodies, and a
`max_rollout_len` you chose after checking `castform launch --list-args`.

### Bake the config into run.py (reproducible launches)

A `run.py` can carry a `LAUNCH_CONFIG` dict (and `VALIDATE_CONFIG` for the
pre-launch check) that `castform launch` / `castform validate` read, so the run
reproduces from the file without remembering flags — a `--set` / CLI flag still
overrides per invocation:

```python
LAUNCH_CONFIG = {
    "max_turns": 7,             # trainer ignores recommended_max_*; bake it here
    "max_rollout_len": 16384,   # whole-rollout token budget
    "num_epochs": 3,
    # "type": "simple",         # gpu pool; "simple-cpu" for a smoke run
}
VALIDATE_CONFIG = {"max_turns": 7, "max_tool_calls": 6, "examples": 6}
```

The launcher keys must be real `--set` args (`--list-args`) — an unknown key is
skipped with a warning, not sent. `max_tool_calls` is **not** a launch arg (stays 8),
so keep it out of `LAUNCH_CONFIG`; it *is* honored in `VALIDATE_CONFIG`. This is the
fix for the `MAX_SEARCH_CALLS` ↔ `--max-turns` sync footgun — set the budget once, in
the file, next to `MAX_SEARCH_CALLS`.

### Epochs, checkpoints, and overfitting

Read eval, not train, as the launch succeeds. A healthy train curve can keep rising
while eval peaks and then falls; in that case the final checkpoint is worse than an
earlier one. For RAG, start with fewer epochs when the validation curve from a prior
run showed a peak before the end, and record the best eval step in your run notes.

The CLI may not expose checkpoint listing/serving for a non-final step yet. If eval
peaks early, do not assume the platform is serving the final model you want; use the
run URL or support/internal tooling to confirm checkpoint availability before calling
the run "done."

### Auth and long-running launches

Device-login sessions can expire during overnight investigation. If a launch or
follow-up `runs` command starts failing auth after a long pause, refresh with
`castform login` (or use `PLATFORM_API_KEY` for headless runs) and retry the read or
launch operation.

Server-controlled fields — `save`, `load`, `global_batch_size`, the eval mirrors —
are **not settable**: the launch handler fills them in and rejects caller input
that carries them. (`rollout_batch_size` is derived too, not a launch arg.)
The run type is platform-internal. `castform launch` uses the standard GPU
training pool; lifecycle smoke tests should use `castform validate` instead of a
launch-only CPU template.
