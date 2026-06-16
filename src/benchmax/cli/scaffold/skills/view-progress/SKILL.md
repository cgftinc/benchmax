---
name: view-progress
description: Monitor a castform training run — status, reward scalars, and logs via the castform runs commands. Use after launching to watch a run or debug a stalled/failed one.
---

# View progress

Track a run with the `castform runs` commands (all take `--json`):

```bash
castform runs list                 # your runs + status
castform runs status <run-id>      # status + step progress + latest activity
castform runs scalars <run-id>     # reward / loss curves (latest value per metric)
castform runs logs <run-id>        # environment / error logs
```

Every run is also viewable at `app.castform.dev/train/<run-id>` (printed at launch
and by `runs get`).

## What to watch

- **Status**: `pending` → `active` (running) → `complete`. Other terminal states:
  `failed`, `stalled`, `cancelled`, `out_of_credits`, `billing_error`.
- **`scalars`** is the signal. `mode` is dynamic per run (`train`, `eval`, …);
  `runs scalars` defaults to `train` — pass `--mode eval` for the eval curves.
  Reward should trend up; watch the eval reward for generalisation, not just train.
- **`logs`** surfaces environment/reward errors per run (and per rollout with
  `--rollout-id`). Check here first when a run `failed` or rewards look wrong.

## Controlling a run

```bash
castform stop <run-id>             # cancel a run you own
```

A launched run reports "Job cancellation requested" and emits `training.cancelling`;
a run with no launcher job is just marked complete.

## If a run misbehaves

- `failed` early → check `runs logs` for an env/import/reward error, fix `run.py`,
  re-`validate`, re-`launch`.
- `stalled` → the worker stopped reporting; check `runs logs` and the run URL.
- Flat/odd reward → the reward function, not the data. Go back to
  **verify-environment** and inspect the values `castform validate` prints.
