---
name: view-progress
description: Monitor a castform training run — status, reward scalars, and logs via the castform runs commands. Use after launching to watch a run or debug a stalled/failed one.
---

# View progress

> The four-step path: `castform setup → data → validate → launch`. This skill is
> **after launch** — monitoring the GPU run. See `GETTING_STARTED.md`.

## Fast path

Track a run with the `castform runs` commands (all take `--json`):

```bash
castform runs status <run-id>      # status + step progress + latest activity
castform runs scalars <run-id>     # reward / loss curves (latest value per metric)
```

Status flows `pending` → `active` (running) → `complete`. `scalars` is the signal
— **reward should trend up.** Every run is also viewable at
`app.castform.dev/train/<run-id>` (printed at launch).

## Going deeper

### Did training beat the baseline?

This is the real question — and there's **no `compare` verb**, so do it manually.
Your **baseline** is the eval reward you saw at `castform validate` (and step-0 of
the run). Compare the **eval** curve, not train — train reward can climb while
eval flatlines (overfitting):

```bash
castform runs scalars <run-id> --mode eval         # this run's eval reward
castform runs scalars <other-run-id> --mode eval   # a second run to compare against
```

`--mode` is dynamic per run (`train`, `eval`, …); `scalars` defaults to `train`
when present, else the first available mode — so pass `--mode eval` explicitly for
generalisation. To compare two runs (e.g. two hyperparameter settings), read each
one's eval scalars and diff them yourself.

### Full run list + logs

```bash
castform runs list                 # your runs + status
castform runs logs <run-id>        # environment / error logs (--rollout-id for one rollout)
```

Terminal states beyond `complete`: `failed`, `stalled`, `cancelled`,
`out_of_credits`, `billing_error`.

### Controlling / debugging a run

```bash
castform stop <run-id>             # cancel a run you own
```

- `failed` early → `castform runs logs` for an env/import/reward error; fix
  `run.py`, re-`validate`, re-`launch`.
- `stalled` → the worker stopped reporting; check `runs logs` and the run URL.
- Flat/odd reward → the reward function, not the data. Go back to
  **verify-environment** and inspect the values `castform validate` prints.
