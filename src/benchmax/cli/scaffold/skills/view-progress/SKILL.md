---
name: view-progress
description: Monitor a castform training run — status, reward scalars, and logs via the castform runs commands. Use after launching to watch a run or debug a stalled/failed one.
---

# View progress

Use this after launch to monitor the GPU run and debug stalled, failed, or
surprising reward curves.

## Fast path

Track a run with the `castform runs` commands (all take `--json`):

```bash
castform runs status <run-id>      # status + step progress + latest activity
castform runs scalars <run-id> --mode eval --json   # full eval trajectories
```

Status flows `pending` → `active` (running) → `complete`. Eval `scalars` are the
signal — **reward should trend up and stay up.** Every run is also viewable at
`app.castform.dev/train/<run-id>` (printed at launch).

## Going deeper

### Did training beat the baseline?

This is the real question — and there's **no `compare` verb**, so do it manually.
Your **baseline** is the eval reward you saw at `castform validate` (and step-0 of
the run) — for a held-out baseline, `castform validate --train eval_dataset.jsonl`
rolls out the eval split (there is no `--eval` rollout flag). Compare the **eval**
curve, not train — train reward can climb while eval flatlines (overfitting):

```bash
castform runs scalars <run-id> --mode eval --json         # this run's eval reward
castform runs scalars <other-run-id> --mode eval --json   # a second run to compare against
```

`--mode` is dynamic per run (`train`, `eval`, …); `scalars` defaults to `train`
when present, else the first available mode — so pass `--mode eval` explicitly for
generalisation. The human table is a latest-value summary; use `--json` for the
trajectory shape, peaks, and post-peak decline. To compare two runs (e.g. two
hyperparameter settings), read each one's eval scalars and diff them yourself.

Train reward can climb while eval falls. If eval peaks early, record the best step
and verify which checkpoint the platform can serve; the final checkpoint may be the
overfit one. A freshly-trained checkpoint can briefly lag in serving
(`model_unavailable` / a 502) — retry the read rather than assuming it failed.

### Compare with external evals

External evals run an external/base model on the same eval examples. Compare them
to the trained model's **eval** mode, never to train reward.

Use the web UI when possible. From an agent, use the platform endpoints:

```python
import httpx
from benchmax import config
from benchmax.platform.credentials import platform_bearer

run_id = "..."
c = httpx.Client(
    base_url=config.platform_url(),
    headers={"Authorization": f"Bearer {platform_bearer()}"},
    timeout=60,
)

evals = c.get(f"/v1/train/runs/{run_id}/external-eval").json().get("evals", [])
eval_id = evals[0]["id"]  # choose by model/status/createdAt
trained_avg = c.get(
    f"/v1/train/runs/{run_id}/rollouts/mode-average",
    params={"mode": "eval"},
).json()
external_avg = c.get(
    f"/v1/train/runs/{run_id}/rollouts/mode-average",
    params={"mode": "external-eval", "externalEvalId": eval_id},
).json()
comparison = c.get(
    f"/v1/train/runs/{run_id}/rollouts/comparison",
    params={"externalEvalId": eval_id, "page": 1, "limit": 20},
).json()
```

Use `mode-average` for the headline (`trained_avg["avg"]` vs
`external_avg["avg"]`) and `/rollouts/comparison` for matched examples
(`modelGroups` trained history, `compGroups` external eval). For one external
transcript, reuse the stored-rollout chain with `mode="external-eval"` and
`externalEvalId=eval_id`.

External evals usually have one batch/step, not a full trajectory. If the trained
model peaked before the final step, confirm which checkpoint can be served before
declaring whether it beat the external model.

### Full run list + logs

```bash
castform runs list                 # your runs + status
castform runs logs <run-id>        # environment / error logs (--rollout-id for one rollout)
```

Terminal states beyond `complete`: `failed`, `stalled`, `cancelled`,
`out_of_credits`, `billing_error`.

`runs logs` is not a stored-rollout transcript browser; it usually contains env
diagnostic/error logs. If reward looks odd, inspect stored rollouts instead.

### Investigate stored rollouts

To read actual answers and per-component rewards from a completed run, use the
built-in rollout commands (no raw HTTP needed):

```bash
castform runs rollouts <run-id> --mode eval            # example groups + latest mean reward
castform runs rollouts <run-id> --example <EXAMPLE ID> # one example's rollouts across steps
castform runs rollout  <run-id> <ROLLOUT ID>           # transcript + per-component rewards + gold
castform runs rollout  <run-id> <ROLLOUT ID> --view    # same, opened in the HTML viewer
```

`runs rollout` joins the **gold/ground truth** back from your local
`eval_dataset.jsonl` (then `train_dataset.jsonl`) by prompt text — pass `--dataset`
to point at a specific file. Add `--json` to any of them for the raw payload.

These wrap the platform read endpoints (`/rollouts/summary`, `/rollouts/heatmap`,
`/rollouts/<id>/details`). For external-eval comparison (below) and per-step
component averages, drop to raw GETs with the SDK bearer:

```python
import httpx
from benchmax import config
from benchmax.platform.credentials import platform_bearer

run_id = "..."
c = httpx.Client(
    base_url=config.platform_url(),
    headers={"Authorization": f"Bearer {platform_bearer()}"},
    timeout=60,
)
```

If `component-averages` only shows latest-step data, use scalar histories such as
`reward_stats/<component>/reward/mean` or average `details.rewards[]` yourself.

### Debug answer quality

When eval moves, read real answers rather than only scalar names:

- Bucket examples by correctness: fully correct, soft partial, wrong/empty.
<!-- rag:start -->
- For RAG, decompose low citation recall: gold cited, gold retrieved but not cited,
  and gold never retrieved. If correct answers cite retrieved valid sources, do not
  reward-shape citations just because the aggregate recall is low.
- Separate lookup from multi-hop questions; multi-hop failures often need data/model
  capability or retrieval changes, not a citation reward tweak.
<!-- rag:end -->
- Check judge leniency: partial credit for vague, hedged, or question-restating
  answers inflates pass@1 and weakens the training signal.

Background terminal monitors or polling loops do not survive a coding-agent/session
restart. After resuming a long run, re-run `status`/`scalars` rather than trusting an
old monitor.

### Controlling / debugging a run

```bash
castform stop <run-id>             # cancel a run you own
```

- `failed` early → `castform runs logs` for an env/import/reward error; fix
  `main.py`, re-`validate`, re-`launch`.
- `stalled` → the worker stopped reporting; check `runs logs` and the run URL.
- Flat/odd reward → inspect stored rollouts, then go back to **verify-environment**
  and audit the reward on transcripts before changing data or launching again.
