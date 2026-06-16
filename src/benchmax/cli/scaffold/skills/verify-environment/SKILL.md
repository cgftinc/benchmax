---
name: verify-environment
description: Verify a castform env with `castform validate` and interpret the rewards/errors before spending GPU. Use after editing the env or data, and before launch-run.
---

# Verify the environment

## Fast path: validate is your baseline

`castform validate` IS the cheap baseline eval. It runs a small **real-rollout
subset** on a cheap model (no GPU) and prints what training will actually see:
per-rollout reward values + means, the group reward, and any reward-function
errors. Run it freely while iterating.

```bash
castform validate                 # uses run.py + train_dataset.jsonl in this dir
castform validate --examples 3    # roll out more examples
castform validate --json          # machine-readable
```

**A green baseline** = validate passes, rewards are sane and **vary** across
rollouts, and no reward-function errors. That's the milestone — and the decision
point:

> **Baseline is green — iterate or launch?**
> - *Iterate* — improve the reward / data / env and re-validate (still no GPU).
> - *Launch* — go to the **launch-run** skill (`castform launch` spends credits).

> A strong cheap model can score *uniformly high* on a tiny sample and trip the
> `⚠ … never varies` warning even when your reward is fine — that means the rows
> are too **easy**, not that the reward is broken. Add harder rows (generate-data's
> difficulty-filter) or sample more with `--examples N`.

## Going deeper

### Read the output, don't just check exit code

**Per-rollout rewards** — a table of each example's reward components + a mean.
- Are the numbers in the range you intended? **All components are summed**, so a
  component that dwarfs the others is dominating the signal.
- All-zero or constant rewards → the reward isn't discriminating, or the rows are
  all too easy/hard. Fix the reward, or difficulty-filter the data.

**Group reward** — one of three things:
- `ok — mean …`: the group path ran and produced values. Good.
- `not run`: the env doesn't override `compute_group_reward`, or the server
  skipped it. Expected if you only use `compute_reward` — not an error.
- `FAILED — …`: `compute_group_reward` raised or violated its contract.

**Errors are surfaced, not swallowed.** Common ones:
- *missing / bad judge API key* → an LLM-judge reward couldn't authenticate. Set
  the judge's key/url (often via the env's constructor args / env vars) and
  re-run. validate shows the error string in the per-rollout row (and the group
  row too if the judge runs inside `compute_group_reward`).
- *contract violation* → `compute_reward` must return `dict[str, float]`;
  `compute_group_reward` one finite dict per rollout.

### A held-out baseline (eval set)

`castform validate` rolls out **`train_dataset.jsonl`** by default. For a baseline
on your **held-out** rows, point `--train` at the eval file:

```bash
castform validate --train eval_dataset.jsonl --examples 10
```

`--train` is the rollout source. `--eval` is only a file path loaded for symmetry
— it is **not** rolled out remotely, so use `--train eval_dataset.jsonl` to read
the held-out set. (A full standalone `castform eval` is coming.)

### Trust the check — inject an error

If validate looks suspiciously clean, confirm it's really exercising your reward:
temporarily make `compute_reward` `raise` (or return a wrong shape) and re-run —
the error should show up in the per-rollout row. Revert once you've seen it.

When the baseline is green and errors are clear, go to the **launch-run** skill.
