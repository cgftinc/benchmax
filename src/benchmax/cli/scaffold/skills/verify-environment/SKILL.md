---
name: verify-environment
description: Verify a castform env with `castform validate` and interpret the rewards/errors before spending GPU. Use after editing the env or data, and before launch-run.
---

# Verify the environment

Run `castform validate` before every launch. It runs a small **real-rollout
subset** on a cheap model (no GPU) and prints what training will actually see:
per-rollout reward values + means, the group reward, and any reward-function
errors. It's cheap — run it freely while iterating.

```bash
castform validate                 # uses run.py + train_dataset.jsonl in this dir
castform validate --examples 3    # roll out more examples
castform validate --json          # machine-readable
```

## Read the output, don't just check exit code

**Per-rollout rewards** — a table of each example's reward components + a mean.
- Are the numbers in the range you intended? **All components are summed**, so a
  component that dwarfs the others is dominating the signal.
- All-zero or constant rewards → the reward isn't discriminating; the model can't
  learn from it. Fix the reward, not the data.

**Group reward** — one of three things:
- `ok — mean …`: the group path ran and produced values. Good.
- `not run`: the env doesn't override `compute_group_reward`, or the server
  skipped it. That's expected if you only use `compute_reward` — not an error.
- `FAILED — …`: `compute_group_reward` raised or violated its contract.

**Errors are surfaced, not swallowed.** Common ones:
- *missing / bad judge API key* → an LLM-judge reward couldn't authenticate. Set
  the judge's key/url (often via the env's constructor args / env vars) and
  re-run. validate shows the error string in the per-rollout row (and the group
  row too if the judge runs inside `compute_group_reward`).
- *contract violation* → `compute_reward` must return `dict[str, float]`;
  `compute_group_reward` one finite dict per rollout.

## When it's green

If rewards look sane and there are no errors, the env is ready — go to the
**launch-run** skill. If a reward is throwing, the error string tells you what to
fix (a key, a url, a return shape); fix it and re-validate.
