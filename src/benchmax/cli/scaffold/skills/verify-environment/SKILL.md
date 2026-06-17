---
name: verify-environment
description: Verify a castform env with `castform validate` and interpret the rewards/errors before spending GPU. Use after editing the env or data, and before launch-run.
---

# Verify the environment

This skill is the **validate** step of the path every run follows:

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

## Fast path: validate is your baseline

`castform validate` IS the cheap baseline eval. It runs a small **real-rollout
subset** on a cheap model (no GPU) and prints a fixed **scorecard** — the same
shape every run — so you read it the same way each time. Run it freely while
iterating.

```bash
castform validate                 # uses run.py + train_dataset.jsonl in this dir
castform validate --examples 3    # roll out more examples
castform validate --json          # machine-readable
```

> **Narrate the wait to the user.** Each `validate` runs **real remote rollouts
> (~30–60s each)** — a full fix-and-re-validate loop can take **10+ minutes**. Say
> what you're checking and why you're re-validating, so the user reads the pause as
> progress, not a hang.

**A green baseline** = validate passes, rewards are sane and **vary** across
rollouts, and no reward-function errors. That's the milestone — and the decision
point:

> **Baseline is green — iterate or launch?**
> - *Iterate* — improve the reward / data / env and re-validate (still no GPU).
> - *Launch* — go to the **launch-run** skill (`castform launch` spends credits).

> A strong cheap model can score *uniformly high* on a tiny sample and trip the
> `⚠ rewards DON'T vary` check even when your reward is fine — that means the rows
> are too **easy**, not that the reward is broken. Add harder rows (generate-data's
> difficulty-filter) or sample more with `--examples N`.
>
> **`validate` rolls out the FIRST N rows in file order** (not a random sample), so at
> `--examples 2` the variance check only sees rows 0–1 — put differing-difficulty rows
> first (or interleave them) so a varied dataset actually shows variance. If the cheap
> eval model simply aces the whole task (pure arithmetic, lookups), no honest row will
> vary: that's fine — verify the reward discriminates via the injected-error check
> below, try a tougher `--model`, and treat a constant-but-verified reward as launchable.

## Going deeper

### Read the scorecard, don't just check the exit code

`validate` prints the same card every run — read it top-down:

```
─── castform validate ──────────────────────────────
  env        CustomEnv · run.py
  model      gpt-5.4-nano  (cheap eval, no GPU)
  rollouts   2 examples · train_dataset.jsonl

  reward component       avg      std
  correct                0.5      0.5
  ───────────────────────────────────
  total reward           0.5

  checks
  ✓  no reward errors             2/2 rollouts ok
  ✓  rewards vary across rollouts
  ·  group reward                 not run (no compute_group_reward)

  ✓ validate passed
  → GREEN baseline — iterate (improve reward/data) or launch.
```

- **Reward table** — each component's `avg` and `std` across the sampled rollouts,
  plus a summed `total`. **All components are summed** into the training signal, so
  a component that dwarfs the others dominates it. A `std` of `0` = that component
  never varied — no gradient.
- **checks** — three glanceable lines:
  - `no reward errors` / `⚠ reward errors` — a reward fn raised; the error string
    is listed underneath. This **fails** validate.
  - `rewards vary across rollouts` / `⚠ rewards DON'T vary` (every component
    constant) / `⚠ some components constant` (lists which). Constant = the reward
    isn't discriminating, or the rows are all too easy/hard.
  - `group reward` — `mean …` (ran), `not run` (no `compute_group_reward`, or the
    server skipped it — expected, not an error), or `⚠ FAILED — …` (it raised or
    broke its contract).
- **the bottom line** — one recommendation, and *that's the real verdict* (it keys
  off variance + errors, not just the exit code):
  - `→ GREEN baseline` — usable; iterate or launch.
  - `⚠ green, but NO training signal` — validate "passed" but every reward is
    constant: a **hollow pass**. For rag the search tool swallowed an error into a
    string → all-zero rewards. Two usual causes: (1) a **provider** env whose SDK
    isn't in the sandbox — re-run with `castform validate --pip <provider>` (see
    design-environment); (2) an unreachable/empty corpus or bad credentials. NOT a
    baseline — read the transcript and confirm retrieval/judge actually work
    (generate-data has a direct retrieval check).
  - `→ NOT passing` — a reward fn errored; fix it and re-validate.

**Common reward errors** (shown under `⚠ reward errors`):
- *missing / bad judge API key* → an LLM-judge reward couldn't authenticate. Set
  the judge's key/url (often via the env's constructor args / env vars) and re-run.
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
the error should surface under the `⚠ reward errors` check. Revert once you've
seen it.

## Output format — report the baseline the same way every time

After a validate pass, summarise it for the user in this fixed shape (don't
free-form it — the scorecard already standardised the numbers):

- **Run config** — env (`run.py` class) · model · N examples · dataset.
- **Rewards** — the component `avg ± std` + total; call out any `std = 0`.
- **Checks** — errors / variance / group, each ✓ or ⚠ with the one-line reason.
- **Recommendation** — exactly one: **iterate** (what you'd change and why) or
  **launch** (go to launch-run). For a hollow green, say so plainly and stop —
  it's not a baseline.

When the baseline is green and errors are clear, go to the **launch-run** skill.
