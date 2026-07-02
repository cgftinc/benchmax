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
castform validate --max-turns 11 --max-tool-calls 10   # raise the rollout budget for a multi-turn/search env
castform validate --json          # machine-readable
```

> **Multi-turn / search envs: raise the budget.** Rollouts default to `max_turns=4` /
> `max_tool_calls=8`. An env whose prompt advertises more (a `SearchEnv` with
> `MAX_SEARCH_CALLS=10`) is **truncated** below that — the scorecard looks flat/weak for
> no obvious reason. Validate at a budget that matches: `castform validate --max-turns M
> --max-tool-calls S` where `S = MAX_SEARCH_CALLS` and `M = S + 1` (the inline answer
> adds a turn, not a tool call). (At launch the turn budget is `--set max_turns=M`; see
> design-environment / launch-run.)

> **Narrate the wait to the user.** Each `validate` runs **real remote rollouts
> (~30–60s each)** — a full fix-and-re-validate loop can take **10+ minutes**. Say
> what you're checking and why you're re-validating, so the user reads the pause as
> progress, not a hang. **A changed dep set** (`--pip` / `--provider`, or editing the
> env's `PIP_DEPENDENCIES`) rebuilds the rollout sandbox before any rollout streams —
> that adds a few minutes on the first validate after the change; a same-deps
> re-validate skips it (don't mistake the rebuild for a hang and start polling).

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
    string → all-zero rewards. Read the per-rollout transcript with
    `castform validate --full-messages` to surface the swallowed `Error:`. Two usual
    causes: (1) a **provider** env whose SDK isn't in the sandbox — re-run with
    `castform validate --provider <name>` (or `--pip <sdk>`; see design-environment);
    (2) an unreachable/empty corpus or bad credentials. NOT a baseline — confirm
    retrieval/judge actually work (generate-data has a direct retrieval check).
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

After a validate pass, report in this exact order — don't lead with numbers;
orient the user first.

**1. What you built**
- One or two sentences: the task and how the env scores it.
- **Reward components** — open with *how rollouts are scored = what the model
  optimizes*, then one terse bullet per dict key: `name` (range) — the behaviour
  it rewards. Mark the primary objective.
- **Tools** — if `list_tools` is non-empty, one bullet each: `name` — what the
  model can do with it. Otherwise "single-turn, no tools."

**2. Baseline report** — the scorecard, re-presented (don't free-form the numbers).
- **Run config** — env class · model · N examples · dataset.
- **Rewards** — a table: each component `avg ± std`, then the summed **total**.
  Flag any `std = 0`.
- **Checks** — errors / variance / group, each ✓ or ⚠ with a one-line reason.
- **Verdict** — the scorecard's bottom line (GREEN / hollow green / not passing).

**3. What it means** — tie each number to a cause. Component by component:
saturated (constant high/low), live (varies), or erroring — and for each flat one
say *why* (reward not discriminating vs rows too easy/hard vs task structurally
hard) and whether it's fine or a problem. Name the single most important finding.
Confirm it's not a hollow green (you verified discrimination via the injected-
error check).

**4. Next steps** — expert guidance, not a menu. Don't offer a launch/iterate
multiple-choice. The baseline was kept deliberately simple and fast so the user
could validate the loop in minutes — it proves the pipeline runs, not that the env
is as deep as their real task. Act as an expert: reason about what's actually thin
*here* and name the 2–3 things most worth the user's time before this is a run
they'd trust, each with why it matters for the trained model and a concrete move. A
fast green baseline is a starting point to build on, not a finished env.

Reason through these — only raise what's genuinely weak, skip what's already right:
- **Environment / tool fidelity** — does the env give the model the tools the real
  task needs? If the task only needs what's there (e.g. a search tool for a RAG
  task, and that's provided), say so and move on. If the user's ask implies more —
  real APIs/integrations a production agent would call — flag the gap and what it'd
  take to close it. The model only learns the world you simulate.
- **Reward fidelity** — you crafted the reward based on what you think the user
  wants; only they can confirm it matches what they actually care about. Surface
  what it rewards vs ignores and invite correction — the model optimizes exactly
  what you score.
- **Data quantity / quality** — the baseline data was generated for speed, not
  coverage. For a custom env you wrote the rows, so name their gaps. For RAG/traces
  the first-party generation ran lean (fewer samples, quality filters/retries/
  multi-hop off — e.g. `qa-gen --fast`) to keep the baseline quick; recommend
  regenerating with quality on and more samples for a real run.

Launch is available (`castform launch`) when the env is faithful enough for their
goal — but lead with the gaps when it isn't. Let the user steer; never auto-launch.

When the baseline is green and errors are clear, go to the **launch-run** skill.
