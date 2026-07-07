---
name: verify-environment
description: Verify a castform env with `castform validate` and interpret the rewards/errors before spending GPU. Use after editing the env or data, and before launch-run.
---

# Verify the environment

This is stage 3 of the castform loop:

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

## Fast path: validate is the baseline

`castform validate` runs a small real-rollout subset on a cheap model (no GPU) and
prints a fixed scorecard. Run it freely while iterating.

```bash
castform validate                 # uses main.py + train_dataset.jsonl in this dir
castform validate --examples 3    # roll out more examples
castform validate --max-turns 7 --max-tool-calls 6     # raise the rollout budget for a 6-search env
castform validate --full-messages --json               # transcript + rewards for audits
castform validate --json          # machine-readable
```

> **Multi-turn envs: raise the budget.** Rollouts default to `max_turns=4` /
> `max_tool_calls=8`. An env whose prompt advertises more tool calls than the rollout
> budget allows is **truncated** below that — the scorecard looks flat/weak for
> no obvious reason. Validate at a budget that matches: `castform validate --max-turns M
> --max-tool-calls N`. (At launch the turn budget is `--set max_turns=M`; see
> design-environment / launch-run.)

<!-- rag:start -->
> **Search envs:** set `N = MAX_SEARCH_CALLS` and `M = N + 1` — each search is one
> turn + one tool call, and the inline answer adds a turn, not a tool call.
<!-- rag:end -->

> **Narrate the wait to the user.** Each `validate` runs **real remote rollouts
> (~30–60s each)** — a full fix-and-re-validate loop can take **10+ minutes**. Say
> what you're checking and why you're re-validating, so the user reads the pause as
> progress, not a hang. **A changed dep set** (`--pip` / `--provider`, or editing the
> env's `PIP_DEPENDENCIES`) rebuilds the rollout sandbox before any rollout streams —
> that adds a few minutes on the first validate after the change; a same-deps
> re-validate skips it (don't mistake the rebuild for a hang and start polling).

**A green baseline** = validate passes, rewards are sane and **vary**, and there
are no reward-function errors. That is the decision point:

> **Baseline is green — iterate or launch?**
> - *Iterate* — improve the reward / data / env and re-validate (still no GPU).
> - *Launch* — go to the **launch-run** skill (`castform launch` spends credits).

For judged tasks, audit real transcripts before launch. A green scorecard can hide
a redundant component or lenient judge.

<!-- rag:start -->
For RAG specifically, it can also read green while citations are scored against the
wrong source identifier — audit the citation matcher before you trust it.
<!-- rag:end -->

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

### Read the scorecard, not just the exit code

Read the card top-down:

```
─── castform validate ──────────────────────────────
  env        CustomEnv · main.py
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
    If a group-only component looks constant in an ungrouped per-example view, verify
    the actual `group reward` line before treating it as a blocker; group-relative
    rewards only have signal across sibling rollouts.
- **Bottom line** — one recommendation, and *that's the real verdict* (it keys
  off variance + errors, not just the exit code):
  - `→ GREEN baseline` — usable; iterate or launch.
  - `⚠ green, but NO training signal` — validate "passed" but every reward is
    constant: a **hollow pass**. Read the per-rollout transcript with
    `castform validate --full-messages` to see what actually happened, then fix the
    reward or data. NOT a baseline.
    <!-- rag:start -->
    For rag the search tool swallows an error into a string → all-zero rewards. Two
    usual causes: (1) a **provider** env whose SDK isn't in the sandbox — re-run with
    `castform validate --provider <name>` (or `--pip <sdk>`; see design-environment);
    (2) an unreachable/empty corpus or bad credentials. Confirm retrieval/judge
    actually work (generate-data has a direct retrieval check).
    <!-- rag:end -->
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

### Audit the reward, not just the exit code

Before launch, especially for RAG, run the built-in reward audit:

```bash
castform validate --examples 6 --reward-audit          # the audit, human-readable
castform validate --examples 6 --reward-audit --json    # same, machine-readable
```

`--reward-audit` implies `--full-messages`, so it prints, per rollout, the real
**question / gold / answer / per-component rewards**, plus a per-component table with
`avg`/`std` and a `note` that flags each component as **constant** (no gradient),
**mirrors correctness** (redundant — constant within every correctness stratum, so it
adds no signal beyond the primary), **group-scored** (N/A per-example), or **primary
(gate)**. Raise `--examples` for a sharper read (more rollouts → the strata carry more
weight). Read the completions and answer these questions:

- **Does each component discriminate?** `std > 0` is necessary but not enough; a
  component that is always `0`, always `1` (the `note` flags this as constant), or
  perfectly duplicates the primary score (flagged *mirrors correctness*) is not adding
  useful signal.
- **Is the primary answer score strict enough?** Check the partial-credit bucket.
  Verbose hedges, question restatements, or answers without the required details
  should not get soft credit just because they are topical.
- **For answer-tagged envs, missing `<answer>` means no answer.** Do not score the
  model's full reasoning as correctness when it never committed an answer.
<!-- rag:start -->
- **For RAG, inspect retrieved chunks and cited sources.** Confirm search returned
  real chunks, the model cites the identifiers it actually saw, and your citation
  matcher accepts valid id/hash/title/path variants without rewarding hallucinated
  sources. The default RAG reward (the audited shape) keeps an **ungated
  `retrieval_hit`** (citing a gold chunk in the final `<answer>` is rewarded even
  when the answer is wrong — an answer-side proxy; it does not inspect raw tool
  traffic); the validate probe reports retrieval **gold-hit@k**, so you can read
  actual retrieval quality apart from correctness.
<!-- rag:end -->
- **Gate bonuses on correctness.** Wrong answers should not earn citation,
  conciseness, or style rewards. Partial answers should get only partial secondary
  credit.

`castform validate` now retries a rollout once on a transient worker/sandbox setup
error (e.g. a missing `worker_args.json`), so a single infra flake no longer fails the
whole report. If a report still comes back failed after a worker/file error rather than
a reward error, rerun before redesigning the env — the flake may simply have recurred.
A result with `ok=false`, `remote_ran=true`, and `examples=[]` is an infra/worker-startup
failure (the rollouts were swallowed), **not** a model verdict — preserve the output and
retry; never report it as a baseline.

## Output format

After validate, report the baseline in this order:

1. **What you built** — one or two sentences on the task, reward, and tools
   (`single-turn, no tools` if `list_tools` is empty).
2. **Baseline report** — env class, model, examples, dataset, reward table
   (`avg ± std` plus total), checks, and the scorecard verdict.
3. **What it means** — name which components are live, flat, saturated, or
   erroring; explain the likely cause and the single most important finding.
4. **Next steps** — give expert guidance, not a generic menu. Name the 2-3 most
   useful fixes or confirmations before a trusted launch: env/tool fidelity,
   reward fidelity, and data quality.
   <!-- rag:start -->
   For RAG, mention whether `qa-gen --fast` should be rerun without `--fast` or
   with more samples before launch.
   <!-- rag:end -->

Launch is available when the env is faithful enough for the goal, but never
auto-launch; let the user steer.

When the baseline is green and errors are clear, go to the **launch-run** skill.
