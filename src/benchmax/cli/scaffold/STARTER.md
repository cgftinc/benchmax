# Get started

`castform setup` already dropped a **working** starter env here (`run.py` +
`train_dataset.jsonl` / `eval_dataset.jsonl`), so the whole chain runs right now —
cheap, no GPU.

**Every castform run is the same four steps:**

```bash
castform setup        # 1. scaffold a working env + starter data
castform data …       # 2. data — keep the starter, upload your own, or generate (rag)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

Only step 2 varies — by where the data comes from:

```bash
castform data upload mydata.jsonl                             # provide your own jsonl
castform corpus ingest ./docs && castform data qa-gen --fast # generate (rag)
```

`setup` did step 1 and the starter dataset covers step 2, so `castform validate`
(step 3) is green out of the box — try it:

```
castform validate
```

Then point your coding agent (Claude Code / Codex) at your real task. Pick the
variant that matches your goal, swap in the `<…>` placeholder, and paste it in.
The agent decomposes the goal and works through the stages — consulting the
skills in `.claude/skills/` as reference, not following them in lockstep.

## Paste one of these into your agent

**Your own task** (fully wired end-to-end today):

```
i want to start a training run to improve a model on <your task>. create a
reasonable environment with relevant tools, generate a small synthetic dataset,
run a baseline eval, review the results, and propose next steps to either
iterate or launch.
```

**RAG over your corpus:**

```
i want to start a training run to improve a model on retrieval-augmented
answering over my own corpus. connect <your corpus or data source>, create a rag
environment with retrieval tools, generate a small synthetic dataset of
questions and grounded answers, run a baseline eval, review the results, and
propose next steps to either iterate or launch.
```

**From your agent's production traces:**

```
i want to start a training run to improve a model on my own agent's tasks using
my production traces. ingest <your exported traces>, create an environment with
the relevant tools, build a small eval set from the traces, run a baseline eval
against my current model, review the results, and propose next steps to either
iterate or launch.
```

> **Generic and RAG both run end-to-end on today's CLI** (RAG's data step is
> `corpus ingest` → `data qa-gen` — see **generate-data**). **Traces** is the one
> exception: its front-half (trace ingest + eval-set build) is library-direct or
> coming. The four-step path above is identical for all three.

## The milestone: a green baseline

When `castform validate` passes on your env with sane, **varying** rewards,
you've hit a **green baseline**. That's the decision point — the agent should
stop and ask:

> **Baseline is green — iterate or launch?**
> - *Iterate* — improve the reward / data / env and re-validate (still no GPU).
> - *Launch* — `castform launch` to train on GPUs (this spends credits).

## Quick commands

| Do | Command |
|----|---------|
| Sign in | `castform login` |
| Verify env + see rewards (no GPU) | `castform validate` |
| See accepted launch args | `castform launch --list-args` |
| Launch a run (GPU) | `castform launch --set model=Qwen/Qwen3.5-4B` |
| Status / progress | `castform runs status <id>` |
| Reward curves | `castform runs scalars <id>` |
| Logs | `castform runs logs <id>` |
| Stop a run | `castform stop <id>` |
| Upload a dataset file | `castform data upload <file>` |
