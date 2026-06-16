# Get started

Paste this to your coding agent to kick off a run (edit the goal first):

---
I want to train a model on **castform** to: **‹describe your task — what should the
model get better at, and how do we judge success?›**

Use the castform skills in `.claude/skills/`. Work through them in order:
1. **design-environment** — write `run.py` (a `BaseEnv` with `compute_reward`;
   use `compute_group_reward` ranking if it fits the task).
2. **generate-data** — write `train_dataset.jsonl` + `eval_dataset.jsonl`
   (`prompt` + `ground_truth` per line).
3. **verify-environment** — run `castform validate` and iterate until the reward
   values look sane and nothing errors. (Cheap — no GPU.)
4. **launch-run** — once I confirm, `castform launch` (this spends GPU).

Start by asking me any clarifying questions about the task and how to reward it.
---

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
