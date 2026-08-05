---
name: view-progress
description: Monitor a Castform run with status, scalar, rollout, and log commands, then diagnose failures or reward drift.
---

# View progress

Start with the run ID printed by `main.py launch`:

```bash
castform runs status <run-id>
castform runs scalars <run-id> --mode eval --json
castform runs logs <run-id>
```

Use eval, not only train reward, to judge generalization. A rising train curve with
a flat or falling eval curve is overfitting, not success.

## Inspect actual rollouts

Scalar totals are not enough to validate a reward. Read transcripts and per-
component scores in the terminal or as JSON:

```bash
castform runs rollouts <run-id> --mode eval
castform runs rollout <run-id> <rollout-id>
castform runs rollout <run-id> <rollout-id> --json
```

`runs rollout` can join ground truth from local JSONL. Pass `--dataset <path>`
when the project does not use the default eval/train filenames. Use the text or
JSON output and the run page printed at launch.

When reviewing stored outcomes, distinguish a valid zero score from execution
failure using available logs and stored error fields. benchmax produces a non-
`finished` termination reason locally, but carrying that field faithfully through
the trainer and hosted rollout views is pending downstream integration; do not
claim a stored run exposes it until the platform does.

## Diagnosis

- `pending` for too long: inspect status and launch/platform logs.
- `failed` early: inspect environment imports, bundle dependencies and the first
  rollout error; fix the project and re-run validation before another launch.
- `stalled`: inspect recent activity and logs; do not infer model quality from an
  incomplete run.
- flat rewards: read correct and incorrect transcripts, then test the reward
  locally for discrimination and accidental bonus paths.
- eval peaks then declines: record the best step and verify which checkpoint is
  available before treating the final checkpoint as best.
- judge errors: fix auth/provider/runtime reliability; never reinterpret the
  zeroed failure reward as the judge's verdict.

For RAG, separate retrieval from answer quality: gold never retrieved, gold
retrieved but not cited, and correct cited answers are different failure modes.
Check source-ID canonicalization before changing reward weights.

Every `runs` read supports `--json`. Use it for programmatic comparison, but do not
build an unbounded polling loop. Re-run status and scalar reads after returning to
a long-running job.

To cancel a run owned by the current account:

```bash
castform stop <run-id>
```

Preserve the run ID, terminal state and decisive log/reward evidence in the handoff.
