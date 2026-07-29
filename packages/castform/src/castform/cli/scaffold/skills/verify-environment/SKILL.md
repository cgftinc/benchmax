---
name: verify-environment
description: Run and inspect the script-owned local two-sibling validation before spending GPU credits.
---

# Verify the environment

Inspect `main.py`, then run:

```bash
uv run python main.py validate
```

The validation stage must first obtain its example through
`env.create_dataset("train", Path("."))`, then call
`castform.validate_environment` once for one real `Environment.run_group`
containing exactly two siblings of that example. This exercises the public data
materialization contract as part of validation. Keep `include_remote=False`.

## Read both outcomes

For each sibling, inspect:

- `termination_reason`;
- the complete reward mapping and total;
- evidence that the response was actually scored by the intended reward;
- any environment, tool, sandbox or judge error logs.

A successful outcome has `termination_reason == "finished"` and exactly the
environment's declared `reward_keys`. Its scores may legitimately all be zero. An
operational failure has a different termination reason, the same keys all zero,
and a visible log entry. It must not cancel the other sibling.

Do not call the baseline green when:

- an outcome failed, even if its reward mapping looks structurally valid;
- rewards are malformed, non-finite or missing declared keys;
- the reward is constant for reasons the task does not justify;
- the judge or verifier failure was mistaken for a valid zero score;
- group-relative scoring depends on failed siblings or cross-group state.

## Targeted checks

Before launch, add unit tests for empty, wrong, partial and correct answers. Put
them in `tests/` next to `main.py` (its `conftest.py` pins the import path so
`from main import ...` resolves) and run `uv run pytest tests`. Exercise
tool exceptions and judge exceptions and assert the failure termination reason,
zeroed declared shape and log message. For a group-relative reward, verify that one
failed sibling does not alter successful siblings' scoring inputs.

If the environment uses `InjectedAuth("judge")`, Castform validation binds that
name to its call-time credential provider for the duration of the run. Rollout
`model_auth` and named environment bindings are independent; overriding one must
not silently override the other. A missing or unknown binding should fail visibly;
the environment must not read a platform token itself.

## SFT validate: local, no rollouts

An env-less SFT project (see design-environment) runs the same command:

```bash
uv run python main.py validate
```

but takes a different path entirely. It loads `train.jsonl` and, when present,
`eval.jsonl` through `benchmax.sft.load_sft_dataset`, then calls
`validate_sft_dataset` on the pair. Nothing rolls out: no remote calls, no LLM,
no GPU, so it is typically fast even on a large file and safe to run on every
data edit. There is no reward table and no "rewards vary" check — row and schema
correctness is the whole gate.

The scorecard reports train/eval row counts, a char/4-heuristic token-length
summary, masking usage when any row carries a per-message `weight`, and each
issue as `path:line  message` tagged `✗` (error, gate-failing) or `⚠` (notice,
informational). The stage exits `0` only when the report is `ok`: no
error-severity issues and at least one train row. A missing or empty train
dataset is an error; a missing eval dataset is not — a train-only SFT project is
a legitimate state.

`VALIDATE_CONFIG` in `main.py` holds the dataset knobs — `max_seq_len` for the
token-budget notice and `max_row_bytes` for the per-row size notice. Both are
notices, not errors: they tell you a row is likely to be truncated or rejected
later without failing the gate on your behalf.

When the baseline is green, report both outcomes and ask whether to iterate or load
**launch-run**. Do not launch automatically.
