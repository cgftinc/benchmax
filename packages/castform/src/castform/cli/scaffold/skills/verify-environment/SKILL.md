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
containing exactly two siblings of that example locally and in the hosted
sandbox. This exercises the public data materialization and deployment contract
as part of validation. Hosted validation always runs against the exact assets
that were just uploaded — the same ones a launch would train on. Keep the local and hosted rollout-model context
budget shared through `VALIDATE_CONFIG["max_context_tokens"]`; the local
wall-clock backstop is `VALIDATE_CONFIG["local_timeout_seconds"]`.

`validate_environment` first performs static model-parameter checks, then uses
tracked model sessions locally and remotely to enforce the same sampling and
multi-turn history contract as training. Review static and runtime warnings as
well as outcomes. A `max_tokens` or `max_completion_tokens` warning is allowed
when the effective cap is acceptable. Sampling conflicts, unsupported controls,
changed tools, overlapping generations, and rewritten assistant history are
errors. Do not launch while any contract error remains. If validation made only
one model call, treat the “multi-turn history was not exercised” warning as a
coverage gap for harnesses expected to loop.

## Read both outcomes

For each sibling, inspect:

- `termination_reason`;
- the complete reward mapping and total;
- evidence that the response was actually scored by the intended reward;
- any environment, tool, sandbox or judge error logs.

A successful outcome has `termination_reason == "finished"` and the reward
components produced by its scoring hooks. Its scores may legitimately all be
zero. An operational failure has a different termination reason, no rewards,
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

If the environment uses `InjectedAuth("judge")` for the Castform LLM endpoint, Castform validation binds that name to its call-time credential provider for the duration of the run. Rollout
`model_auth` and named environment bindings are independent; overriding one must
not silently override the other. A missing or unknown binding should fail visibly;
the environment must not read a platform token itself.

When the baseline is green, report both outcomes and ask whether to iterate or load
**launch-run**. Do not launch automatically.
