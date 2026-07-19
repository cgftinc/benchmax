---
name: verify-environment
description: Verify a Castform environment through the script-owned validation function before spending GPU credits.
---

# Verify the environment

Validation is part of the project script, not a Castform CLI command. Inspect
`main.py` before running it and confirm its `validate()` stage calls
`castform.validate_environment`.

Run:

```bash
uv run python main.py validate
```

The local check always executes one real `Environment.run_group` call containing
exactly two sibling rollouts for the same example. Read both reward mappings and
confirm that:

- both rollouts terminate successfully;
- every expected reward key is present and finite;
- rewards respond to meaningful output differences;
- group-relative rewards do not depend on global or cross-group state.

The validation model and whether hosted validation is requested live in
`VALIDATE_CONFIG` in `main.py`. `include_remote=False` means local only.
`include_remote=True` means local first, then hosted validation; hosted group
validation is currently unavailable until rollout-service supports the same
group-native contract.

If the environment uses an LLM judge, its auth must be declared in the
environment, for example `InjectedAuth("judge")`. A missing declaration is an
environment bug. The trainer/runtime supplies the concrete rotating provider;
the environment must not read Castform credentials itself.

For a held-out check, change the dataset split/example selected by the script,
then rerun the same command. Keep validation configuration in source so another
person can reproduce the result without reconstructing CLI flags.

Do not launch when validation raises, returns malformed reward maps, or produces
only constant rewards without a task-specific reason.
