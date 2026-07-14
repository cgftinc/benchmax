# New environment group contract tests

This directory is an isolated review area for the group-native Benchmax API.
It intentionally does not reuse the legacy environment test fixtures.

Run only this suite while the new contract is under review:

```bash
uv run pytest tests/new
```

The suite proves:

- `Environment.run_group` executes one request per rollout concurrently;
- group results stay aligned by rollout ID regardless of completion order;
- execution and reward failures never return partial groups;
- `BaseEnv` sends every attempt through its assigned OpenAI-compatible URL;
- `BaseEnv.run_rollout` computes optional individual rewards for every valid
  terminal attempt;
- group-relative scoring receives the complete set of completed rollouts;
- individual and group reward dimensions merge without silent key conflicts;
- either individual or group scoring can provide the complete reward; and
- model infrastructure failures are not silently converted into rewards.

It does not validate the existing bundled environments, real Harbor execution,
TITO collection, trainer retries, or legacy consumers. Those are follow-up
migrations after this contract and suite are approved.
