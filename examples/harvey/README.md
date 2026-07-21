# harvey

Runs [`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
— realistic legal-work tasks (memos, spreadsheets, document drafting) — through
BenchMax's generic Harbor environment, with Modal sandboxes and Harvey's own
native agent loop driving the model.

Purpose: the proof that an arbitrary third-party harness trains on this
platform without core special-casing — Harbor owns the sandbox, Harvey owns
the agent loop, and the trainer only sees rollouts.

## Getting started

```bash
uv sync            # from the benchmax workspace root
cd examples/harvey
# credentials: Modal from ~/.modal.toml, judge key from HARVEY_JUDGE_API_KEY
uv run python main.py             # validate: two real Modal trials (no GPU)
uv run python main.py launch      # train on GPUs (asks first; spends credits)
```

The integration has three explicit pieces:

- `main.py` defines `HarveyLabHarborEnv` (dataset, Modal environment,
  RewardKit verifier, custom agent) and the data/validate/launch stages.
- `harvey_agent.py` is a Harbor custom agent. It uploads the Harvey source and
  starts its runtime inside the Harbor environment.
- `harvey_runtime.py` runs Harvey's native agent loop inside that environment.
  It maps Harvey's sandbox interface onto the sandbox Harbor already owns and
  connects model calls to the rollout's OpenAI-compatible endpoint.

The environment constructor requires `ModalCredentials` and a judge API key.
Both are fixed credentials that ride in bundles; use a dedicated, revocable
key rather than a personal one:

```python
from benchmax.envs.harbor import ModalCredentials
from harvey_env import HarveyLabHarborEnv

env_args = {
    "sandbox_credentials": ModalCredentials(
        token_id="...",
        token_secret="...",
    ),
    "judge_api_key": "...",
}
env = HarveyLabHarborEnv(**env_args)
```

The agent defaults to 30 Harvey turns and a one-hour harness timeout. Override
them with `HARBOR_HARVEY_*` agent environment values when constructing a custom
`TrialAgentConfig`.

`BundledHarborAgent` carries the agent and runtime source without depending on
this checkout. Harbor gives its verifier only a static sandbox environment
variable, so the judge key is a fixed per-bundle credential, exactly like the
Modal pair. Per-request rotation would require Harbor-side support for a
runtime verifier credential.

For a local trial, the agent prepares a pinned Harvey source checkout on the
trainer host before uploading it to each sandbox. Set `HARBOR_HARVEY_ROOT`
explicitly to use an existing checkout instead. The runtime requires
`OPENAI_API_KEY` and `OPENAI_BASE_URL`; it does not discover credentials or load
local env files.

The automatic checkout is pinned to the Harvey revision verified by this
example. Set `HARBOR_HARVEY_GIT_REF` deliberately when testing a newer Harvey
revision.
