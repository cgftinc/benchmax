# Harvey LAB

This example runs [`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
through BenchMax's generic Harbor environment. It uses Modal for each Harbor
sandbox and Harvey's native agent loop from the `harvey-labs` repository.

The integration has three explicit pieces:

- `harvey_env.py` configures the dataset, Modal environment, RewardKit verifier,
  and custom agent through current `HarborEnv` types.
- `harvey_agent.py` is a Harbor custom agent. It uploads the Harvey source and
  starts its runtime inside the Harbor environment.
- `harvey_runtime.py` runs Harvey's native agent loop inside that environment.
  It maps Harvey's sandbox interface onto the sandbox Harbor already owns and
  connects model calls to the rollout's OpenAI-compatible endpoint.

The environment constructor requires `ModalCredentials` and a judge API key:

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

The agent defaults to 30 Harvey turns, 16,384 model tokens, and a one-hour
harness timeout. Override them with `HARBOR_HARVEY_*` agent environment values
when constructing a custom `TrialAgentConfig`.

A portable BenchMax bundle carries the agent and runtime source. The agent
prepares a pinned Harvey source checkout on the trainer host before uploading
it to each sandbox. Set `HARBOR_HARVEY_ROOT` explicitly to use an existing
checkout instead. The runtime requires `OPENAI_API_KEY` and `OPENAI_BASE_URL`;
it does not discover credentials or load local env files.

The automatic checkout is pinned to the Harvey revision verified by this
example. Set `HARBOR_HARVEY_GIT_REF` deliberately when testing a newer Harvey
revision.
