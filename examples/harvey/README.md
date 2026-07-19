# Harvey LAB

This example runs [`harveyai/lab:latest`](https://hub.harborframework.com/datasets/harveyai/lab/latest)
through BenchMax's generic Harbor environment. It uses Modal for each Harbor
sandbox and Harvey's native agent loop from the `harvey-labs` repository.

The implementation is adapted from the `gateway-harbor-prototype` trainer
integration. The important pieces are:

- `harvey_env.py` configures the dataset, Modal environment, RewardKit verifier,
  and custom agent through current `HarborEnv` types.
- `harvey_agent.py` uploads the Harvey harness and runs it against the Harbor
  task's `/workspace/documents` and `/workspace/output` directories.
- `harvey_castform_probe.py` connects Harvey's native tool loop to the rollout's
  OpenAI-compatible model endpoint.

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

The agent defaults match the prototype: 30 Harvey turns, 16,384 model tokens,
a 12,000-character tool-result cap, and a one-hour harness timeout. Override
them with `HARBOR_HARVEY_*` agent environment values when constructing a custom
`TrialAgentConfig`.

On a host checkout, `archive/harvey-labs` is used when present. A portable
BenchMax bundle carries the agent and probe source; the agent prepares the
Harvey source checkout on the trainer host before uploading it to each sandbox.
