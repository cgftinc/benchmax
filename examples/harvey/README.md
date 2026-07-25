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
# Modal credentials come from ~/.modal.toml; configure the verifier as below.
uv run python main.py             # data (Harbor resolve) → validate: two real Modal trials (no launch)
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

The environment constructor requires `ModalCredentials`, a RewardKit judge
model, and the exact environment variables its provider needs. These are fixed
values that ride in bundles; use dedicated, revocable credentials with a
lifetime suitable for the complete run:

```python
from benchmax.envs.harbor import ModalCredentials
from main import HarveyLabHarborEnv

env_args = {
    "sandbox_credentials": ModalCredentials(
        token_id="...",
        token_secret="...",
    ),
    "judge_model": "anthropic/claude-sonnet-4-6",
    "verifier_env": {
        "ANTHROPIC_API_KEY": "...",
    },
}
env = HarveyLabHarborEnv(**env_args)
```

The runnable entrypoint reads the model from `HARVEY_JUDGE_MODEL`. Set
`HARVEY_VERIFIER_ENV_VARS` to a comma-separated allowlist of variable names to
copy from the launching shell into `verifier_env`. It does not discover a
provider, infer credential names, or copy one credential between providers.

To judge directly with Anthropic:

```bash
export HARVEY_JUDGE_MODEL=anthropic/claude-sonnet-4-6
export ANTHROPIC_API_KEY=<anthropic-api-key>
export HARVEY_VERIFIER_ENV_VARS=ANTHROPIC_API_KEY
```

To judge through Castform's OpenAI-compatible endpoint, create a dedicated API
key in the Castform platform rather than using a login/bootstrap token:

```bash
export HARVEY_JUDGE_MODEL=openai/gpt-5.4-nano
export OPENAI_API_KEY=<dedicated-castform-api-key>
export OPENAI_BASE_URL=https://llm.castform.dev/v1
export OPENAI_API_BASE=https://llm.castform.dev/v1
# harveyai/lab currently declares this required legacy placeholder. RewardKit
# does not use its value when HARVEY_JUDGE_MODEL selects the OpenAI adapter.
export ANTHROPIC_API_KEY=unused-for-openai-judge
export HARVEY_VERIFIER_ENV_VARS=OPENAI_API_KEY,OPENAI_BASE_URL,OPENAI_API_BASE,ANTHROPIC_API_KEY
```

`CASTFORM_AUTH_TOKEN` is a short-lived Castform CLI/bootstrap credential, not a
model-provider API key. Do not include it—or copy its value under another
name—in `HARVEY_VERIFIER_ENV_VARS` or `verifier_env`. The provider-neutral
adapter passes through the variables selected by the caller and cannot reliably
identify credentials by their values or formats.

The agent defaults to 30 Harvey turns and a one-hour harness timeout. Override
them with `HARBOR_HARVEY_*` agent environment values when constructing a custom
`TrialAgentConfig`.

`BundledHarborAgent` carries the agent and runtime source without depending on
this checkout. Harbor gives its verifier only static sandbox environment
variables, so verifier credentials are fixed per bundle, exactly like the
Modal pair. Per-request rotation or Castform `InjectedAuth` would require
Harbor-side support for a runtime verifier credential.

For a local trial, the agent prepares a pinned Harvey source checkout on the
trainer host before uploading it to each sandbox. Set `HARBOR_HARVEY_ROOT`
explicitly to use an existing checkout instead. The runtime requires
`OPENAI_API_KEY` and `OPENAI_BASE_URL`; it does not discover credentials or load
local env files.

The automatic checkout is pinned to the Harvey revision verified by this
example. Set `HARBOR_HARVEY_GIT_REF` deliberately when testing a newer Harvey
revision.
