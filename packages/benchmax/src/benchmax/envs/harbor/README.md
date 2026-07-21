# Harbor environments

`HarborEnv` runs Harbor tasks without requiring a custom BenchMax subclass.
Users configure the dataset, agent, sandbox, and verifier with Harbor's own
models:

```python
from harbor import (
    DatasetConfig,
    EnvironmentType,
    TrialAgentConfig,
    TrialEnvironmentConfig,
    TrialVerifierConfig,
)

from benchmax.envs.harbor import (
    HarborEnv,
    HarborTrialTemplate,
    ModalCredentials,
)

env = HarborEnv(
    dataset=DatasetConfig(name="org/dataset", ref="latest"),
    reward_keys=("reward", "partial_credit"),
    eval_ratio=0.1,
    trial=HarborTrialTemplate(
        agent=TrialAgentConfig(
            name="mini-swe-agent",
            model_name="openai/default",
        ),
        environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
        verifier=TrialVerifierConfig(
            # Use the variable expected by this dataset's verifier.
            env={"JUDGE_MODEL": "openai/judge-model"},
        ),
    ),
    sandbox_credentials=ModalCredentials(
        token_id="...",
        token_secret="...",
    ),
)
```

`DatasetConfig` already supports local directories, Harbor packages,
registries, and Git repositories. `HarborEnv.create_dataset()` resolves it into
a content-addressed local snapshot; normal users do not implement a dataset.

Choose evaluation configuration based on how the results will be used:

| Configuration | Train | Eval | Intended use |
| --- | --- | --- | --- |
| `eval_dataset=...` | Complete primary dataset | Complete explicit dataset | Curated or benchmark evaluation |
| No `eval_dataset`, `eval_ratio=0.1` | Remaining 90% | Deterministic 10% holdout | Convenient development signal |
| No `eval_dataset`, `eval_ratio=0` | Complete primary dataset | Disabled | Training without evaluation |

`eval_dataset` always takes precedence; `eval_ratio` is consulted only when no
explicit eval dataset exists. Ratio splitting selects the lowest canonical task
hashes, so source ordering and machine paths cannot change the split. Train and
eval are complementary views over one resolved snapshot.

The sandbox is selected explicitly by `TrialEnvironmentConfig.type` or its
`import_path`. Credentials authenticate that selection but never choose it.
Custom sandboxes normally implement Harbor's `BaseEnvironment` and use
`import_path`; they do not subclass `HarborEnv`.

For Modal environments, BenchMax defaults Harbor's `app_name` to
`harbor-benchmax`. Set `environment.kwargs["app_name"]` explicitly to group
sandboxes under a different Modal App. Non-Modal environments are unaffected.

Modal and Daytona currently accept raw, explicit `sandbox_credentials`;
`HarborEnv` does not read them from the launching shell. Review this constructor
input before creating a bundle because sandbox credential reference injection is
not implemented yet. `DaytonaCredentials` accepts
either an API key or a JWT plus organization ID, with an optional named target.
`ModalCredentials` lets the Modal client wait up to 60 seconds for API
throttling by default; set `max_throttle_wait_seconds=0` to disable that retry.

The rollout request supplies the per-attempt TITO URL and key. `HarborEnv`
injects them only into the agent configuration and leaves verifier/judge
configuration untouched. An explicit `agent.model_name` is preserved; when it
is absent, the request model becomes an OpenAI-qualified Harbor model name.
Because agents run inside remote sandboxes, `HarborEnv` declares
`requires_public_model_endpoint=True` by default so the trainer routes model
calls through a publicly reachable endpoint; pass `False` only when every
sandbox can reach the trainer network directly.
`reward_keys` must name the complete shape produced by this verifier. Harbor
does not expose that schema itself, so BenchMax requires it explicitly instead
of guessing from another rollout.

Harbor verifiers accept credentials only as static environment variables, so a
verifier credential such as a judge API key is a fixed per-bundle value in
`TrialVerifierConfig(env=...)`, exactly like the sandbox credential pair. Use a
dedicated, revocable key rather than a personal one. Per-request rotation would
require Harbor-side support for runtime verifier credentials.

Agent timeouts, context/output limits, nonzero harness exits, and sandbox,
verifier, transport, or provider failures are logged and returned with the
declared reward keys all zero. One failed trial does not cancel or distort its
siblings. Request validation and task-configuration errors still fail loudly
after siblings settle.

Successful verifier rewards must exactly match `reward_keys`. When a successful
trial also contains RewardKit's `verifier/reward-details.json`, `HarborEnv` adds
the weighted criterion score as `partial_credit` when that key was declared and
the verifier did not provide it.

Install `benchmax[harbor]` while authoring. Add the chosen Harbor provider extra
to the rollout bundle's pip dependencies, for example `harbor[modal]>=0.18,<0.19`
or `harbor[daytona]>=0.18,<0.19`.

BenchMax owns rollout-group concurrency. A caller or trainer may add retry policy
around that contract; BenchMax itself does not retry a group. Harbor's job-queue-
only `agent.n_concurrent` and `agent.concurrency_group` settings are rejected
rather than silently ignored. `max_concurrent_trials` caps how many sandbox
trials this environment runs at once (unbounded by default); use it when the
sandbox provider throttles concurrent sessions.

## Bundled custom agents

Harbor resolves custom agents from a string import path after an environment has
been deserialized. Capture custom agent code and its adjacent resources eagerly
when defining the environment so that import does not depend on the authoring
checkout:

```python
from pathlib import Path

from harbor import TrialAgentConfig

from benchmax.envs.harbor import BundledAgentSource, BundledHarborAgent

agent_source = BundledAgentSource.from_directory(
    Path(__file__).parent,
    files=("my_agent.py", "helpers.py", "prompts/system.txt"),
)
agent = BundledHarborAgent(
    config=TrialAgentConfig(import_path="my_agent:MyAgent"),
    source=agent_source,
)
```

The declared bytes, canonical relative paths, and a deterministic content ID
travel with the environment. At runtime BenchMax materializes the tree once per
process and imports it through a private content-addressed package namespace. It
does not modify `sys.path`, and different source revisions cannot claim the same
module name. The materialized tree remains available for the lifetime of the
process because agent methods may read adjacent resources after construction.
Individual Harbor trial configs contain only the resolved content-addressed
import path; they do not duplicate the captured source tree.

Modules inside a captured package must use package-relative imports, such as
`from .helpers import parse`. Top-level absolute imports such as `import helpers`
would require changing the process-wide import path and are intentionally not
supported. Adjacent resources may be located from the agent module's `__file__`.
Only explicitly listed files are captured; directories, symlinks, caches, and
undeclared checkout files are never included implicitly.
