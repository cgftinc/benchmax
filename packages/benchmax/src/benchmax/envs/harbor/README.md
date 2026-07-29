# harbor environments

`HarborEnv` is for existing Harbor tasks and harnesses where the Harbor agent, sandbox, and verifier own the rollout loop.

see the [benchmax architecture](../../../../README.md#architecture) for the shared environment contract and [`examples/`](../../../../../../examples/) for complete environments.

## configuration

configure `HarborEnv` with Harbor's native dataset, agent, sandbox, and verifier models. most users configure it directly instead of subclassing it.

```python
env = HarborEnv(
    dataset=DatasetConfig(name="org/dataset", ref="latest"),
    eval_ratio=0.1,
    trial=HarborTrialTemplate(
        agent=TrialAgentConfig(name="mini-swe-agent"),
        environment=TrialEnvironmentConfig(type=EnvironmentType.MODAL),
        verifier=TrialVerifierConfig(
            env={"JUDGE_MODEL": "openai/judge-model"},
        ),
    ),
    sandbox_credentials=ModalCredentials(
        token_id="...",
        token_secret="...",
    ),
)
```

install `benchmax[harbor]` while authoring. include the selected provider in the bundle dependencies, such as `harbor[modal]>=0.18,<0.19` or `harbor[daytona]>=0.18,<0.19`.

## datasets

Harbor's `DatasetConfig` supports local directories, Harbor packages, registries, and git repositories. `HarborEnv.create_dataset` resolves the source into a content-addressed snapshot and returns the same fundamental benchmax `Dataset` type used by every environment.

choose evaluation data based on how the results will be used:

| configuration | train | eval |
| --- | --- | --- |
| `eval_dataset=...` | complete primary dataset | complete explicit dataset |
| `eval_ratio=0.1` | remaining 90% | deterministic 10% holdout |
| `eval_ratio=0` | complete primary dataset | disabled |

an explicit `eval_dataset` takes precedence over `eval_ratio`. ratio splits use canonical task hashes, so source ordering and machine paths do not change membership.

when `max_examples` is set, Harbor resolves at most that many examples for the selected split.

## execution and scoring

for each rollout, `HarborEnv` gives the task to the configured agent and sandbox. the harness produces the agent output, then its verifier or RewardKit scores the trial. `HarborEnv` preserves those reward components and includes RewardKit partial credit when available.

the rollout request supplies the model endpoint and credential used by the agent. verifier and judge configuration remains separate. because agents usually run inside remote sandboxes, `HarborEnv` requests a publicly reachable model endpoint by default.

benchmax owns rollout-group concurrency. `max_concurrent_trials` can additionally cap how many Harbor sandbox trials this environment runs at once.

## sandboxes and credentials

the Harbor environment configuration selects the sandbox provider. credentials authenticate that provider but never select it.

Modal and Daytona currently use explicit `sandbox_credentials`. these values become part of the environment bundle, so use dedicated, revocable credentials rather than personal ones.

Harbor verifiers currently receive credentials as static environment variables in `TrialVerifierConfig`. runtime credential injection is available to the rollout agent, not to the verifier.

## bundled custom agents

custom agents can capture their Python source and adjacent resources so their import path does not depend on the author's checkout:

```python
agent_source = BundledAgentSource.from_directory(
    Path(__file__).parent,
    files=("my_agent.py", "helpers.py", "prompts/system.txt"),
)
agent = BundledHarborAgent(
    config=TrialAgentConfig(import_path="my_agent:MyAgent"),
    source=agent_source,
)
```

only explicitly listed files are captured. captured modules should use package-relative imports, and adjacent resources can be located from the agent module's `__file__`.

## further reading

- [benchmax architecture](../../../../README.md#architecture)
- [Harbor examples](../../../../../../examples/)
- [bundling](../../../../README.md#bundling)
