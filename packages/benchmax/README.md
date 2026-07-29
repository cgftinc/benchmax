# benchmax

benchmax envs is where you define datasets, how to execute the rollout, and scoring each rollout.

for installation and project setup, start with the [main readme](../../README.md#get-started). working environments live in [`examples/`](../../examples/).

## choose an environment

all benchmax environments implement the same dataset and rollout contracts. choose the adapter based on who owns the agent loop:

| environment | use it when | what it provides |
| --- | --- | --- |
| [`BaseEnv`](src/benchmax/envs/base/README.md) | default environment to extend - runs a simple loop with the option to make tool calls | chat completions, tool dispatch, turn limits, and reward hooks |
| [`HarborEnv`](src/benchmax/envs/harbor/README.md) | you already have a Harbor task or harness | Harbor agents, sandboxes, verifiers, and RewardKit integration |
| [`Environment`](src/benchmax/envs/README.md) | extend `Environment` if you need custom behavior not covered by `BaseEnv` and `HarborEnv` | the fundamental dataset, group execution, and outcome contracts |

most custom environments should extend `BaseEnv`. most Harbor users configure `HarborEnv` directly rather than subclassing it.

## architecture

an environment defines its dataset and how a group of rollouts runs against each example.

```text
Environment
├── create_dataset(split, base_dir, max_examples)
│   └── Dataset
│       └── Example(id, payload)
└── run_group(requests)
    ├── run_rollout(request) × group_size → RolloutAttempt × group_size
    ├── adapter-specific scoring
    ├── optional group-relative scoring
    └── RolloutOutcome(rewards, termination_reason)
```

every environment follows this shape. the environment decides what an example contains, how each attempt runs, which tools are available, and how the result is scored.

## datasets

`create_dataset` receives a `train` or `eval` split and returns a fixed, ordered `Dataset` of `Example` objects.

each example contains:

- a stable id used to identify the datapoint across runs;
- an environment-owned payload consumed by its rollout implementation.

the optional `max_examples` argument limits how many examples are returned. when the data source supports it, the environment should stop loading once it reaches that limit.

`JsonlDataset`, Harbor datasets, and custom datasets all produce the same fundamental `Dataset` type. the trainer and validation flow do not depend on the source file format.

## tools

`BaseEnv` exposes OpenAI-compatible function tools through `list_tools` and executes them through `run_tool`. an environment can provide no tools, one tool, or a collection of stateful tools.

with `HarborEnv`, the Harbor agent and harness define the available tools and how they interact with the sandbox. benchmax does not convert Harbor tools into `BaseEnv` tools.

## execution and scoring

`run_group` receives multiple rollout requests for the same example, runs them concurrently, waits for all siblings, and returns one `RolloutOutcome` for each request.

every environment declares `reward_keys`, which defines the complete reward shape for every outcome. operational failures return the same keys with zero values and do not cancel successful siblings. partial attempts that reach a context, output, turn, or tool limit can still be scored.

- `BaseEnv` runs the model and tool loop, then passes the transcript and example payload to `compute_reward`. `compute_group_rewards` can score the completed sibling group.
- `HarborEnv` runs the configured Harbor agent and sandbox, then maps its verifier or RewardKit output into the declared reward shape.

### helpers

`benchmax.rewards` provides deterministic text helpers, model judges, rubrics, ranking, adaptive rubrics, and diversity scoring for `BaseEnv` and direct `Environment` implementations. see the [rewards guide](src/benchmax/rewards/README.md).

Harbor environments normally use their harness verifier and RewardKit instead of benchmax reward helpers.

## bundling

a bundle contains the environment class, its constructor arguments, the project-local source it needs, and its declared remote dependencies.

```text
environment class + constructor arguments
                 + local source
                 + dependency metadata
                           │
                           ▼
                    portable bundle
```

benchmax creates the portable artifact so the same environment can be loaded outside the author's checkout. `castform` handles uploading the bundle, validating it remotely, and using it for training.

## further reading

- [base environment guide](src/benchmax/envs/base/README.md)
- [harbor environment guide](src/benchmax/envs/harbor/README.md)
- [reward helpers](src/benchmax/rewards/README.md)
- [examples](../../examples/)
- [development instructions](../../README.md#development)

apache 2.0 © 2026 CGFT Inc.
