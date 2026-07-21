# BenchMax

BenchMax is a platform-independent runtime for defining and running reinforcement-
learning environments. It owns the execution contract, ordered datasets, stable
example identities, reward helpers and bundling. It has no dependency on any
training platform.

Python 3.12 is required.

## Define an environment

Most environments extend `BaseEnv` and declare their complete reward shape:

```python
from pathlib import Path

from benchmax.envs import BaseEnv, BaseRollout, DatasetSplit, JsonlDataset
from benchmax.rewards import extract_completion_text


class AnswerEnv(BaseEnv):
    reward_keys = ("correct",)
    max_turns = 1

    async def create_dataset(
        self, split: DatasetSplit, base_dir: Path
    ) -> JsonlDataset:
        return JsonlDataset(base_dir / f"{split}.jsonl", row_to_example=...)

    async def compute_reward(self, rollout: BaseRollout) -> dict[str, float]:
        answer = extract_completion_text(rollout.messages)
        return {"correct": float(answer == rollout.example_args["answer"])}
```

`Dataset` is a fixed-order base class. Concrete datasets provide stable
`Example` objects and may keep lightweight references in each payload instead of
materializing large data in memory.

`reward_keys` is authoritative. A successful rollout must return exactly those
keys. Operational failures keep the same shape with every value set to zero,
record the reason in `termination_reason`, and are logged without cancelling
successful siblings. Configuration and programming errors still fail loudly after
the sibling group settles.

See the [BaseEnv guide](src/benchmax/envs/base/README.md) and
[Harbor adapter guide](src/benchmax/envs/harbor/README.md).

## Bundle an environment

Declare remote runtime dependencies at the script boundary:

```python
from benchmax.bundle import dump_bundle

bundle = dump_bundle(
    AnswerEnv,
    constructor_args={},
    pip_dependencies=["httpx>=0.28,<0.29"],
)
```

BenchMax automatically captures project-local Python modules reachable from the
environment. Source from a different project is never captured implicitly: pass
its module object through `local_modules=` to include it, or list its installed
distribution in `pip_dependencies` to keep it as a remote reference. External
packages are never inferred from project metadata.

BenchMax only prepares the bundle. Uploading it and launching a hosted run belong
to the platform integration chosen by the caller.

## Breaking-version policy

This reshuffle intentionally removes the old `benchmax.rubrics`,
`benchmax.envs.reward_helpers`, `benchmax.prompts`, and `FrozenDataset` import
surfaces. There are no compatibility aliases. Rebuild environments and bundles
against the new `benchmax.rewards` and `Dataset` APIs; a runtime that must execute
an older stored bundle must remain pinned to the older BenchMax version.

Rubric judges now enforce their declared score set. A binary rubric accepts only
`0` or `1`; include intermediate values explicitly in `score_map`, or use a
ranking reward when continuous relative scores are intended. An out-of-set judge
score is an operational `judge_error`, not a trusted reward.

## Development

```bash
uv run --project packages/benchmax pytest \
  -c packages/benchmax/pytest.ini packages/benchmax/tests
```

Apache 2.0 © 2026 CGFT Inc.
