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

## Define judge-backed rewards

Judge configuration is one serializable value shared by rubric, adaptive-rubric,
and semantic-diversity rewards. Authentication is resolved immediately before
each model call; bundles carry an `InjectedAuth` reference, never its token.

```python
from benchmax.auth import InjectedAuth
from benchmax.rewards import Judge, Rubric, score_rubrics

judge = Judge(
    model="judge-model",
    base_url="https://models.example/v1",
    auth=InjectedAuth("judge"),
)
rubrics = [
    Rubric("Correctness", "The answer is factually correct."),
    Rubric(
        "Fabrication",
        "The answer invents unsupported facts.",
        polarity="negative",
    ),
]

rewards = await score_rubrics(
    rollout_id,
    completion,
    ground_truth=reference,
    rubrics=rubrics,
    question=question,
    judge=judge,
)
```

`evaluate_single_rubric` and `evaluate_rubric_ranking` return typed results.
`score_rubrics`, `score_group_rubrics`, and `rank_group_rubrics` turn those
results into reward maps. Empty completions keep the declared rubric reward
shape and receive zeros without calling the judge. Invalid or out-of-set judge
output raises `JudgeError` so the environment runtime can record an operational
failure instead of trusting a fabricated score.

Adaptive rubric state is explicit and caller-owned through `RubricCache`.
Diversity backends are explicit as well: use `NgramDiversityConfig` for local
single-linkage clustering or `LLMDiversityConfig(judge=judge)` for semantic
clustering.

The rewards package follows a deep-module design:

- Callers provide domain intent; modules own prompt structure, authentication
  retries, parsing, score normalization, clustering, and cache keys.
- Related values use validated types such as `Judge`, `Rubric`, and
  `RankingAnchor` instead of parallel parameters or loose dictionaries.
- Module names describe capabilities (`prompts`, `scoring`, `deterministic`),
  not visibility or generic “helper” status.
- A new public abstraction should hide substantially more complexity than it
  adds to the interface.

## Bundle an environment

Declare remote runtime dependencies at the script boundary:

```python
from benchmax.bundle import bundle_digest, dump_bundle

bundle = dump_bundle(
    AnswerEnv,
    constructor_args={},
    pip_dependencies=["httpx>=0.28,<0.29"],
)
print(bundle_digest(bundle))
```

BenchMax automatically captures project-local Python modules reachable from the
environment. Source from a different project is never captured implicitly: pass
its module object through `local_modules=` to include it, or list its installed
distribution in `pip_dependencies` to keep it as a remote reference. External
packages are never inferred from project metadata. Dependency declarations must
be valid PEP 508 strings; BenchMax canonicalizes and stores them as an immutable,
order-independent collection in bundle metadata. Declare each distribution once,
combining its constraints and extras in that declaration; repeated targets are
rejected instead of relying on resolver-specific conflict behavior.

Keep project-local imports at module scope so source capture can see them.
BenchMax refuses method-local imports of local source, including literal
`importlib.import_module(...)` calls, because reconstructed by-value modules do
not satisfy a later Python import. Runtime-computed dynamic import names cannot
be inferred; install those modules remotely and declare their distributions in
`pip_dependencies`.

BenchMax only prepares the bundle. Uploading it and launching a hosted run belong
to the platform integration chosen by the caller. `bundle_digest` is the
artifact identity for storage and caching; it covers both the exact pickle and
canonical metadata. Execution runtimes can call `validate_bundle_compatibility`
on metadata before installing dependencies or unpickling. Python and BenchMax
versions must both match exactly, and a source digest distinguishes runtime
revisions that share a development version. Metadata also checksums the pickle,
so mixing the two files from different bundles fails before unpickling.

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
