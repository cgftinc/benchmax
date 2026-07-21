# Castform environment project

This is a standalone Python 3.12 project. Treat `main.py` as the reviewed,
reproducible workflow for data preparation, validation, bundling, upload and
launch. Do not recreate those stages with hidden CLI state.

## Required loop

```bash
uv sync
uv run python main.py data
uv run python main.py validate
# stop, inspect both outcomes, and ask before spending credits
uv run python main.py launch
```

Bare `uv run python main.py` runs data then validation and stops. Never launch as
a side effect of setup, generation or validation.

Before changing a stage, load its skill:

| Work | Skill |
|---|---|
| environment, tools and rewards | `.claude/skills/design-environment/SKILL.md` |
| data preparation or references | `.claude/skills/generate-data/SKILL.md` |
| validation and baseline review | `.claude/skills/verify-environment/SKILL.md` |
| bundle, upload and launch | `.claude/skills/launch-run/SKILL.md` |
| status, curves, rollouts and logs | `.claude/skills/view-progress/SKILL.md` |

## Project contract

- `pyproject.toml` declares dependencies needed to develop and execute this
  project locally.
- `main.py` defines the environment and all workflow stages.
- JSONL files are a common BaseEnv data source, but not a universal requirement.
  A Harbor environment may resolve a Harbor package, registry or Git dataset at
  runtime instead.
- `RUNTIME_DEPENDENCIES` is the explicit list installed with the rollout bundle.
  Keep it limited to packages imported while the environment is running.

Most custom environments extend `BaseEnv`. Declare `reward_keys` as the complete
final reward shape and return exactly those keys from successful reward hooks.
Operational rollout or judge failures return the same keys with zero values, a
non-`finished` `termination_reason`, and an error log. They do not cancel or
distort siblings. Do not turn a configuration or programming error into a reward.

Local validation obtains its example through the environment's public
`create_dataset` method, then calls `validate_environment` once with exactly two
siblings. Review both outcomes; a completed zero reward is valid, while a zeroed
result with a failure termination reason is not. Validation is local-only in this
workflow.

## Bundle and launch boundary

The launch stage must remain visibly ordered:

1. validate the environment;
2. obtain explicit human confirmation that GPU training spends credits;
3. call `dump_bundle` with explicit `pip_dependencies`;
4. pass that exact `Bundle` to `upload_training_run(bundle=...)`, supplying only
   dataset splits that should be uploaded;
5. pass the uploaded paths to `TrainerClient.launch_training_run`.

BenchMax captures project-local Python modules automatically. For source from a
different project, either pass the module through `local_modules=` to capture it
or name its installed distribution in `pip_dependencies` to keep it remote.
Undeclared cross-project source fails bundling. Do not bundle raw model tokens;
use `InjectedAuth` for call-time model or judge credentials. Harbor sandbox
credentials are currently explicit constructor inputs and deserve extra care when
reviewing a bundle.

Dataset uploads are optional. Omit a split argument when the environment resolves
that data at runtime. Passing `[]` explicitly uploads an empty JSONL file.

## Reward review

- Keep correctness dominant and gate secondary bonuses when appropriate.
- Use finite, non-negative components with deliberate scales; components are
  summed by training.
- Score the committed answer, not hidden reasoning or tool output.
- Test empty, wrong, partial and correct answers before trusting a baseline.
- For LLM judges, use a fixed task reference and treat judge failure as an
  operational failure, never as a legitimate score.

<!-- rag:start -->
For the RAG template, inspect retrieval separately from answer correctness. Verify
that reference source IDs and model citations use the same canonicalization, and
test a known query against the configured corpus before validation. Keep corpus
ingestion or QA generation in `generate_data` using public `castform.rag` library
interfaces; the rollout environment should depend only on the search client and
other code it actually imports.
<!-- rag:end -->

The Castform CLI is for login, setup, doctor, guide, run inspection and run
cancellation. Project data and launches stay in scripts.
