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
| supervised finetuning from labeled data | `.claude/skills/train-sft/SKILL.md` |

If the user already has the completions the model should imitate (labeled
conversations, transcripts, input→output pairs), that is **supervised
finetuning** — read the `train-sft` skill and skip the RL loop above entirely:
SFT has no environment, validate stage, or rewards.

## Project contract

- `pyproject.toml` declares dependencies needed to develop and execute this
  project locally.
- `main.py` defines the environment and all workflow stages.
- JSONL files are a common BaseEnv data source, but not a universal requirement.
  A Harbor environment may resolve a Harbor package, registry or Git dataset at
  runtime instead.
- `RUNTIME_DEPENDENCIES` is the explicit list installed with the rollout bundle.
  Keep it limited to packages imported while the environment is running.

Most custom environments extend `BaseEnv`. Successful reward hooks return their
named reward components. Operational rollout or judge failures return no
rewards, a non-`finished` `termination_reason`, and an error log. They do not
cancel or distort siblings. Do not turn a configuration or programming error
into a reward.

Local validation obtains its example through the environment's public
`create_dataset` method, then calls `validate_environment` once with exactly two
siblings through ephemeral tracked llm-proxy sessions. The generated validation
config shares one context budget across local and hosted execution and applies a
wall-clock backstop to the complete local lifecycle. Review both outcomes; a
completed zero reward is valid, while an empty result with a failure termination
reason is not. Hosted validation uses the exact uploaded assets that launch would
consume.

Model sampling is trainer-owned. Harness code may request only an output ceiling
with `max_tokens` or `max_completion_tokens`; validation prints a warning because
the effective cap may be lower. Do not set `temperature`, `top_p`, `top_k`,
penalties, `seed`, or `stop` in harness model kwargs. Static validation rejects
those controls and unsupported response options. Tracked local and hosted
validation also rejects sampling conflicts, changed tools, overlapping calls,
and rewritten multi-turn history. Never launch with a contract error.

## Bundle and launch boundary

The launch stage must remain visibly ordered:

1. call `dump_bundle` with explicit `pip_dependencies`;
2. pass that exact `Bundle` to `upload_assets(bundle=...)`, supplying only
   dataset splits that should be uploaded;
3. validate the exact uploaded assets, locally and in a hosted sandbox;
4. obtain explicit human confirmation that GPU training spends credits;
5. pass the same uploaded paths to `TrainerClient.launch_training_run` — the
   run trains on precisely what was validated.

benchmax captures project-local Python modules automatically. For source from a
different project, either pass the module through `local_modules=` to capture it
or name its installed distribution in `pip_dependencies` to keep it remote.
Undeclared cross-project source fails bundling. For the Castform LLM endpoint
use `InjectedAuth` so Castform supplies the current session credential. For an
external endpoint use its explicit `StaticBearerAuth`; that key is pickled into the bundle. Harbor sandbox
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
test a known query against the configured corpus before validation. Start from the
matching provider example under
https://github.com/castform-ai/benchmax/tree/main/examples. Keep corpus ingestion
or QA generation in the data stage using public `castform.rag` interfaces; the
rollout environment should depend only on its search adapter and runtime SDKs.
<!-- rag:end -->

The Castform CLI is for login, setup, doctor, guide, run inspection and run
cancellation. Project data and launches stay in scripts.
