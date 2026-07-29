# Get started

`castform setup` created a standalone Python 3.12 project. The project script,
not the CLI, owns the reproducible training workflow:

```bash
uv sync
uv run python main.py data       # prepare or refresh project data
uv run python main.py validate   # one local group with two siblings; no GPU
uv run python main.py launch     # validate, confirm cost, upload and launch
```

Bare `uv run python main.py` (the `all` stage) runs data preparation followed by
validation and then stops. A launch is always a separate, confirmed action.
`--force` regenerates data that already exists; `-y`/`--yes` skips launch
confirmation prompts.

Unit tests live in `tests/` next to `main.py` (a `conftest.py` there pins the
import path so `from main import ...` works). Run them with
`uv run pytest tests`, and grow them alongside the reward: cover empty, wrong,
partial and correct answers.

## Work with your agent

Point your coding agent at the real task and ask it to load the matching skill in
`.claude/skills/` before each stage.

**General environment:**

```
Build a Castform environment for <task>. Start with a small representative train
and eval set, define an explicit reward shape, validate two sibling rollouts, and
show me the result before proposing a launch.
```

**RAG environment:**

```
Build a Castform search environment over <corpus>. Keep corpus preparation in the
data stage, test retrieval and reward behavior locally, and show me validation
results before proposing a launch.
```

**SFT dataset** (`castform setup --template sft` — no environment, no reward):

```
Build a Castform SFT dataset for <task> from these demonstrations. Keep the rows
in the OpenAI fine-tuning chat format, validate them locally, and show me the
scorecard before proposing a launch.
```

Use these skills in order:

| Stage | Skill |
|---|---|
| environment and rewards | `design-environment` |
| data preparation or references | `generate-data` |
| local group validation | `verify-environment` |
| explicit GPU launch | `launch-run` |
| monitoring and diagnosis | `view-progress` |

An SFT project has no environment to design and no reward to score, so
`design-environment` only tells you to skip it. Its stages are `generate-data`
(the `messages` row format), `verify-environment` (a local, no-rollout dataset
check) and `launch-run` — read each skill's SFT section.

## A green baseline

A green baseline means both validation siblings finished, returned the declared
reward keys, and produced believable task-specific scores. A zero score can be a
valid completed result. An operational failure instead has a non-`finished`
`termination_reason`, the same reward keys all set to zero, and a corresponding
log entry.

For an SFT project there are no siblings: a green baseline is a `validate` that
exits `0` — no error-severity issues and at least one train row.

After a green baseline, decide deliberately:

- iterate on the environment, data or reward and validate again; or
- inspect the bundle dependencies and run `uv run python main.py launch`.

The seed script uploads its JSONL splits. If the environment instead resolves a
split at runtime—through Harbor, Git, or another provider—omit that split from
`upload_training_run`. Use `None`/omission for no upload; `[]` uploads an empty
JSONL deliberately.

## Useful CLI commands

| Task | Command |
|---|---|
| sign in | `castform login` |
| scaffold a project | `castform setup` |
| check local/platform setup | `castform doctor` |
| show this workflow | `castform guide` |
| list runs | `castform runs list` |
| inspect status | `castform runs status <run-id>` |
| inspect a rollout | `castform runs rollout <run-id> <rollout-id>` |
| inspect logs | `castform runs logs <run-id>` |
| cancel a run | `castform stop <run-id>` |

Data and corpus workflows are ordinary Python library calls in `main.py` or a
separate project script. Launching is likewise script-owned; there is no parallel
CLI orchestration path.
