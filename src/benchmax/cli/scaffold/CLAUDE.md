# Castform RL training project

This project trains an LLM with reinforcement learning on **castform**. You (the
coding agent) drive the whole loop with the `castform` CLI: design an environment
→ make data → **validate** (cheap, real rollouts) → **launch** (GPU) → monitor.

> Skills for each stage live in `.claude/skills/` — `design-environment`,
> `generate-data`, `verify-environment`, `launch-run`, `view-progress`. Read the
> matching skill before doing that stage.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install castform        # the CLI is the benchmax package, published as castform
castform login                 # browser sign-in; writes ~/.castform (no API key to manage)
```

For headless / CI, set `PLATFORM_API_KEY` (from `app.castform.dev/account/api-keys`)
instead of `castform login`.

## Project files (the convention `castform validate` / `launch` expect)

- `run.py` — defines your environment: a single `BaseEnv` subclass.
- `train_dataset.jsonl` / `eval_dataset.jsonl` — one JSON object per line; each
  needs at least a `prompt` (and usually a `ground_truth`).

`castform validate`/`launch` import the one `BaseEnv` subclass from `run.py` and
load those two files. Keep that layout.

## The loop

1. **Design the environment** (`design-environment` skill). A `BaseEnv` subclass
   with `list_tools` / `run_tool` / `compute_reward`, and optionally
   `compute_group_reward` for relative/ranking rewards.
2. **Make the data** (`generate-data` skill). Write `train_dataset.jsonl` /
   `eval_dataset.jsonl`. Upload a local file with `castform data upload <file>`.
3. **Validate** (`verify-environment` skill): `castform validate`. Runs a small
   real-rollout subset on a cheap model (no GPU) and prints per-rollout + group
   reward values and any reward-function errors. Do this until rewards look sane.
4. **Launch** (`launch-run` skill): `castform launch`. Validates, uploads, and
   launches a GPU run; prints the run URL.
5. **Monitor** (`view-progress` skill): `castform runs status/scalars/logs <id>`.

## Reward functions — get these right

Rewards are the training signal; robust rewards matter more than anything else.

- **Return positive scores.** Negative rewards destabilise training.
- **All reward components are SUMMED** into one scalar per rollout (the trainer
  adds the values of the dict `compute_reward` returns). Scale them so the sum is
  meaningful — a `{"correct": 1.0, "format": 0.1}` weights format at ~10%.
- **Prefer comparative rewards** for qualitative/LLM-judge scoring: compare the
  completion against `ground_truth`, or use `compute_group_reward` to *rank*
  completions within a group rather than score them absolutely. Ranking is far
  more stable than an absolute 1–10 judge score.

## Dependencies — bundle them at upload

Your env can depend on (1) external PyPI packages or (2) other local files. Both
must be passed when uploading (the trainer runs your env in an isolated image):

- external PyPI packages → `pip_dependencies=["httpx", ...]`
- local modules → `local_modules=[scoring_utils]` — pass the **imported module
  objects**, not their names as strings.

`castform launch` reads these from `--pip` and bundles your `run.py` module
automatically; if you call the SDK directly, pass them to `upload_training_run`.

## Gotchas that silently cost you (verified against the trainer)

- **`max_turns` defaults to 4, `max_tool_calls` to 8.** A multi-turn env that
  needs more will be silently truncated. The trainer does **not** consult an env's
  `recommended_max_turns` (it never passes the env class to the limit resolver) —
  so always set the limit explicitly at launch: `castform launch --set max_turns=N`.
- **The launch token budget is `max_rollout_len`** (total tokens across the whole
  rollout, all turns) — **not** `max_response_len`. The server rejects unknown
  arg names. `castform launch --list-args` shows the live, accepted set.
- **Companion-server envs** (an env that talks to a separate game/sim server, e.g.
  a Showdown-style env) need that server provisioned alongside the rollout — the
  `SkypilotProvisioner` pattern. This is manual today and is the biggest
  env-authoring footgun; see the `verify-environment` skill.

## First-party use-cases

If the task is **search/RAG** (post-training an LLM to use a search tool over a
corpus) or **traces** (training from collected production agent traces), castform
has first-party support — see `castform.com/docs/rag` and `.../traces`.
