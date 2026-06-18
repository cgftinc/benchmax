# Castform RL training project

This project trains an LLM with reinforcement learning on **castform**. You (the
coding agent) drive the whole loop with the `castform` CLI: design an environment
→ make data → reach a **baseline** (`castform validate` — cheap, real rollouts) →
review → **iterate or launch** (GPU) → monitor.

**Every castform run is the same four steps:**

```bash
castform setup        # 1. scaffold agent skills + project guides
castform data …       # 2. data — write your own jsonl, or generate (rag/traces)
castform validate     # 3. validate the env — baseline on real rollouts, cheap, no GPU
castform launch       # 4. launch — train on GPUs (spends credits)
```

`castform setup` does **not** write `run.py` — you create it: a `BaseEnv`
subclass, following the **design-environment** skill (the example code lives in
the skills, not a scaffolded file). Write your datasets too, then `castform
validate`. The green baseline is the milestone: when validate passes with sane,
**varying** rewards, stop and decide **iterate or launch**.

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
   `compute_group_reward` for relative/ranking rewards. Tools are optional — return
   `[]` from `list_tools` for single-turn tasks. (RAG/traces tasks connect/ingest
   the source first — see `generate-data`.)
2. **Make the data** (`generate-data` skill). Write `train_dataset.jsonl` /
   `eval_dataset.jsonl`. Upload a local file with `castform data upload <file>`.
3. **Baseline** (`verify-environment` skill): `castform validate`. Runs a small
   real-rollout subset on a cheap model (no GPU) and prints a fixed **scorecard** —
   reward `avg`/`std`, pass/fail checks, and a one-line verdict. A green baseline =
   validate passes with sane, **varying** rewards.
4. **Review & decide** — the **iterate-or-launch** point. Read the rewards: if
   they don't discriminate (the constant-reward warning, or every rollout scores
   alike), the rows may be too easy/hard for the model — fix the reward or data
   and re-validate. Otherwise, launch. **Keep the user posted while you iterate** —
   narrate what you're changing and why you're re-validating. Each `validate` runs
   **real remote rollouts (~30–60s each)**, so a full fix-and-re-validate loop can
   take **10+ minutes**; tell the user the wait is expected, not a hang.
5. **Launch** (`launch-run` skill): `castform launch`. Validates, uploads, and
   launches a GPU run; prints the run URL. (Spends credits.)
6. **Monitor** (`view-progress` skill): `castform runs status/scalars/logs <id>`.

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
  needs more is silently truncated, and the trainer does **not** consult an env's
  `recommended_max_*` (it never passes the env class to the limit resolver). Set the
  budget explicitly: `castform validate --max-turns N --max-tool-calls N` (both
  settable) and `castform launch --set max_turns=N`. ⚠ At launch `max_tool_calls` is
  **not** a `--set` knob (stays 8), so a SearchEnv with `MAX_SEARCH_CALLS=10` is capped
  at 8 tool calls in training — keep `MAX_SEARCH_CALLS` ≤ 8 unless `--list-args` shows a
  higher cap.
- **The launch token budget is `max_rollout_len`** (total tokens across the whole
  rollout, all turns) — **not** `max_response_len`. The server rejects unknown
  arg names. `castform launch --list-args` shows the live, accepted set. Set it
  generously: a rollout that hits the budget is truncated and **dropped from the
  loss**, so too small a value silently wastes rollouts.
- **Companion-server envs** (an env that talks to a separate game/sim server, e.g.
  a Showdown-style env) need that server provisioned alongside the rollout — the
  `SkypilotProvisioner` pattern. This is manual today and is the biggest
  env-authoring footgun; see the `verify-environment` skill.

## First-party use-cases

If the task is **search/RAG** (post-training an LLM to use a search tool over a
corpus) or **traces** (training from collected production agent traces), castform
has first-party support — see `castform.com/docs/rag` and `.../traces`.
