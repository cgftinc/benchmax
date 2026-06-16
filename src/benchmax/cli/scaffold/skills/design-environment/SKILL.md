---
name: design-environment
description: Design a castform RL environment — a BaseEnv subclass with tools and reward functions. Use when creating or editing run.py / the env for a training run.
---

# Design the environment

The environment is a single `BaseEnv` subclass in `run.py`. It defines what the
model can do (tools), how a rollout is scored (rewards), and the system prompt.

## Minimal shape

```python
from benchmax.envs.base_env import BaseEnv

class MyEnv(BaseEnv):
    system_prompt = "…"

    async def list_tools(self):
        return []                      # [] = single-turn, no tools

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        ...                            # only reached if list_tools is non-empty

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        # messages = full transcript; task = the dataset row (prompt, ground_truth, …)
        return {"quality": 0.0}        # dict[str, float]

    # optional: relative/ranking reward across a rollout group
    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        return [{"rank": 0.0} for _ in rollout_ids]   # one dict per rollout
```

`dataset_preprocess` is inherited: it turns a row into a prompt from the row's
`prompt` (or `messages` / `prompt_messages`) field, and exposes the whole row as
`task`. Override it only if your columns differ.

## Reading the rollout (`messages` and `task`)

Both reward hooks are `async`. `messages` is the full transcript as a list of
`{"role", "content"}` dicts (OpenAI chat shape):

```python
[
    {"role": "system", "content": "…"},     # your system_prompt
    {"role": "user", "content": "…"},        # the dataset prompt
    {"role": "assistant", "content": "…"},   # the model's answer
    # multi-turn (tools) appends more assistant / tool messages here
]
```

- `role` is one of `system` / `user` / `assistant` / `tool`; `content` is a
  string. The model's output is the **`assistant`** turn(s).
- `task` is the **dataset row as a dict** (e.g. `{"prompt": …, "ground_truth": …}`),
  or `None` if the env grades without per-row data — read it defensively with
  `(task or {}).get("ground_truth")`.

Copy-paste — get the model's final text answer:

```python
def last_answer(messages) -> str:
    """The model's final text answer (last assistant turn)."""
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("content"):
            return m["content"]
    return ""
```

To join *every* assistant turn instead (multi-turn rollouts), use the shipped
helper: `from benchmax.envs.reward_helpers import extract_completion_text`.

## Reward rules (these decide whether training works)

- Return **positive** scores. Negatives destabilise training.
- **Every component is summed** into one scalar — scale components so the sum
  reflects the priorities you want.
- For qualitative scoring, be **comparative**: judge against `ground_truth`, or
  use `compute_group_reward` to **rank** completions within the group. Ranking is
  much more stable than an absolute LLM-judge score.
- `compute_group_reward` must return one `dict[str, float]` per rollout, all
  finite. Override it only when reward needs cross-rollout context.

## Tools / turns

- No tool need? Return `[]` from `list_tools` (single-turn). Don't add tools the
  task doesn't require.
- If you DO use tools, the env is multi-turn — and `max_turns` defaults to **4**,
  `max_tool_calls` to **8**. The trainer ignores any `recommended_max_*` on the
  env. Plan to pass the real limit at launch (`--set max_turns=N`); note it in
  `run.py` so it isn't forgotten.

## Companion-server envs (advanced)

If the env needs a separate server (a game/sim like Showdown), that server must
be provisioned alongside the rollout (the `SkypilotProvisioner` pattern). This is
manual today and the biggest footgun — get a single-turn, no-companion env
working first.

## Dependencies

Imports beyond benchmax must be bundled at launch: external PyPI via
`--pip <pkg>` (or `pip_dependencies=[…]`), local files are bundled from `run.py`
automatically by `castform launch` (pass `local_modules=[mod]` if calling the SDK).

When the env looks right, go to the **verify-environment** skill and run
`castform validate`.
