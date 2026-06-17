---
name: design-environment
description: Design a castform RL environment — a BaseEnv subclass with tools and reward functions. Use when creating or editing run.py / the env for a training run.
---

# Design the environment

> The four-step path: `castform setup → data → validate → launch`. This skill
> shapes the **env** that `setup` scaffolds and `validate` (step 3) checks — see
> `GETTING_STARTED.md` for the whole chain.

The environment is a single `BaseEnv` subclass in `run.py`. It defines what the
model can do (tools), how a rollout is scored (rewards), and the system prompt.

## Fast path (to a green baseline)

`castform setup` already wrote a **working** `run.py` — a single-turn QA env whose
reward scores `1.0` when the model's final answer contains the row's
`ground_truth`, else `0.0` (a discriminating reward, not an all-zero stub). So you
can validate it immediately:

```bash
castform validate
```

Then customize it for your task — usually just three things:

1. **`system_prompt`** — what the model is told it's doing.
2. **`compute_reward`** — how a rollout is scored. Keep it **discriminating** (it
   must give different scores to better/worse answers).
3. the datasets — the **generate-data** skill.

The starter is single-turn with no tools (`list_tools` returns `[]`). That's the
right default — only reach for tools if the task genuinely needs them.

Next: **generate-data** for the datasets, then **verify-environment** to validate.

## Going deeper

### The BaseEnv shape

```python
from benchmax.envs.base_env import BaseEnv

class MyEnv(BaseEnv):
    system_prompt = "…"

    async def list_tools(self):
        return []                      # [] = single-turn, no tools

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        ...                            # only reached if list_tools is non-empty

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        # messages = full transcript; task = the dataset row (prompt, ground_truth…).
        # Return a DISCRIMINATING dict[str, float] — see Reward rules below. The
        # starter scores `correct = ground_truth in the model's final answer`.
        ...

    # optional: relative/ranking reward across a rollout group
    async def compute_group_reward(self, rollout_ids, messages_list, tasks, **kwargs):
        return [{"rank": 0.0} for _ in rollout_ids]   # one dict per rollout
```

`dataset_preprocess` is inherited: it turns a row into a prompt from the row's
`prompt` (or `messages` / `prompt_messages`) field, and exposes the whole row as
`task`. Override it only if your columns differ.

### Reading the rollout (`messages` and `task`)

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

Copy-paste — get the model's final text answer (the starter inlines exactly this;
there is **no importable `last_answer`** helper, so don't `import` one):

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

### Reward rules (these decide whether training works)

- Return **positive** scores. Negatives destabilise training.
- **Every component is summed** into one scalar — scale components so the sum
  reflects the priorities you want.
- Keep it **discriminating**: a reward that returns the same value for every
  rollout gives no gradient. If `validate` warns a component never varies, the
  reward or the data needs work (see generate-data's difficulty-filter).
- For qualitative scoring, be **comparative**: judge against `ground_truth`, or
  use `compute_group_reward` to **rank** completions within the group. Ranking is
  much more stable than an absolute LLM-judge score.
- `compute_group_reward` must return one `dict[str, float]` per rollout, all
  finite. Override it only when reward needs cross-rollout context.

### Tools / turns

- No tool need? Return `[]` from `list_tools` (single-turn). Don't add tools the
  task doesn't require.
- If you DO use tools, the env is multi-turn. Two contract rules the runtime
  enforces — break either and the rollout errors, not just the tool call:
  - `list_tools` returns **`ToolDefinition` dataclasses**
    (`from benchmax.envs.types import ToolDefinition`), **not** OpenAI
    `{"type": "function", "function": {…}}` dicts. The runtime reads `tool.name` /
    `tool.input_schema`, so a dict throws `'dict' object has no attribute 'name'`.
  - `run_tool` gets the model's call `arguments` spread as `**tool_args` and must
    **return a string** (the tool's result text). On bad input, return a guidance
    string — **don't raise** (an exception aborts the whole rollout).

  ```python
  from benchmax.envs.types import ToolDefinition

  async def list_tools(self):
      return [
          ToolDefinition(
              name="lookup",
              description="Look up the definition of a term.",
              input_schema={                       # JSON-schema for the args
                  "type": "object",
                  "properties": {"term": {"type": "string"}},
                  "required": ["term"],
              },
          )
      ]

  async def run_tool(self, rollout_id, tool_name, **tool_args):
      term = tool_args.get("term")
      if not term:                                 # guide the model, don't raise
          return "Error: `lookup` needs a `term` argument."
      return self._glossary.get(term, f"No entry for {term!r}.")
  ```

- Tools make the env multi-turn, so `max_turns` defaults to **4**, `max_tool_calls`
  to **8**. The trainer ignores any `recommended_max_*` on the env. Plan to pass the
  real limit at launch (`castform launch --set max_turns=N`); note it in `run.py` so
  it isn't forgotten.

### Companion-server envs (advanced)

If the env needs a separate server (a game/sim like Showdown), that server must
be provisioned alongside the rollout (the `SkypilotProvisioner` pattern). This is
manual today and the biggest footgun — get a single-turn, no-companion env
working first.

### Dependencies

Imports beyond benchmax must be bundled at launch: external PyPI via
`--pip <pkg>` (or `pip_dependencies=[…]`), local files are bundled from `run.py`
automatically by `castform launch` (pass `local_modules=[mod]` if calling the SDK).
