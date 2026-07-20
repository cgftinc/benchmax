---
name: design-environment
description: Design a custom Castform BaseEnv, its dataset examples, optional tools, and rewards.
---

# Design an environment

Use `BaseEnv` for the standard OpenAI-compatible conversation loop. Implement
the structural `Environment` protocol directly only when another harness owns
execution.

## Required shape

```python
from benchmax.envs import BaseEnv


class MyEnv(BaseEnv):
    max_turns = 1

    async def create_dataset(self, split, base_dir):
        ...

    async def compute_reward(
        self,
        rollout_id,
        messages,
        example_args,
        *,
        termination_reason,
    ):
        return {"correctness": ...}
```

Every example payload is a JSON object with one reserved field:

```json
{
  "prompt_messages": [{"role": "user", "content": "..."}],
  "ground_truth": "..."
}
```

`BaseEnv` removes `prompt_messages` and passes every other field together as
`example_args`. Put system messages directly in `prompt_messages`; BaseEnv has
no separate system-prompt configuration.

## JSONL datasets

Use `JsonlDataset` for ordinary JSONL sources. The callback returns the complete
`Example`, including stable identity:

```python
from benchmax.envs import Example, JsonlDataset, canonical_example_id


def make_example(row):
    payload = {
        "prompt_messages": [
            {"role": "user", "content": row["question"]},
        ],
        "ground_truth": row["answer"],
    }
    return Example(id=canonical_example_id(payload), payload=payload)


async def create_dataset(self, split, base_dir):
    return JsonlDataset(
        base_dir / f"{split}.jsonl",
        row_to_example=make_example,
    )
```

Do not introduce a generic row parser on BaseEnv. Each environment owns its row
semantics and decides which values define canonical identity.

## Rewards

- Score the complete transcript in `messages`.
- Use `example_args` for ground truth and other example-owned scoring data.
- `termination_reason` is tracking context; every completed attempt is scored.
- Let judge, verifier, model-client, and tool infrastructure failures raise.
- Return a non-empty `dict[str, float]` with finite values.
- Make the reward discriminating across plausible model outputs.
- Keep correctness dominant and gate secondary bonuses when appropriate.

Group-relative environments may additionally implement:

```python
async def compute_group_reward(
    self,
    rollout_ids,
    messages_list,
    example_args_list,
    termination_reasons,
):
    ...  # one reward mapping per rollout, in the same order
```

## Optional tools

Environments without tools inherit `list_tools() -> []` and do not implement
`run_tool`.

Tool-using environments return OpenAI-compatible tool dictionaries directly:

```python
async def list_tools(self):
    return [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a term.",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
    }]


async def run_tool(self, rollout_id, tool_name, **tool_args):
    ...
```

Tool results may be strings or JSON-serializable values. A raised exception is
an infrastructure failure and aborts the attempt.

## Before launch

1. Use **generate-data** to create representative train and eval JSONL files.
2. Use **verify-environment** to exercise real reward and tool edge cases.
3. Run `castform validate` and inspect reward variation and failures.
4. Launch only after the dataset identity and reward behavior are reviewed.
