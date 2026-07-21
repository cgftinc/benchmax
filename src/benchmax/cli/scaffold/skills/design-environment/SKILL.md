---
name: design-environment
description: Design a custom Castform BaseEnv, its dataset examples, optional tools, and rewards.
---

# Design an environment

Use `BaseEnv` for the standard OpenAI-compatible conversation loop. Implement
the structural `Environment` protocol directly only when another harness owns
execution.

## RL env, or no env at all?

Build a `BaseEnv` only when the task needs reward-scored rollouts: the model
acts (optionally with tools), something scores each attempt, and training
improves the policy from that reward. If the task is closer to "train the model
to reproduce these input → output conversations" — supervised fine-tuning (SFT)
on existing demonstrations, no reward function involved — skip the env
entirely. Set `TRAINING_MODE = "sft"` at the top of `main.py` instead of
defining a `BaseEnv` subclass; `castform` reads that marker before it goes
looking for an env class, and a project can't mix the two (a `TRAINING_MODE =
"sft"` file with a `BaseEnv` subclass in it is a loud error, not a fallback).
Scaffold the SFT layout directly with `castform setup --template sft`, then
follow **generate-data**, **verify-environment**, and **launch-run** for the
SFT-specific parts of each stage.

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

## Multimodal content (SFT datasets)

A message's `content` does not have to be a plain string. A user message with
`content: [{"type": "image_url", "image_url": {"url": "data:..."}}, {"type":
"text", "text": "..."}]` is already legal — this is the **SFT dataset pathway**
(env-less `TRAINING_MODE = "sft"` rows, see generate-data), where such rows are
validated as-is and preserved through canonicalization. Read and render that
content with `benchmax.envs.base.content`: `message_text` for the joined text,
`content_preview` for a truncated single-line preview, `iter_image_refs` to walk
image URLs in order, and `image_to_data_uri` to turn a local path or raw bytes
into a `data:` URI for a row.

This skill and the SFT dataset pathway do not cover RL-side multimodal — a
`BaseEnv` transporting image content through a rollout, or a `compute_reward`
reading list-shaped content. That is owned natively by harbor-proper's own
multimodal environments (e.g. its Geo3K env); do not build a custom multimodal
`BaseEnv` here — see harbor-proper's own docs for that path.

## Before launch

1. Use **generate-data** to create representative train and eval JSONL files.
2. Use **verify-environment** to exercise real reward and tool edge cases.
3. Run `castform validate` and inspect reward variation and failures.
4. Launch only after the dataset identity and reward behavior are reviewed.
