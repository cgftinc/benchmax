---
name: design-environment
description: Design a BenchMax environment, its ordered dataset, tools, and explicit reward shape for a Castform project.
---

# Design an environment

Use `BaseEnv` for the standard OpenAI-compatible chat and tool loop. Use
`HarborEnv` when Harbor owns the complete agent/sandbox/verifier harness. Extend
`Environment` directly only for another genuinely different rollout loop.

## An environment, or no environment at all?

Build an environment only when the task needs reward-scored rollouts: the model
acts, optionally with tools, something scores each attempt, and training improves
the policy from that reward. If the task is closer to "train the model to
reproduce these input → output conversations" — supervised fine-tuning on
existing demonstrations, no reward function involved — skip the environment
entirely.

Scaffold that layout with `castform setup --template sft`. The generated
`main.py` defines no environment class; it carries `training_mode` in its
`LAUNCH_CONFIG` and works directly on `{"messages": [...]}` rows through the
`benchmax.sft` dataset boundary. Follow **generate-data**, **verify-environment**
and **launch-run** for the SFT-specific part of each stage; the environment,
reward and tool material below does not apply to it.

## Required BaseEnv shape

```python
from pathlib import Path

from benchmax.envs import BaseEnv, BaseRollout, DatasetSplit, JsonlDataset
from benchmax.rewards import extract_completion_text


class MyEnv(BaseEnv):
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

Build each `Example` with a stable ID, normally
`canonical_example_id(payload)`. `prompt_messages` is the reserved BaseEnv payload
field; all other fields become `rollout.example_args`. Put system messages in
`prompt_messages` rather than ambient module state.

`Dataset` is an ordered base class, not a cleaning pipeline. The environment owns
the runtime representation and can store lightweight references in payloads.
Preparation, cleaning and QA generation belong in the project data script.

## Reward contract

- Declare the complete final shape in `reward_keys`.
- Successful individual and group reward hooks must combine to exactly that shape.
- Return finite numbers and make correctness the dominant signal.
- Let judge, model, tool and sandbox operational failures propagate through the
  typed runtime path. BenchMax logs them and returns the declared keys all zero
  with a non-`finished` termination reason.
- Do not catch a judge failure and report it as a legitimate score.
- Programming, malformed-result and configuration errors should remain loud.

Override `compute_group_rewards` only when scoring genuinely depends on successful
siblings. Failed siblings are excluded from group-relative scoring, and a failed
group judge zeroes otherwise-successful siblings without cancelling the group.

## Optional tools

`BaseEnv` supplies no tools by default. A tool-using environment returns standard
OpenAI tool schemas from `list_tools` and dispatches them in `run_tool`:

```python
async def run_tool(self, rollout_id: str, tool_name: str, **tool_args):
    if tool_name != "lookup":
        raise ValueError(f"unknown tool: {tool_name}")
    return await self.lookup(tool_args["query"])
```

Keep clients pickle-safe. Resolve rotating model/judge credentials per call with
`InjectedAuth`; never read Castform credentials from BenchMax environment code.

## Multimodal content (SFT datasets)

A message's `content` does not have to be a plain string. A user message whose
content is a list of parts — a `text` part alongside an
`{"type": "image_url", "image_url": {"url": "data:..."}}` part — is already legal
on the **SFT dataset pathway**: those rows are validated as-is and preserved
byte-for-byte through canonicalization. Read and build that content with
`benchmax.envs.base.content`:

- `message_text` joins a message's text parts into one string;
- `content_preview` renders a truncated single-line preview;
- `iter_image_refs` walks image URLs across messages in order;
- `image_to_data_uri` turns a local path or raw bytes into a `data:` URI.

Two boundaries are worth stating plainly. Multimodal rows need a vision base
model, and trainer-side image support is not implemented yet — the client and
dataset layers preserve image content, training does not consume it. And this
skill does not cover RL-side multimodal: a `BaseEnv` transporting image content
through a rollout, or a `compute_reward` reading list-shaped content, is owned by
Harbor's own multimodal environments. Do not build a custom multimodal `BaseEnv`
here.

## Review before handoff

1. Test dataset identity and split ordering.
2. Unit-test empty, wrong, partial and correct completions against every reward key.
3. Exercise tool errors and judge errors and confirm zero rewards plus an explicit
   termination reason and log.
4. Load **verify-environment** and run the real two-sibling validation.
5. Record every remote runtime import in `RUNTIME_DEPENDENCIES` for **launch-run**.

For Harbor, require explicit `reward_keys`, sandbox credentials and the matching
provider extra in the bundle dependencies, for example
`harbor[modal]>=0.18,<0.19`.
