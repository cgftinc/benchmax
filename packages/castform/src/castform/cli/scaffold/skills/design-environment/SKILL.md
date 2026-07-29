---
name: design-environment
description: Design a BenchMax environment, its ordered dataset, tools, and explicit reward shape for a Castform project.
---

# Design an environment

Use `BaseEnv` for the standard OpenAI-compatible chat and tool loop. Use
`HarborEnv` when Harbor owns the complete agent/sandbox/verifier harness. Extend
`Environment` directly only for another genuinely different rollout loop.

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

Keep clients pickle-safe. Use `InjectedAuth` for calls through the Castform LLM endpoint so Castform supplies the current session credential. Use explicit `StaticBearerAuth` for a user-managed external endpoint; never read Castform credentials from BenchMax environment code.

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
