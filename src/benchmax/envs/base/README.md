# Base environments

`BaseEnv` provides the standard OpenAI-compatible chat loop. Tools are optional.
Implement `Environment` directly when a different component, such as Harbor,
owns the rollout loop.

Each dataset example has a JSON-object payload with one reserved field:

```json
{
  "prompt_messages": [{"role": "user", "content": "What is 2 + 2?"}],
  "answer": "4"
}
```

Everything except `prompt_messages` is available on the completed rollout as
`example_args`.

```python
from pathlib import Path

from benchmax.envs import (
    BaseEnv,
    Example,
    JsonlDataset,
    canonical_example_id,
)
from benchmax.envs.reward_helpers import extract_completion_text


def math_example(row):
    payload = {
        "prompt_messages": [{"role": "user", "content": row["question"]}],
        "answer": row["answer"],
    }
    return Example(id=canonical_example_id(payload), payload=payload)


class MathEnv(BaseEnv):
    max_turns = 1

    async def create_dataset(self, split, base_dir: Path):
        return JsonlDataset(
            base_dir / f"{split}.jsonl",
            row_to_example=math_example,
        )

    async def compute_reward(
        self,
        rollout,
    ):
        answer = extract_completion_text(rollout.messages).strip()
        return {"correct": float(answer == rollout.example_args["answer"])}
```

`run_rollout` calls `compute_reward` after the chat loop. Override
`compute_group_rewards` when scoring also depends on the completed group. The
group reward keys are merged into each rollout's individual reward map;
duplicate keys raise instead of silently overwriting a score. Either hook may
return `None`, but every final rollout must receive at least one named reward.

For tools, override `list_tools` with OpenAI-compatible tool dictionaries and
override `run_tool` to dispatch them. Environments without tools inherit the
empty default.
