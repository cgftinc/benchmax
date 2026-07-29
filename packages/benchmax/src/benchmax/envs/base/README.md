# base environments

`BaseEnv` is the default environment to extend when benchmax should run a simple model loop with optional tool calls.

see the [benchmax architecture](../../../../README.md#architecture) for the shared environment contract and [`examples/`](../../../../../../examples/) for complete environments.

## dataset shape

each example payload contains `prompt_messages`, which starts the model conversation. every other field is passed to the completed rollout through `example_args`.

```json
{
  "prompt_messages": [{"role": "user", "content": "22 * 12 / 3 - 23"}],
  "answer": "65"
}
```

implement `create_dataset` to turn the selected split into the fundamental `Dataset` type:

```python
def to_example(row):
    payload = {
        "prompt_messages": [{"role": "user", "content": row["question"]}],
        "answer": row["answer"],
    }
    return Example(id=canonical_example_id(payload), payload=payload)


class MathEnv(BaseEnv):
    reward_keys = ("correct",)
    max_turns = 3

    async def create_dataset(
        self,
        split,
        base_dir,
        *,
        max_examples=None,
    ):
        return JsonlDataset(
            base_dir / f"{split}.jsonl",
            row_to_example=to_example,
            max_examples=max_examples,
        )

    async def list_tools(self):
        return [...]  # add, subtract, multiply, divide

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        ...  # dispatch tool_name with tool_args

    async def compute_reward(self, rollout):
        answer = extract_completion_text(rollout.messages).strip()
        return {"correct": float(answer == rollout.example_args["answer"])}
```

`max_examples` lets validation or a small training run ask the dataset implementation to stop loading early.

## model and tool loop

`max_turns` limits model turns and is required before a rollout starts. `max_tool_calls` can separately limit the total number of tool calls.

override `list_tools` to return OpenAI-compatible function definitions and `run_tool` to execute them. environments without tools inherit the empty default.

## scoring

`compute_reward` scores one completed transcript. `compute_group_rewards` can add scores that compare or otherwise depend on the completed sibling group.

`reward_keys` declares the complete output shape. the shared environment runtime handles partial attempts and operational failures consistently; see [execution and scoring](../../../../README.md#execution-and-scoring).

for deterministic helpers, judges, rubrics, ranking, and diversity scoring, see the [rewards guide](../../rewards/README.md).

## lifecycle

override `rollout_context` to acquire and release resources for one attempt. override `aclose` to close resources owned by the environment after execution ends.

## further reading

- [benchmax architecture](../../../../README.md#architecture)
- [reward helpers](../../rewards/README.md)
- [examples](../../../../../../examples/)
