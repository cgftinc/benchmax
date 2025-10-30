## Create your env from BaseEnv

### 1. **Define the system prompt**

This helps instruct the model on how to interact with the tool and format output.

```python
SYSTEM_PROMPT = """Use the `evaluate` tool to perform any computation.
Write your final answer on the last line inside <answer>...</answer>.
"""
```

### 2. **Create a reward function**

We'll score the model 1.0 if it places the correct answer inside `<answer>...</answer>` tags:

```python
import re
from html import unescape
from pathlib import Path

def reward_func(prompt: str, completion: str, ground_truth: str, workspace: Path, **kwargs) -> float:
    m = re.search(r'<answer>(.*?)</answer>', completion, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0
    answer_text = unescape(m.group(1)).strip().lower()
    return float(answer_text == ground_truth.lower())
```

### 3. **Define your math tool**

A simple safe `eval` for math expressions:

```python
def evaluate_expression(expr: str) -> str:
    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. **Create the environment class**

Bring it all together in a subclass of `BaseEnv`:

```python
class SimpleMathEnv(BaseEnv):
    system_prompt: str = SYSTEM_PROMPT
    _reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self):
        eval_tool = ToolDefinition(
            name="evaluate",
            description="Safely evaluate a math expression like '2 + 3 * 4'.",
            input_schema={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "Math expression to evaluate.",
                    },
                },
                "required": ["expr"],
            }
        )
        self.tools: Dict[str, Tuple[ToolDefinition, Callable]] = {
            "evaluate": (eval_tool, evaluate_expression)
        }
    def dataset_preprocess(self, example: dict) -> StandardizedExample:
        return {
            "prompt": f"Question: {example['question']}\n\nWrite your answer below.",
            "ground_truth": example.get("answer", ""),
            "init_rollout_args": {}
    }

    def list_tools(self) -> List[ToolDefinition]:
        return [tool_def for tool_def, _ in self.tools.values()]

    def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        _, tool_fn = self.tools[tool_name]
        return tool_fn(**tool_args)
```

