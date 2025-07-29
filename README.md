<picture>
  <img alt="Benchmax" src="./static/benchmax.png"  width="full">
</picture>

## benchmax: Framework-Agnostic RL Environments for LLM Fine-Tuning
*A lightweight, training-framework agnostic library for defining, running, and parallelizing environments, to fine-tune OSS LLMs with reinforcement learning.*
<div align="center">
</div>
<div id="badges" align="center">
  <a href="https://cgft.io">
    <img src="https://img.shields.io/badge/cgft.io-blue?style=for-the-badge" alt="Website"/>
  </a>
  <a href="https://x.com/cgftlabs">
    <img src="https://img.shields.io/badge/Follow @cgftlabs-black?style=for-the-badge&logo=X&logoColor=white" alt="@cgftlabs"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/girishbarca/benchmax/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
</div>

## Overview

`benchmax` comes with:

- A collection of ready-to-use reinforcement learning (RL) environments for LLM fine-tuning ranging from multi-hop search to spreadsheet manipulation to CRM agents
- An easy to define, compose, and parallelize your own environments, including leveraging the existing ecosystem of MCP servers
- Built-in integrations with popular RL training libraries (verl, verifiers, etc.). `benchmax` is trainer-agnostic by design

Define your environment as:

1. A **toolset** (LLM calls, external APIs, calculators, MCPs, etc.).
2. **Output parsing** logic to extract structured observations.
3. **Reward functions** to score model outputs.

Rollout management, parallel execution, etc. comes out of the box.

⭐ Star our repository to show your support!

## 💡 Core Features

**Built-in examples & templates**

Get started with ready to use recipes, from Wikipedia search to spreadsheet manipulation. Easy to copy, customize, and extend. And yes, more are on the way.

**Trainer Integrations**

Use your own trainer or training framework - no lock-in. `benchmax` is already integrated into verl and verifiers, with more integrations (SkyRL, etc.) coming soon!

**MCP Support**
Tap into the growing MCP ecosystem and integrate them as tools within your environments.

**Parallel execution & state management**

- Local multi‐process pool
- State is isolated across roll-outs (e.g. editing files on local filesystem, etc.)
- Multi-Node Parallelization (Coming soon!)

## 📘 Quickstart

**Example: Math Question Answering with a Calculator MCP**

**verl** is a training framework `benchmax` is currently integrated with. Use our ***verl*** integration to RL finetune Qwen-3 to do math using a calculator MCP (https://github.com/githejie/mcp-server-calculator). The environment is defined at `benchmax.envs.math.math_env.MathEnv`

1. **Installation**

    `pip install benchmax[verl]`

    \* Note that benchmax installs our verl fork (temporary until [PR gets merged](https://github.com/volcengine/verl/pull/2792))

1. **Prepare the dataset**
    
    ```bash
    python benchmax/adapters/verl/benchmax_data_process.py \
      --local_dir ~/data/math \
      --dataset_name dawidmt/arithmetic50 \
      --env_path benchmax.envs.math.math_env.MathEnv
    ```
    
2. **Run training**
    
    ```bash
    sh examples/verl/run_qwen2.5-3b_benchmax_math.sh
    ```

This math environment is just a quick example. Explore some of the more complex environments like `excel`, `crm` in `benchmax/envs`.

## 🌐 Creating & Training with Environments

### What is an environment?

An environment consists of:

- A list of tools that an LLM can call
- A list of reward functions that evaluate the quality & correctness of the model's final output.

We also support MCP servers natively, allowing you to easily leverage the many servers built by the community.

### Pre-built environments

Ready-to-use environments with pre-configured tools and reward functions.

- [CRM](benchmax/envs/crm/README.md)
- [Excel](benchmax/envs/excel/README.md) 
- [Math](benchmax/envs/math/README.md)
- [Wikipedia](benchmax/envs/wikipedia/README.md)

### How do I create a custom environment?

<details>
<summary>With existing MCP Servers</summary>
    
To create a custom environment using an MCP server (like a calculator, browser, or spreadsheet), you can extend `LocalMCPEnv`. Here's a quick step-by-step guide using `benchmax.envs.math.math_env.MathEnv` as an example.

### 1. **Define a System Prompt**

This prompt guides the LLM’s behavior. It can include any instruction, such as how to format the answer or when to use tools.

```python
SYSTEM_PROMPT = """Please use the tools provided to do any computation.
Write your complete answer on the final line only, within the xml tags <answer></answer>.
"""
```

### 2. **Configure MCP Server(s)**

Define the MCP servers to be launched. You can configure one or more:

```python
MCP_CONFIG = """
{
  "mcpServers": {
    "server-name": {
      "command": "uvx",
      "args": ["mcp_server_calculator"]
    }
  }
}
"""
```

### 3. **Write a Reward Function**

The reward function evaluates how "correct" the model's output is, based on structured output. Here’s a simple XML-based example:

Note that `**kwargs` contains all the other fields in your dataset, so feel free to use them in `reward_func` calculations.

```python
def reward_func(prompt, completion, ground_truth, workspace, **kwargs):
    m = re.search(r'<answer>(.*?)</answer>', completion, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0
    answer_text = unescape(m.group(1)).strip().lower()
    return float(ground_truth.lower() == answer_text)
```

### 4. Define **`dataset_preprocess`**

If your dataset is not already standardized, implement this method to convert a raw example into a standardized one with:

- `"prompt"`: A fully constructed string prompt.
- `"ground_truth"`: A known correct output (optional depending on reward).
- `"init_rollout_args"`: Arguments needed to initialize a rollout.

Example for our math task:

```python
def dataset_preprocess(self, example: dict) -> StandardizedExample:
    return StandardizedExample(
        prompt=example.get("task", ""),
        ground_truth=example.get("answer", ""),
        init_rollout_args={}
    )
```

<details>
<summary>Notes on init_rollout_args</summary>
The `init_rollout_args` dictionary is passed from `dataset_preprocess()` to your environment's `init_rollout()` method. It is used to initialize any **per-example files, resources, or execution context** needed before a rollout begins.

Common use cases include:

- **Input files**: For environments that manipulate files like spreadsheets, images, or databases, pass the necessary file paths.
- **Version control**: For code-related tasks, you might pass a `commit_id` to check out the correct code state.
- **Task-specific settings**: Pass metadata like cell ranges, task IDs, or execution flags.

Example:

```python
# Inside dataset_preprocess
return {
    "prompt": "...",
    "ground_truth": "...",
    "init_rollout_args": {
        "spreadsheet_path": "/path/to/1_001_input.xlsx"
    }
}
```

Then in your `init_rollout()` method:

```python
def init_rollout(self, rollout_id: str, **rollout_args):
    spreadsheet_path = rollout_args["spreadsheet_path"]
    workspace = self.get_rollout_workspace(rollout_id)

    # Copy the input file into the rollout's workspace
    shutil.copy(spreadsheet_path, workspace / Path(spreadsheet_path).name)
```

This pattern ensures each rollout starts with the correct inputs and configuration.
</details>
    

### 5. **Extend `LocalMCPEnv`**

Now bring everything together into a custom environment class:

```python
from envs.local_mcp_env import LocalMCPEnv
from typing import List

class MathEnv(LocalMCPEnv):
    """Environment for math problems, using local MCP tools."""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self, **kwargs):
        super().__init__(MCP_CONFIG)
    
    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("task", ""),
            ground_truth=example.get("answer", ""),
            init_rollout_args={}
        )
```

You're done! This environment is now compatible with `benchmax` and can be plugged into any compatible RL trainer.
</details>
<details>
<summary>Extend BaseEnv</summary>
If you don’t need MCP servers, you can build a environment from scratch by extending `BaseEnv` directly. Here's how to make a minimal math environment with a single tool: an arithmetic evaluator.

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
</details>

### How about more complex environments?

- Check out our excel spreadsheet RL environment: `benchmax.envs.excel.excel_env.ExcelEnv`

### How do I use an environment with my preferred RL Trainer?

We currently have integrations with both verifiers and verl. More incoming!

[`benchmax` environments with verl](/examples/verl/README.md)

[`benchmax` environments with verifiers](/examples/verifiers/README.md)

### I want a specific environment

Open an issue and tag us & we will look into building you one!

---

## 🎯 Motivation

- **Modularity and Simplicity**:
    
    We set out to build a lightweight, modular system for defining RL environments—breaking them down into simple, composable parts: tools, tool output parsing, and reward functions.
    
    The goal’s to make it easy for software engineers to build and experiment with RL environments without needing deep RL expertise.
    
- **Trainer Integrations**:
    
    There’s been lots of new RL training frameworks popping up (e.g., numerous forks of verl) & we expect this to continue. They are often tightly coupled with specific environments, leading to fragmentation and limited compatibility. 
    
    We are building `benchmax` as a standalone library with integrations to these different training frameworks & as an easy way for new frameworks to tap into an existing pool of environments. We're already integrated with verl and verifiers. More integrations (e.g. SkyRL) coming soon!
    
- **Task Recipes and Ideas**:
    
    We want `benchmax` to be a living library of reusable, RL-compatible task recipes, ready to inspire and extend beyond the usual suspects like math and coding. We aim to support more real-world workflows, including open-ended and long-horizon tasks.
    
- **Parallelization and Cloud Compatibility**:
    - Enable efficient parallelization with maintained statefulness between rollouts.
    - Facilitate easy deployment and scalability in cloud environments.
- **MCP as a first class citizen**:
    
    There has been an explosion of MCP servers/tools built out for usecases ranging from browser use to excel to game creation.`benchmax` allows folks to leverage and compose these existing MCP servers to build environments integrated with real world systems e.g. excel
    

## 🤝 Contributing

We welcome new environment recipes, bug reports, and trainer integrations!

⭐ Star our repository to show your support!

## 📜 License

Apache 2.0 © 2025 CGFT Inc.
