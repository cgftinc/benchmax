# Parallel MCP Env

This directory contains:
```bash
├── example_workdir/        # An example of a workdir
├── parallel_mcp_env.py     # Parallel MCP env
├── provisioners/           # Various methods of provisioning server (e.g. local, skypilot)
├── proxy_server.py         # Proxy server that redirects MCP connection
├── server_pool.py          # Helper to manage servers
└── utils.py                # Shared utils
```

With parallel MCP env, you can easily run multiple MCP servers in parallel either locally or in the cloud. The environment supports both local development and cloud deployment through various provisioners, making it flexible for different use cases.

## Workdir Structure

The `workdir` directory contains all the files necessary to run your MCP environment. It should include:

1. `mcp_config.yaml`: Configuration for spinning up MCP servers
2. `reward_fn.py`: Implementation of the reward function for evaluating model outputs
3. `setup.sh`: Script that runs at startup to install dependencies
4. Additional files: Any other files required by your MCP server (e.g., mcp server code, assets, helper scripts)

When the environment is initialized, the entire workdir is copied to the execution machine, ensuring all necessary components are available whether running locally or in the cloud.


## Creating a custom environment using MCP

To create a custom environment using an MCP server (like a calculator, browser, or spreadsheet), you can extend `ParallelMcpEnv`. Here's a quick step-by-step guide using `benchmax.envs.math.math_env.MathEnv` as an example.

### 1. **Define a System Prompt**

This prompt guides the LLM’s behavior. It can include any instruction, such as how to format the answer or when to use tools.

```python
SYSTEM_PROMPT = """Please use the tools provided to do any computation.
Write your complete answer on the final line only, within the xml tags <answer></answer>.
"""
```

### 2. **Configure MCP Server(s)**

Define the MCP servers to be launched. You can configure one or more in `src/benchmax/envs/math/workdir/mcp_config.yaml`:

```yaml
mcpServers:
  calculator:
    command: uvx
    args:
      - mcp-server-calculator
```

This MCP config will be used to spin up multiple MCP servers, which can run on a single machine (if configured locally) or distributed across multiple nodes in the cloud (if using SkyPilot multi-node).

### 3. **Write a Reward Function**

The reward function evaluates how "correct" the model's output is, based on structured output. Here’s a simple XML-based example:

Note that `**kwargs` contains all the other fields in your dataset example, so feel free to use them in `reward_fn` calculations.

```python
async def text_match_reward(
    completion: str,
    ground_truth: str,
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any
) -> float:
    """
    Reward = 1 if `ground_truth` (case-insensitive) appears anywhere *inside*
    the first <answer> … </answer> block of `completion`; otherwise 0.

    Falls back to 0 if the tag is missing or empty.
    """
    
    # Grab only the text inside the first <answer> … </answer> pair (case-insensitive).
    m = re.search(r'<answer>(.*?)</answer>', completion, flags=re.IGNORECASE | re.DOTALL)
    if m is None:
        return 0.0

    # Unescape any XML entities (&amp; → &, etc.) and normalise whitespace.
    answer_text = unescape(m.group(1)).strip().lower()

    try:
        # Try to interpret both as floats for numerical comparison.
        return float(float(ground_truth.lower()) == float(answer_text))
    except ValueError:
        return 0.0
```

Check out [math env's reward_fn.py](/src/benchmax/envs/math/workdir/reward_fn.py) for the full implementation. You can reference [this example reward_fn.py](/src/benchmax/envs/mcp/example_workdir/reward_fn.py) for a comprehensive usage of various ways to compute reward.

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

### 5. **Extend `ParallelMcpEnv`**

Now bring everything together into a custom environment class:

```python
SYSTEM_PROMPT = """Please use the tools provided to do any computation.
Write your complete answer on the final line only, within the xml tags <answer></answer>.\n
"""

class MathEnv(ParallelMcpEnv):
    """Environment for math problems, using local MCP tools."""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(self, workdir_path: Path, provisioner: BaseProvisioner, **kwargs):
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> StandardizedExample:
        return StandardizedExample(
            prompt=example.get("task", ""),
            ground_truth=example.get("answer", ""),
            init_rollout_args=None,
        )
```

You're done! This environment is now compatible with `benchmax` and can be plugged into any compatible RL trainer.

In [math_env.py](/src/benchmax/envs/math/math_env.py), we have also provided `MathEnvLocal` class to run the env locally and `MathEnvSkypilot` to run it remotely across multiple machines. 
