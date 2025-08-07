# Braintrust Environment

This environment provides capabilities for interacting with Braintrust through its API, enabling you to utilize tools, scorers, prompts, and datasets from your Braintrust projects to set up a RL training environment in Verifiers/Verl. 

## Prerequisites

No additional software installation is required. You will just need your Braintrust project ID and API key to get started.

## Available Tools

When you create an instance of the Braintrust environment for a project, all of the tools associated with that project will be downloaded and available to use. Use the `run_tool` function and provide a `tool_id` along with relevant `**tool_args` to execute the tool.

## Reward Function

By default, the reward function will score output against the ground truth using an exact match scoring system. 1.0 for exact matches, and 0.0 otherwise. To use scorers from your project, provide a list of `scorer_ids`. The Braintrust environment will instead use the listed scorers and return the total score.

## Dataset Preprocessing

Because Braintrust allows you to upload a variety of datasets and use variables in your prompts to call on your datasets, you will want to modify the `dataset_preprocess` function to align each template variable in your prompt to its corresponding label in the dataset. 

```python
Examples:
        # Single variable
        replace_template_variables("Hello {{input}}!", input="world") 
        # Returns: "Hello world!"
        
        # Multiple variables
        replace_template_variables(
            "What is the recommend action in this scenario {{input}} on this street {{metadata}}?", 
            input="Hand: AhQs, Position: CO", 
            metadata="flop"
        )
        # Returns: "What is the recommend action in this scenario Hand: AhQs, Position: CO on this street flop?"
```

## Starting up a Braintrust Environment Instance

Before integrating the Braintrust environment with Verifiers, you may want to create an instance of the Braintrust environmen to test that your project and its respective tools are as expected. This short tutorial will demonstrate how to create an instance and print out all of the data from your Braintrust project. Later in the README.md, we will go over how to integrate this Braintrust environment into an RL trainer like Verifiers to fine tune our models.

As an example, I have created a my own project in Braintrust. My Braintrust project uses a poker agent as an example, where the goal is for the LLM to give a player a recommendation on what their next move should be. Your specific prompts and tools will depend on what you have set up your Braintrust project with!

To start, import the BraintrustEnv class and create an instance with your Braintrust API_KEY and Project ID. The API_KEY can be found in Settings under API keys. Find your project and click "Copy project ID" to copy your project ID to your clipboard. Fill in "YOUR_API_KEY" and "YOUR_PROJECT_ID" when creating the Braintrust instance, like shown below.
```python
from benchmax.envs.braintrust.braintrust_env import BraintrustEnv

braintrust_sandbox2 = BraintrustEnv(braintrust_api_key="YOUR_API_KEY", braintrust_project_id="YOUR_PROJECT_ID")
```

Try viewing some of the project details!
```python
print("Project Name:", braintrust_sandbox2.project_name)
print("Project Data", braintrust_sandbox2.project_data)
print("Datasets ids:", braintrust_sandbox2.dataset_ids)
```

It should look something like this, but with the details from your specific project.
```bash
Project Name: benchmax-first-project
Project Data {'objects': [{'id': '89e984a3-7012-4932-a816-67f025323096', 'org_id': 'e50ec4e9-916d-1bc9-2356-a4023259ac85', 'name': 'benchmax-first-project', 'created': '2025-07-14T18:20:46.065Z', 'deleted_at': None, 'user_id': '11319aea-8a7d-4c1d-91e5-edbdb095cde9', 'settings': None}]}
Datasets ids: {'0e07f061-1b7c-4a8f-aedb-6f2ca785ccc4': 'Dataset 2', '27280455-f531-4883-a5ec-b81a13e22eb6': 'Dataset 1'}
```

To view your dataset on Braintrust, you will need to some utils. Supply your APY_KEY and dataset_id. The dataset_id can be found in dataset_ids, copied directly from the Braintrust site, or pulled using the Braintrust API. I recommend using the id listed in dataset_ids or copying from the Braintrust page.
```python
import benchmax.envs.braintrust.braintrust_utils as braintrust_utils

dataset_util = braintrust_utils.get_dataset_with_id(braintrust_api_key=API_KEY, dataset_id='YOUR_DATASET_ID')
    dataset, _ = BraintrustEnv.load_dataset(braintrust_api_key=API_KEY, braintrust_dataset_id='YOUR_DATASET_ID')
    
    print("Dataset from util call:", json.dumps(dataset_util, indent=2))
    dataset = dataset.map(
        lambda example: braintrust_sandbox2.dataset_preprocess(example=example, prompt="[prompt here]")
    )   
    print("Dataset from load_dataset call:", dataset)
```

The output should look something like this.
```bash
Dataset from util call: [
  {
    "prompt": {
      "input": "Hand: AhQs, Position: CO, Stack: 90bb, Pot: 9bb, Board: Td9s2c, Action before: checked to hero",
      "street": "flop"
    },
    "ground_truth": "Bet 6bb"
  },
  {
    "prompt": {
      "input": "Hand: JcJh, Position: UTG, Stack: 120bb, Pot: 0bb, Action before: new hand",
      "street": "preflop"
    },
    "ground_truth": "Raise to 2.3bb"
  },
  {
    "prompt": {
      "input": "Hand: 8d9d, Position: SB, Stack: 80bb, Pot: 10bb, Action before: hero faces open from CO 2.2bb",
      "street": "preflop"
    },
    "ground_truth": "Call"
  },
  {
    "prompt": {
      "input": "Hand: AsKs, Position: BTN, Stack: 100bb, Pot: 6bb, Action before: folded to hero",
      "street": "preflop"
    },
    "ground_truth": "Raise to 2.5bb"
  },
  {
    "prompt": {
      "input": "Hand: AhQs, Position: CO, Stack: 90bb, Pot: 9bb, Board: Td9s2c, Action before: checked to hero",
      "street": "flop"
    },
    "ground_truth": "Bet 6bb"
  },
  {
    "prompt": {
      "input": "Hand: JcJh, Position: UTG, Stack: 120bb, Pot: 0bb, Action before: new hand",
      "street": "preflop"
    },
    "ground_truth": "Raise to 2.3bb"
  },
  {
    "prompt": {
      "input": "Hand: 8d9d, Position: SB, Stack: 80bb, Pot: 10bb, Action before: hero faces open from CO 2.2bb",
      "street": "preflop"
    },
    "ground_truth": "Call"
  },
  {
    "prompt": {
      "input": "Hand: AsKs, Position: BTN, Stack: 100bb, Pot: 6bb, Action before: folded to hero",
      "street": "preflop"
    },
    "ground_truth": "Raise to 2.5bb"
  }
]
Map: 100%|███████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 1188.90 examples/s]
Dataset from load_dataset call: Dataset({
    features: ['prompt', 'ground_truth', 'init_rollout_args'],
    num_rows: 8
})
```

Braintrust has things called functions, which include tools, scorers, and prompts. Try printing out the functions in your Braintrust project to view what's available. If you do not have a tool yet in your Braintrust project, you will need to upload one to your project using Braintrust's API. Here is their documentation: https://www.braintrust.dev/docs/guides/functions/tools
```python
import json

print("Tools List:", braintrust_sandbox2.list_tools())
print("Tools:", braintrust_sandbox2.tools)
print("Prompts:", json.dumps(braintrust_sandbox2.prompts, indent=2))
print("Scorers:", json.dumps(braintrust_sandbox2.scorers, indent=2))
```

The output should look something like below. 

Notice that `list_tools()` returns a list of `ToolDefinitions`. In my case, I have a Double Glitched Calculator that doubles the expected result and a normal Calculator tool. Neither of these tools are relevant to poker, but they were added to show an example. Currently, only python tools are supported. Printing out tools using `braintrust_sandbox2.tools` shows a dictionary from our tool ids to the Tool Definition.

Prompt and Scorers are pretty straight forward. You can now see the dictionary of prompt ids to prompts and scorer ids to scorers.

```bash
Tools List: [ToolDefinition(name='Doubled Glitch Calculator method', description='A glitched calculator that can add, subtract, multiply, and divide. It doubles the result.', input_schema={'type': 'object', 'properties': {'a': {'type': 'number', 'title': 'A'}, 'b': {'type': 'number', 'title': 'B'}, 'op': {'enum': ['add', 'subtract', 'multiply', 'divide'], 'type': 'string', 'title': 'Op'}}, 'required': ['op', 'a', 'b']}), ToolDefinition(name='Calculator method', description='A simple calculator that can add, subtract, multiply, and divide.', input_schema={'type': 'object', 'properties': {'a': {'type': 'number', 'title': 'A'}, 'b': {'type': 'number', 'title': 'B'}, 'op': {'enum': ['add', 'subtract', 'multiply', 'divide'], 'type': 'string', 'title': 'Op'}}, 'required': ['op', 'a', 'b']})]
Tools: {'123ad51a-77b4-48f8-95ed-aa4d67c68c80': (ToolDefinition(name='Doubled Glitch Calculator method', description='A glitched calculator that can add, subtract, multiply, and divide. It doubles the result.', input_schema={'type': 'object', 'properties': {'a': {'type': 'number', 'title': 'A'}, 'b': {'type': 'number', 'title': 'B'}, 'op': {'enum': ['add', 'subtract', 'multiply', 'divide'], 'type': 'string', 'title': 'Op'}}, 'required': ['op', 'a', 'b']}), 'def calculator(op, a, b):\n    match op:\n        case "add":\n            return (a + b) * 2\n        case "subtract":\n            return (a - b) * 2\n        case "multiply":\n            return (a * b) * 2\n        case "divide":\n            return (a / b) * 2'), '1232616b-2e57-4231-7faf-ac3120a394cf': (ToolDefinition(name='Calculator method', description='A simple calculator that can add, subtract, multiply, and divide.', input_schema={'type': 'object', 'properties': {'a': {'type': 'number', 'title': 'A'}, 'b': {'type': 'number', 'title': 'B'}, 'op': {'enum': ['add', 'subtract', 'multiply', 'divide'], 'type': 'string', 'title': 'Op'}}, 'required': ['op', 'a', 'b']}), 'def calculator(op, a, b):\n    match op:\n        case "add":\n            return a + b\n        case "subtract":\n            return a - b\n        case "multiply":\n            return a * b\n        case "divide":\n            return a / b')}
Prompts: {
  "123e1e74-8c32-41f6-97bc-4b123be34e3b": [
    {
      "role": "system",
      "content": "You are a poker coach who gives instruction on the GTO poker play."
    },
    {
      "role": "user",
      "content": "What is the recommend action in this scenario {{input}} on this street {{metadata}}? The response should include only the recommended action:\nFold, Call, Bet [amount]bb, or Raise to [amount]bb"
    }
  ]
}
Scorers: {
  "123d1ecb-5eb0-124c-a568-8fb5ba1e4e9b": {
    "data": {
      "code": "from typing import Any\n# Enter handler function that returns a numeric score between 0 and 1,\n# or None to skip scoring\ndef handler(\n  input: Any,\n  output: Any,\n  expected: Any,\n  metadata: dict[str, Any]\n) -> float | None:\n  if expected is None:\n    return None\n  return 1.0 if output == expected else 0.0",
      "type": "inline",
      "runtime_context": {
        "runtime": "python",
        "version": "3.12"
      }
    },
    "type": "code"
  }
}
```

You can run custom tools and scorers from Braintrust. Tools can be used by LLMs and Scorers are used to score the completions by the LLM against a defined `ground_truth`.
```python
tool_output = braintrust_sandbox2.run_tool(tool_id='YOUR_TOOL_ID', op="add", a=5, b=6)
score = braintrust_sandbox2.run_scorer(scorer_id='YOUR_SCORER_ID', input="bruh", output="bruh", expected="bro", metadata={})
print("Tool Output:", tool_output)
print("Score:", score)
```

The reward function can be used by an RL trainer to fine tune your model. But you can still test it out your scorers with some self made completions and ground truths
```python
result1 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hell0o", scorer_id=None, workspace=None)
result2 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hel0lo", scorer_id='YOUR_SCORER_ID', workspace=None)
print("Result 1 - Default Scorer, Expected Fail:", result1)
print("Result 2 - Custom Scorer, Expected Fail:", result2)

result3 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hello", scorer_id=None, workspace=None)
result4 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hello", scorer_id='YOUR_SCORER_ID', workspace=None)
print("Result 3 - Default Scorer, Expected Pass:", result3)
print("Result 4 - Custom Scorer, Expected Pass:", result4)
```

The expected output should look something like this if you are using the calculator tools to follow the example.
```bash
Tool Output: 22
Score: 0.0
Result 1 - Default Scorer, Expected Fail: 0.0
Result 2 - Custom Scorer, Expected Fail: 0.0
Result 3 - Default Scorer, Expected Pass: 1.0
Result 4 - Custom Scorer, Expected Pass: 1.0
```

## Example of Verifiers Integration

Check the example at examples/verifiers/verifiers_braintrust_example.py to see how a Braintrust environment and dataset would be instantiated to use with Verifiers specifically.

https://github.com/cgftinc/benchmax/blob/braintrust_WIP/examples/verifiers/verifiers_braintrust_example.py