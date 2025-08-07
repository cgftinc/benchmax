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

Import the BraintrustEnv class and create an instance with your Braintrust API_KEY and Project ID.
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

To view your dataset on Braintrust, you will need to some utils. Supply your APY_KEY and dataset_id. The dataset_id can be found in dataset_ids or copied directly from the Braintrust site.
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

Braintrust has things called functions, which include tools, scorers, and prompts. Try printing out the functions in your Braintrust project to view what's available.
```python
import json

print("Tools List:", braintrust_sandbox2.list_tools())
print("Tools:", braintrust_sandbox2.tools)
print("Prompts:", json.dumps(braintrust_sandbox2.prompts, indent=2))
print("Scorers:", json.dumps(braintrust_sandbox2.scorers, indent=2))
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

## Example of Verifiers Integration

Check the example at examples/verifiers/verifiers_braintrust_example.py to see how a Braintrust environment and dataset would be instantiated to use with Verifiers specifically.

https://github.com/cgftinc/benchmax/blob/braintrust_WIP/examples/verifiers/verifiers_braintrust_example.py