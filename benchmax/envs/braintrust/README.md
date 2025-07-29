# Braintrust Environment

This environment provides capabilities for interacting with Braintrust through its API, enabling you to utilize tools, scorers, prompts, and datasets from your Braintrust projects.

## Prerequisites

No additional software installation is required. You will just need your Braintrust project ID and API key to get started.

## Installation

```bash
pip install "benchmax[braintrust]"
```
Includes:
- fastmcp: For MCP server functionality

## Available Tools

When you create an instance of the Braintrust environment for a project, all of the tools associated with that project will be downloaded and available to use. Use the `run_tool` function and provide a `tool_id` along with relevant `**tool_args` to execute the tool.

## Reward Function

By default, the reward function will score output against the ground truth using an exact match scoring system. 1.0 for exact matches, and 0.0 otherwise. To use scorers from your project, provide a list of `scorer_ids`. The Braintrust environment will instead use the listed scorers and return the total score.

## Dataset Preprocessing

Because Braintrust allows you to upload a variety of datasets and use variables in your prompts to call on your datasets, you will want to modify the `dataset_preprocess` function to align each template variable in your prompt to its corresponding label in the dataset. 

```
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