import json
from dotenv import load_dotenv
import os
from typing import Any, List, Optional, Tuple
from pathlib import Path

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import ToolDefinition, RewardFunction, StandardizedExample
import benchmax.envs.braintrust.braintrust_utils as braintrust_utils

from datasets import (
    DatasetDict, Dataset, IterableDatasetDict,
    IterableDataset, load_dataset
)

load_dotenv()  # load variables from .env

API_KEY = os.environ.get("API_KEY")

def replace_template_variables(prompt_template: str, **variables) -> str:
    """
    Replace template variables in a prompt string with provided values.
    
    Args:
        prompt_template: String containing template variables like {{input}}, {{metadata}}, etc.
        **variables: Keyword arguments where keys match template variable names
        
    Returns:
        String with template variables replaced with actual values
        
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
    """
    result = prompt_template
    
    for var_name, var_value in variables.items():
        placeholder = f"{{{{{var_name}}}}}"
        if placeholder in result:
            result = result.replace(placeholder, str(var_value))
    
    return result

# Given an API key and project id, pull the project data(tools, scorers, datasets, etc).
class BraintrustSandbox(BaseEnv):
    def __init__(self, braintrust_api_key: str, braintrust_project_id: str, scorer_ids=[]):
        # Set API Key and Project ID
        self.braintrust_api_key = braintrust_api_key
        self.braintrust_project_id = braintrust_project_id

        # Collect Project Name and Data
        self.project_data = braintrust_utils.get_project_data(self.braintrust_api_key, self.braintrust_project_id)
        self.project_name = self.project_data["objects"][0]["name"]

        # Collect Dataset IDs 
        self.dataset_ids = braintrust_utils.get_dataset_ids(self.braintrust_api_key, self.braintrust_project_id)

        # Collect Functions(tools, prompts, and scorers)
        self.functions = braintrust_utils.get_functions(self.braintrust_api_key, self.braintrust_project_id)
        self.tools = self.functions["tools"]
        self.prompts = self.functions["prompts"]
        self.scorers = self.functions["scorers"]
        self.activated_scorers = {}
        for id in scorer_ids:
            self.activated_scorers[id] = self.scorers[id]

        self.system_prompt: str = ""
        self.reward_funcs: List[RewardFunction] = [self.reward_func]

    # Override this method if your example does not match the default structure
    def dataset_preprocess(self, example: Any, prompt: str) -> StandardizedExample:
        """
        Preprocess a single dataset example into a dict with keys:
        - "prompt": str
        - "ground_truth": Any
        - "init_rollout_args": Optional[Dict[str, Any]]
        """
        if "prompt" in example and "input" in example:
            input = example["prompt"]["input"]
        else:
            raise Exception("Error with data preprocessing.")
        
        # Replace template variables in the prompt -- can support multiple template variables to fit dataset
        processed_prompt = replace_template_variables(prompt, input=input)
        
        ground_truth = example.pop("ground_truth", "")
        init_rollout_args = example.pop("init_rollout_args", "")
        return StandardizedExample(
            prompt=processed_prompt,
            ground_truth=ground_truth,
            init_rollout_args=init_rollout_args,
            **example,
        )

    @classmethod
    def load_dataset(
        cls, dataset_name: str, braintrust_api_key: str, braintrust_dataset_id: str, **kwargs
    ) -> (
        Tuple[DatasetDict | Dataset | IterableDatasetDict | IterableDataset, str | None]
    ):
        """
        Download and prepare a dataset for use with this environment.

        This method should handle retrieving the specified dataset (e.g., from HuggingFace, local files,
        or a custom source), preprocessing or converting it into a compatible structure, and storing it
        locally in a reusable format. The processed dataset should be suitable for downstream use with
        `dataset_preprocess`, which standardizes individual examples into the expected format.

        Args:
            dataset_name (str): Identifier of the dataset to be loaded.
            **kwargs: Additional dataset-specific arguments (e.g., split, filtering options, cache directory).

        Returns:
            Dataset: A dataset object (e.g., HuggingFace Dataset or similar) ready for processing.
            str: Optional string pointing to where the dataset is stored locally
        """
        dataset = braintrust_utils.get_dataset_with_id(braintrust_api_key, braintrust_dataset_id)
        dataset = Dataset.from_list(dataset)
        return dataset, None

    def reward_func(self, prompt: str, completion: str, ground_truth: str, **kwargs) -> float:
        """ If given scorer id, run scorer. Otherwise, use ExactMatch to score.
        Score 1.0 if the ground-truth answer exactly matches completion; otherwise 0.0.

        Args:
            prompt:   The user’s original query.
            completion:  The model’s generated text.
            ground_truth:  Expected answer to match (case-insensitive).
            workspace:  Path to the current workspace (unused).
            **kwargs:  Catch-all for BaseEnv signature compatibility.
        Returns:
            1.0 if `ground_truth` (lowercased) exactly matches else 0.0.
        """
        
        if len(self.activated_scorers) > 0:
            score = 0.0
            for scorer_id in self.activated_scorers.keys():
                score += self.run_scorer(scorer_id=scorer_id, input=prompt, output=completion, expected=ground_truth, metadata={})
            return score
        else:
            return float(ground_truth.lower() == completion.lower())

    def run_scorer(self, scorer_id, **scorer_args) -> Any:
        """Execute scorer with id given arguments. Expecting scorer to function defined as follows:
        def handler(
            input: Any,
            output: Any,
            expected: Any,
            metadata: dict[str, Any]
            ) -> float | None:
        """
        namespace = {}
        code_str = self.scorers[scorer_id]["data"]["code"]
        exec(code_str, namespace)

        func = namespace["handler"]
        result = func(**scorer_args)
        return result

    def list_dataset_ids(self) -> List[str]:
        """List of dataset names and corresponding id"""
        return [
            (self.dataset_ids[id], id) for id in self.dataset_ids 
        ]

    def list_tools(self) -> List[ToolDefinition]:
        """List of available tools with id."""
        return [
            (self.tools[k][0]["name"], k) for k in sorted(self.tools)
        ]
        
    def run_tool(self, tool_id: str, **tool_args) -> Any:
        """Execute tool with id given arguments"""
        namespace = {}
        code_str = self.tools[tool_id][1]
        exec(code_str, namespace)

        user_funcs = [name for name, obj in namespace.items() if callable(obj) and not name.startswith("__")]

        if len(user_funcs) == 1:
            func = namespace[user_funcs[0]]
            result = func(**tool_args)
            return result
        else:
            print("Tool unable to run. Multiple functions found:", user_funcs)

    def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        return super().init_rollout(rollout_id, **rollout_args)
    
    def cleanup_rollout(self, rollout_id: str) -> None:
        return super().cleanup_rollout(rollout_id)
    
    def get_rollout_workspace(self, rollout_id: str) -> Path:
        return super().get_rollout_workspace(rollout_id)     

# calls braintrust api to pull tools and scorers and then populates the sandbox with them
if __name__ == "__main__":
    #braintrust_sandbox1 = BraintrustSandbox(API_KEY, "afc6fc6e-ef12-41f1-ad2b-b3ab3e760db5")
    braintrust_sandbox2 = BraintrustSandbox(API_KEY, "89d983a3-7012-4932-a816-67f025318096")
    print("Project Name:", braintrust_sandbox2.project_name)
    print("Project Data", braintrust_sandbox2.project_data)
    print("Datasets ids:", braintrust_sandbox2.dataset_ids)
    print("Datasets:", json.dumps(braintrust_sandbox2.datasets, indent=2))
    print("Tools List:", braintrust_sandbox2.list_tools())
    print("Tools:", json.dumps(braintrust_sandbox2.tools, indent=2))
    print("Prompts:", json.dumps(braintrust_sandbox2.prompts, indent=2))
    print("Scorers:", json.dumps(braintrust_sandbox2.scorers, indent=2))

    tool_output = braintrust_sandbox2.run_tool(tool_id='229ad51a-88b4-48f8-95ed-aa4d67c68c80', op="add", a=5, b=6)
    score = braintrust_sandbox2.run_scorer(scorer_id='138d1ecb-5eb0-424c-a568-8fb5ba1e4e9b', input="bruh", output="bruh", expected="bro", metadata={})
    print("Tool Output:", tool_output)
    print("Score:", score)

    result1 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hell0o", scorer_id=None)
    result2 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hel0lo", scorer_id='138d1ecb-5eb0-424c-a568-8fb5ba1e4e9b')
    print("Result 1 - Default Scorer, Expected Fail:", result1)
    print("Result 2 - Custom Scorer, Expected Fail:", result2)

    result3 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hello", scorer_id=None)
    result4 = braintrust_sandbox2.reward_func(prompt="", completion="hello", ground_truth="hello", scorer_id='138d1ecb-5eb0-424c-a568-8fb5ba1e4e9b')
    print("Result 3 - Default Scorer, Expected Pass:", result3)
    print("Result 4 - Custom Scorer, Expected Pass:", result4)
