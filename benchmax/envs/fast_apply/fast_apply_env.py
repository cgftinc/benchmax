from pathlib import Path
from typing import List, Any, Tuple

from datasets import (
    DatasetDict, Dataset, IterableDatasetDict,
    IterableDataset
)

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.fast_apply.utils import calc_score
from benchmax.envs.types import RewardFunction, StandardizedExample

SYSTEM_PROMPT = \
'''
You are a helpful assistant for a code editor that applies an edit to code to merge them together. That is, you will be given code wrapper in <code> tags and an edit wrapped in <edit> tags, and you will apply the edit to the code.

For example:

<code>
CODE_SNIPPET
</code>

<edit>
EDIT_SNIPPET
</edit>

The code is any type of code and the edit is in the form of code that you need to update along with comments that indicate where existing code is to be kept.

// ... existing code ...
FIRST_EDIT
// ... existing elements ...
SECOND_EDIT
// ... existing methods ...
THIRD_EDIT
// ... existing methods ...

The merged code must be exact with no room for any errors. Make sure all whitespaces are preserved correctly. A small typo in code will cause it to fail to compile or error out, leading to poor user experience.

Output the code wrapped in <code> tags.
'''

def reward_func(prompt: str, completion: str, ground_truth: str, workspace: Path, **kwargs) -> float:
   return calc_score(completion, ground_truth)

def prompt_builder(original_code: str, code_update_snippet: str):
    return f"<code>\n{original_code}</code>\n\n<edit>{code_update_snippet}</edit>"


class FastApplyEnv(BaseEnv):
    """Single Turn Environment to train fast apply models for code."""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    @classmethod
    def load_dataset(
        cls, dataset_name: str = "Kortix/FastApply-dataset-v1.0"
    ) -> (
        Tuple[DatasetDict | Dataset | IterableDatasetDict | IterableDataset, str | None]
    ):
        return super().load_dataset(dataset_name)
    
    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        return StandardizedExample(
            prompt=prompt_builder(example.get("original_code"), example.get("update_snippet")),
            ground_truth=example.get("final_code"),
            init_rollout_args={}
        )
