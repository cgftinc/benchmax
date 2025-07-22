from pathlib import Path
import re
from typing import Any, List, Optional
from benchmax.envs.local_mcp_env import LocalMCPEnv
from benchmax.envs.types import RewardFunction, StandardizedExample


SYSTEM_PROMPT = """\
You are an expert in Salesforce and you have access to a Salesforce instance.

# Instructions
- You will be provided a question, the system description, and relevant task context.
- Interact with the Salesforce instance using the tools provided to help you answer the question.
- You should ALWAYS make ONLY ONE tool call at a time. If you want to submit your final answer, just respond with the answer without tool call. If not, you should call some other tool.
- Always end by respond with ONLY the answer, NO full sentence or any explanation.
- If your answer is empty that is there are no records found matching the requirements mentioned, just return 'None' as the response.
"""


def parse_answers(proposed_answer: str, task_name: str) -> List[str]:
    """
    Parse the proposed answer based on the task type.
    This is a placeholder - implement according to your specific parsing logic.
    """
    # Basic implementation - you may need to customize this based on your task requirements
    if not proposed_answer or proposed_answer.strip().lower() == "none":
        return ["None"]
    
    # Split by common delimiters and clean up
    answers = []
    for delimiter in [',', ';', '\n']:
        if delimiter in proposed_answer:
            answers = [ans.strip().strip('"').strip("'") for ans in proposed_answer.split(delimiter)]
            break
    
    if not answers:
        answers = [proposed_answer.strip().strip('"').strip("'")]
    
    return [ans for ans in answers if ans]  # Remove empty strings


def get_all_metrics(proposed_answer: str, ground_truth: str) -> float:
    """
    Compute fuzzy matching score between proposed answer and ground truth.
    Uses F1 score as the primary metric.
    """
    import re
    import string
    from collections import Counter
    
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace('_', ' ')

        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    return f1_score(proposed_answer, ground_truth)


def reward_func(
    prompt: str,
    completion: str,
    ground_truth: List[str],
    workspace: Path,
    reward_metric: str = "exact_match",
    **kwargs: Any
) -> float:
    """
    Reward function for CRM environment that evaluates model completions.
    
    Args:
        prompt: Input prompt given to the model
        completion: Model's generated completion/response
        ground_truth: Expected/correct output (should be a list)
        workspace: Path to rollout's workspace
        reward_metric: Type of matching ("exact_match" or "fuzzy_match")
        **kwargs: Additional context
        
    Returns:
        float: Reward score between 0 and 1
    """
    # Handle None ground truth
    if not ground_truth or ground_truth[0] is None:
        ground_truth = ["None"]

    proposed_answer = completion.strip() if completion else ""
    
    if reward_metric == "exact_match":
        # Parse and normalize the completion text
        completion_tokens = parse_text_to_tokens(proposed_answer)
        
        # Parse and normalize all ground truth items
        all_ground_truth_tokens = set()
        for gt_item in ground_truth:
            gt_tokens = parse_text_to_tokens(str(gt_item))
            all_ground_truth_tokens.update(gt_tokens)
        
        # Calculate IoU (Intersection over Union)
        if not all_ground_truth_tokens and not completion_tokens:
            return 1.0  # Both empty sets match perfectly
        elif not all_ground_truth_tokens or not completion_tokens:
            return 0.0  # One empty, one non-empty
        
        intersection = completion_tokens.intersection(all_ground_truth_tokens)
        union = completion_tokens.union(all_ground_truth_tokens)
        
        iou_score = len(intersection) / len(union) if union else 0.0
        
        # Return 1.0 if perfect match (IoU = 1.0), otherwise return IoU score
        return iou_score
            
    elif reward_metric == "fuzzy_match":
        # For fuzzy match, we only have 1 ground truth item
        if ground_truth[0] is not None:
            return get_all_metrics(proposed_answer, str(ground_truth[0]))
        else:
            return 0.0
    
    else:
        print(f"Unknown reward metric: {reward_metric}")
        return 0.0


def parse_text_to_tokens(text: str) -> set:
    """
    Parse text into normalized tokens using common separators.
    
    Args:
        text: Input text to parse
        
    Returns:
        set: Set of normalized tokens
    """
    if not text:
        return set()
    
    # Clean up the text by removing quotes and extra whitespace
    cleaned_text = text.strip().strip('"').strip("'").lower()
    
    # Split by common separators: spaces, commas, semicolons, pipes, tabs, newlines
    # Using regex to split on multiple separators
    tokens = re.split(r'[,\s|]+', cleaned_text)
    
    # Filter out empty tokens and normalize
    normalized_tokens = {token.strip() for token in tokens if token.strip()}
    
    return normalized_tokens


# Salesforce configuration dictionaries
SALESFORCE_CONFIGS = {
    "b2b": {
        "username": "crmarena_b2b@gmaill.com",
        "password": "crmarenatest", 
        "security_token": "zdaqqSYBEQTjjLuq0zLUHkC3"
    },
    "b2c": {
        "username": "crmarena_b2c@gmaill.com",
        "password": "crmarenatest",
        "security_token": "2AQCtK8MnnV4lJdRNF0DGCs1"
    }
}


def get_mcp_config(config_type: str = "b2b") -> str:
    """
    Generate MCP configuration based on the specified config type.
    
    Args:
        config_type (str): Configuration type - "b2b" or "b2c"
        
    Returns:
        str: JSON configuration string for MCP
        
    Raises:
        ValueError: If config_type is not supported
    """
    if config_type not in SALESFORCE_CONFIGS:
        raise ValueError(f"Unsupported config type: {config_type}. Must be one of: {list(SALESFORCE_CONFIGS.keys())}")
    
    config = SALESFORCE_CONFIGS[config_type]
    
    return f"""
{{
    "mcpServers": {{
      "server-name": {{
        "command": "python",
        "args": ["{Path(__file__).resolve().parents[0]}/salesforce_mcp.py"],
        "env": {{
            "SALESFORCE_USERNAME": "{config['username']}",
            "SALESFORCE_PASSWORD": "{config['password']}",
            "SALESFORCE_SECURITY_TOKEN": "{config['security_token']}"
        }}
      }}
    }}
}}
"""


class CRMEnv(LocalMCPEnv):
    """Environment for CRM tasks using MCP with Salesforce"""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self, dataset_path: Optional[str] = None, **kwargs):
        """ Initialize the CRMEnv with an optional dataset path.
        Args:
            dataset_path (Optional[str]): Path to the dataset directory containing spreadsheets.
        """
        super().__init__(get_mcp_config("b2b"), **kwargs)
        self.dataset_path = dataset_path

    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        # convert dataset example into standardized example
        task = example.get("task", "")
        persona = example.get("persona", "")
        metadata = example.get("metadata", {})
        answer = example.get("answer", [])
        query = example.get("query", "")

        prompt = f"{persona}\n{task}\n{query}"
        if metadata and "required" in metadata:
            required_metadata = metadata["required"]
            prompt = f"{persona}\n{task}\n{required_metadata}\n{query}"

        return {
            "prompt": prompt,
            "ground_truth": answer,
            "init_rollout_args": None
        }
