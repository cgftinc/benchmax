import re
import string
from html import unescape
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from fastmcp import Client

RewardFunction = Callable[..., Union[float, Awaitable[float]]]


def parse_answers(proposed_answer: str) -> str:
    """
    Parse the proposed answer.
    """
    m = re.search(
        r"<answer>(.*?)</answer>", proposed_answer, flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        proposed_answer = ""
    else:
        # Unescape any XML entities (&amp; → &, etc.) and normalise whitespace.
        proposed_answer = unescape(m.group(1)).strip().lower()
    return proposed_answer


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
    tokens = re.split(r"[,\s|]+", cleaned_text)

    # Filter out empty tokens and normalize
    normalized_tokens = {token.strip() for token in tokens if token.strip()}

    return normalized_tokens


def get_all_metrics(proposed_answer: str, ground_truth: str) -> float:
    """
    Compute fuzzy matching score between proposed answer and ground truth.
    Uses F1 score as the primary metric.
    """

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
            return "".join(ch if ch not in exclude else " " for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace("_", " ")

        return white_space_fix(
            remove_articles(handle_punc(lower(replace_underscore(s))))
        ).strip()

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


def crm_matching_reward_function(
    completion: str,
    ground_truth: List[str],
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any,
) -> float:
    """
    Reward function for CRM environment that evaluates model completions.

    Args:
        prompt: Input prompt given to the model
        completion: Model's generated completion/response
        ground_truth: Expected/correct output (should be a list)
        workspace: Path to rollout's workspace
        **kwargs: Additional context

    Returns:
        float: Reward score between 0 and 1
    """
    reward_metric: Optional[str] = kwargs.get("reward_metric")

    if not reward_metric:
        raise ValueError("kwargs must contain reward metric")

    proposed_answer = completion.strip() if completion else ""
    proposed_answer = parse_answers(proposed_answer)

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


# -------------------------------
# Export reward functions
# -------------------------------
reward_functions: Dict[str, RewardFunction] = {
    "match": crm_matching_reward_function,
}
