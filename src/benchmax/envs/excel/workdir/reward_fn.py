from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastmcp import Client

try:
    from excel_utils import compare_excel_cells
except:
    # Added except local import for unit testing purposes
    from .excel_utils import compare_excel_cells

RewardFunction = Callable[..., Union[float, Awaitable[float]]]


def spreadsheet_comparison_reward(
    completion: str,
    ground_truth: dict,
    mcp_client: Client,
    workspace: Path,
    **kwargs: Any,
) -> float:
    """
    Compares the output spreadsheet to the ground truth using cell values in the specified range.
    Returns 1.0 if all values match, else 0.0.
    """
    answer_position: Optional[str] = kwargs.get("answer_position")
    output_filename: Optional[str] = kwargs.get("output_filename")
    ground_truth_filename: Optional[str] = kwargs.get("ground_truth_filename")

    if not answer_position or not output_filename or not ground_truth_filename:
        raise ValueError(
            "kwargs must contain 'answer_position', 'output_filename', and 'ground_truth_filename' fields"
        )

    output_path = workspace / output_filename
    ground_truth_path = workspace / ground_truth_filename

    # Return 1.0 score if the output completely matches the ground truth
    try:
        match, _ = compare_excel_cells(
            str(ground_truth_path), str(output_path), answer_position
        )
        return 1.0 if match else 0.0
    except Exception as e:
        print(
            f"Error comparing spreadsheets {ground_truth_path} and {output_path}: {e}"
        )
        return 0.0


# -------------------------------
# Export reward functions
# -------------------------------
reward_functions: Dict[str, RewardFunction] = {
    "spreadsheet": spreadsheet_comparison_reward,
}
