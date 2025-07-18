import os
from pathlib import Path
from typing import Any, List, Optional

from envs.local_mcp_env import LocalMCPEnv
from envs.types import RewardFunction, StandardizedExample
from envs.excel.excel_utils import compare_excel_cells, excel_to_str_repr

SYSTEM_PROMPT = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains six types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- spreadsheet_content: The content of speadsheet file.
- instruction_type: There are two values (Cell-Level Manipulation, Sheet-Level Manipulation) used to indicate whether the answer to this question applies only to specific cells or to the entire worksheet.
- answer_position: The position need to be modified or filled. For Cell-Level Manipulation questions, this field is filled with the cell position; for Sheet-Level Manipulation, it is the maximum range of cells you need to modify. You only need to modify or fill in values within the cell range specified by answer_position.
- output_path: You need to generate the modified spreadsheet file in this new path.
"""

MCP_CONFIG = """
{
    "mcpServers": {
      "server-name": {
        "command": "python",
        "args": ["custom_mcp/excel_code_runner_mcp.py"]
      }
    }
}
"""

def reward_func(
    prompt: str, completion: str, ground_truth: str, workspace: Path, **kwargs
) -> float:
    """
    Compares the output spreadsheet to the ground truth using cell values in the specified range.
    Returns 1.0 if all values match, else 0.0.
    """
    example_id = kwargs.get("id")
    answer_position = kwargs.get("answer_position")
    spreadsheet_path = kwargs.get("spreadsheet_path")
    if not example_id or not answer_position or not spreadsheet_path:
        return 0.0

    # Build file paths
    ground_truth_path = workspace / f"1_{example_id}_answer.xlsx"
    output_path = workspace / f"1_{example_id}_output.xlsx"

    # Return 1.0 score if the output completely matches the ground truth
    try:
        match, _ = compare_excel_cells(str(ground_truth_path), str(output_path), answer_position, )
        return 1.0 if match else 0.0
    except Exception as e:
        print(f"Error comparing spreadsheets for example {example_id}: {e}")
        return 0.0

class ExcelEnv(LocalMCPEnv):
    """Environment for spreadsheet manipulation tasks using MCP with Excel support"""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self, dataset_path: Optional[str] = None, **kwargs):
        """ Initialize the ExcelEnv with an optional dataset path.
        Args:
            dataset_path (Optional[str]): Path to the dataset directory containing spreadsheets.
"""
        super().__init__(MCP_CONFIG, **kwargs)
        self.dataset_path = dataset_path

    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        # convert dataset json into standardized example
        example_id = example.get("id")
        if not example_id:
            raise ValueError("Example must contain an 'id' field")
        spreadsheet_path = example.get("spreadsheet_path")
        if not spreadsheet_path:
            raise ValueError("spreadsheet_path must be provided in the example")
        spreadsheet_path = os.path.join(self.dataset_path, spreadsheet_path) if self.dataset_path else spreadsheet_path
        target_input_path = f"1_{example_id}_input.xlsx"
        target_output_path = f"1_{example_id}_output.xlsx"
        target_answer_path = f"1_{example_id}_answer.xlsx"
        source_input_path = Path(spreadsheet_path) / target_input_path

        spreadsheet_content = excel_to_str_repr(str(source_input_path), True)

        prompt = f"""
        Instruction: {example['instruction']}
        Spreadsheet Path: {target_input_path}
        Spreadsheet Content: {spreadsheet_content}
        Instruction Type: {example['instruction_type']} 
        Answer Position: {example['answer_position']}
        Output Path: {target_output_path}
        """

        return {
            "prompt": prompt.strip(),
            "ground_truth": "",
            "init_rollout_args": {
                "spreadsheet_path": str(source_input_path),
                "answer_spreadsheet_path": str(Path(spreadsheet_path) / target_answer_path),
            }
        }

    def init_rollout(self, rollout_id: str, **rollout_args):
        if "spreadsheet_path" not in rollout_args:
            raise ValueError("spreadsheet_path must be provided in rollout_args")
        spreadsheet_path = rollout_args["spreadsheet_path"]
        answer_spreadsheet_path = rollout_args["answer_spreadsheet_path"]
        
        super().init_rollout(rollout_id, **rollout_args)
        workspace = self.get_rollout_workspace(rollout_id)

        def _copy_to_workspace(src_path: Path):
            """
            Copy the spreadsheet file to the workspace if it doesn't already exist.
            """
            dest_path = workspace / src_path.name
            if not dest_path.exists():
                dest_path.write_bytes(src_path.read_bytes())

        # Copy the spreadsheet to the workspace
        _copy_to_workspace(Path(spreadsheet_path))
        _copy_to_workspace(Path(answer_spreadsheet_path))
