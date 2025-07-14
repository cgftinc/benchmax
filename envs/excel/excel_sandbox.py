from pathlib import Path
from typing import Any, Dict, List, Optional

from envs.local_mcp_sandbox import LocalMCPSandbox
from envs.types import RewardFunction, StandardizedExample

SYSTEM_PROMPT = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains six types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- spreadsheet_content: The first few rows of the content of speadsheet file.
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
    TODO: Copy the logic from spreadsheet bench
    """
    return 1.0

class ExcelSandbox(LocalMCPSandbox):
    """Sandbox for spreadsheet manipulation tasks using MCP with Excel support"""

    system_prompt: str = SYSTEM_PROMPT
    reward_funcs: List[RewardFunction] = [reward_func]

    def __init__(self):
        super().__init__(MCP_CONFIG)

    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        # convert dataset json into standardized example
        # Here is the example structure of a single data point:
        # {
        #     "instruction": "What is the sum of all values in column B?",
        #     - instruction holds the question, local path, content, type, position, and output path
        #     "init_rollout_args": {
        #         "spreadsheet_path": "path/to/spreadsheet.xlsx",
        #         -  used to copy spreadsheet to the workspace which is the cwd of the MCP server
        #     }
        # }


    def init_rollout(self, rollout_id: str, **rollout_args):
        if "spreadsheet_path" not in rollout_args:
            raise ValueError("spreadsheet_path must be provided in rollout_args")
        spreadsheet_path = rollout_args["spreadsheet_path"]
        
        super().init_rollout(rollout_id, **rollout_args)
        workspace = self.get_rollout_workspace(rollout_id)

        # Copy the spreadsheet to the workspace
        src_path = Path(spreadsheet_path)
        dest_path = workspace / src_path.name
        if not dest_path.exists():
            dest_path.write_bytes(src_path.read_bytes())
