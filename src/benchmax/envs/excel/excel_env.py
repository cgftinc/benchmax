import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple, Optional

from benchmax.envs.mcp.parallel_mcp_env import ParallelMcpEnv
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner
from benchmax.envs.mcp.provisioners.local_provisioner import LocalProvisioner
from benchmax.envs.mcp.provisioners.skypilot_provisioner import SkypilotProvisioner
from benchmax.envs.types import StandardizedExample
from .data_utils import download_and_extract

# Using library shared with mcp workdir
from .workdir.excel_utils import excel_to_str_repr

if TYPE_CHECKING:
    import sky
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

SYSTEM_PROMPT = """You are a spreadsheet expert who can manipulate spreadsheets through Python code.

You need to solve the given spreadsheet manipulation question, which contains six types of information:
- instruction: The question about spreadsheet manipulation.
- spreadsheet_path: The path of the spreadsheet file you need to manipulate.
- spreadsheet_content: The content of speadsheet file.
- instruction_type: There are two values (Cell-Level Manipulation, Sheet-Level Manipulation) used to indicate whether the answer to this question applies only to specific cells or to the entire worksheet.
- answer_position: The position need to be modified or filled. For Cell-Level Manipulation questions, this field is filled with the cell position; for Sheet-Level Manipulation, it is the maximum range of cells you need to modify. You only need to modify or fill in values within the cell range specified by answer_position.
- output_path: You need to generate the modified spreadsheet file in this new path.
"""

DEFAULT_DATA_OUTPUT_PATH = os.path.expanduser("~/.cache/excel_data")
SPREADSHEET_FULL = "all_data_912_v0.1"
SPREADSHEET_SAMPLE = "sample_data_200"

# Set train data to full for proper training
SPREADSHEET_BENCH_TRAIN_DATA = SPREADSHEET_SAMPLE


class ExcelExample(StandardizedExample):
    id: str
    answer_position: str
    output_filename: str
    ground_truth_filename: str
    spreadsheet_base_dir: str


class ExcelEnv(ParallelMcpEnv):
    """Environment for spreadsheet manipulation tasks using MCP with Excel support"""

    system_prompt: str = SYSTEM_PROMPT

    def __init__(
        self,
        workdir_path: Path,
        provisioner: BaseProvisioner,
        **kwargs,
    ):
        """Initialize the ExcelEnv with an optional dataset path."""
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)

    @classmethod
    def load_dataset(
        cls,
        dataset_name: str = "spreadsheetbench",
        data_output_path: str = DEFAULT_DATA_OUTPUT_PATH,
        **kwargs,
    ) -> Tuple[
        "DatasetDict | Dataset | IterableDatasetDict | IterableDataset", str | None
    ]:
        from datasets import Dataset

        # Currently only support spreadsheetbench dataset but can be extended to other datasets in the future
        if dataset_name == "spreadsheetbench":
            folder_path = Path(data_output_path) / SPREADSHEET_BENCH_TRAIN_DATA
            json_path = folder_path / "dataset.json"
            if not os.path.exists(json_path):
                download_and_extract(
                    f"https://github.com/RUCKBReasoning/SpreadsheetBench/raw/refs/heads/main/data/{SPREADSHEET_BENCH_TRAIN_DATA}.tar.gz",
                    data_output_path,
                )
            with open(json_path, "r") as f:
                data = json.load(f)
                for example in data:
                    example["id"] = str(example["id"])  # Ensure IDs are strings
            dataset = Dataset.from_list(data)
            return dataset, str(folder_path)
        return super().load_dataset(dataset_name, **kwargs)

    @classmethod
    def dataset_preprocess(
        cls, example: Any, dataset_path: Optional[str | Path] = None, **kwargs
    ) -> ExcelExample:
        # convert dataset json into ExcelExample (a subclass of StandardizedExample)
        example_id: Optional[str] = example.get("id")
        spreadsheet_path: Optional[str] = example.get("spreadsheet_path")
        instruction: Optional[str] = example.get("instruction")
        instruction_type: Optional[str] = example.get("instruction_type")
        answer_position: Optional[str] = example.get("answer_position")

        if (
            not example_id
            or not spreadsheet_path
            or not instruction
            or not instruction_type
            or not answer_position
        ):
            raise ValueError(
                "Example must contain 'id', 'spreadsheet_path', 'instruction', 'instruction_type', and 'answer_position' fields"
            )
        if not isinstance(spreadsheet_path, str):
            raise TypeError("spreadsheet_path must be a string")

        if dataset_path is None:
            dataset_path = Path(DEFAULT_DATA_OUTPUT_PATH) / SPREADSHEET_BENCH_TRAIN_DATA
        elif not isinstance(dataset_path, (str, Path)):
            raise TypeError("dataset_path must be a str or Path")

        spreadsheet_base_dir = Path(dataset_path) / spreadsheet_path

        if os.path.exists(spreadsheet_base_dir) is False:
            raise FileNotFoundError(
                f"Spreadsheet path {spreadsheet_base_dir} does not exist."
            )

        # File path in the workspace (input spreadsheet will be copied into the workspace at init_rollout)
        input_filename = f"1_{example_id}_input.xlsx"
        output_filename = f"1_{example_id}_output.xlsx"
        ground_truth_filename = f"1_{example_id}_answer.xlsx"

        input_src_path = spreadsheet_base_dir / input_filename
        input_spreadsheet_content = excel_to_str_repr(input_src_path, True)

        prompt = f"""
Instruction: {instruction}
Spreadsheet Path: {input_filename}
Spreadsheet Content: {input_spreadsheet_content}
Instruction Type: {instruction_type} 
Answer Position: {answer_position}
Output Path: {output_filename}"""

        return ExcelExample(
            prompt=prompt.strip(),
            # Ground truth unused in ExcelEnv
            ground_truth=None,
            init_rollout_args={
                "input_src_path": str(input_src_path),
            },
            id=example_id,
            answer_position=answer_position,
            output_filename=output_filename,
            ground_truth_filename=ground_truth_filename,
            spreadsheet_base_dir=str(spreadsheet_base_dir),
        )

    async def init_rollout(self, rollout_id: str, **rollout_args):
        input_src_path: Optional[str] = rollout_args.get("input_src_path")

        if not input_src_path:
            raise ValueError("rollout_args must contain 'input_src_path' field")

        await super().init_rollout(rollout_id, **rollout_args)
        await self.copy_to_workspace(rollout_id, Path(input_src_path))

    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        answer_position: Optional[str] = kwargs.get("answer_position")
        output_filename: Optional[str] = kwargs.get("output_filename")
        ground_truth_filename: Optional[str] = kwargs.get("ground_truth_filename")
        spreadsheet_base_dir: Optional[str] = kwargs.get("spreadsheet_base_dir")

        if (
            not answer_position
            or not output_filename
            or not ground_truth_filename
            or not spreadsheet_base_dir
        ):
            raise ValueError(
                "kwargs must contain 'answer_position', 'output_filename', 'ground_truth_filename', and 'spreadsheet_base_dir' fields"
            )

        # Copy ground truth file to workspace for reward computation
        await self.copy_to_workspace(
            rollout_id, Path(spreadsheet_base_dir) / ground_truth_filename
        )
        return await super().compute_reward(
            rollout_id,
            completion,
            ground_truth,
            answer_position=answer_position,
            output_filename=output_filename,
            ground_truth_filename=ground_truth_filename,
        )


class ExcelEnvLocal(ExcelEnv):
    """Import this env to run environment locally"""

    def __init__(self, num_local_servers: int = 5, **kwargs):
        workdir_path = Path(__file__).parent / "workdir"
        provisioner = LocalProvisioner(
            workdir_path=workdir_path, num_servers=num_local_servers
        )
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)


class ExcelEnvSkypilot(ExcelEnv):
    """Import this env to run environment on any cloud (i.e. AWS / GCP / Azure) with Skypilot"""

    def __init__(
        self,
        cloud: "sky.clouds.Cloud | None" = None,
        num_nodes: int = 2,
        servers_per_node: int = 5,
        **kwargs,
    ):
        if cloud is None:
            import sky

            cloud = sky.Azure()
        workdir_path = Path(__file__).parent / "workdir"
        provisioner = SkypilotProvisioner(
            workdir_path=workdir_path,
            cloud=cloud,
            num_nodes=num_nodes,
            servers_per_node=servers_per_node,
        )
        super().__init__(workdir_path=workdir_path, provisioner=provisioner, **kwargs)
