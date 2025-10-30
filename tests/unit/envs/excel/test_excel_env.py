"""
Unit tests for ExcelEnv.

All tests are fast with no external service calls.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

from benchmax.envs.excel.excel_env import ExcelEnv
from benchmax.envs.excel.workdir.reward_fn import spreadsheet_comparison_reward
from benchmax.envs.mcp.provisioners.base_provisioner import BaseProvisioner


# Fixtures
@pytest.fixture(scope="session")
def test_xlsx_path() -> str:
    return str(Path(__file__).parent / "test_inputs" / "test.xlsx")


@pytest.fixture
def excel_env(tmp_path: Path) -> ExcelEnv:
    """Fixture to create an ExcelEnv instance without initializing the parent."""
    env = ExcelEnv(
        workdir_path=tmp_path,
        provisioner=Mock(spec=BaseProvisioner),
        provision_at_init=False,
    )
    env._servers_provisioned = True
    return env


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a fake dataset folder with a sample input file.

    The ExcelEnv expects the dataset layout to contain a folder (spreadsheet_path)
    with files like `1_<id>_input.xlsx`. We touch a file to satisfy existence
    checks; reading is patched in tests that need it.
    """
    base = tmp_path / "dataset_root"
    sheet_dir = base / "sheet_folder"
    sheet_dir.mkdir(parents=True)

    # Create a dummy input file path that ExcelEnv will look for
    (sheet_dir / "1_42_input.xlsx").write_text("dummy")

    # Also create an answer file (not strictly required for dataset_preprocess)
    (sheet_dir / "1_42_answer.xlsx").write_text("dummy-answer")

    return base


@pytest.fixture
def mock_mcp_client() -> Mock:
    return Mock()


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestDatasetPreprocess:
    """Tests for ExcelEnv.dataset_preprocess."""

    def test_valid_example(
        self, mock_dataset: Path, test_xlsx_path: str
    ) -> None:
        """Valid example returns an ExcelExample with expected fields."""
        example: Dict[str, Any] = {
            "id": "42",
            "spreadsheet_path": test_xlsx_path,
            "instruction": "Fill cell A1 with 10",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }

        with patch(
            "benchmax.envs.excel.excel_env.excel_to_str_repr",
            return_value="A1=10",
        ):
            result = ExcelEnv.dataset_preprocess(example, dataset_path=mock_dataset)

        # Result is a mapping-like StandardizedExample (TypedDict / dataclass)
        assert result is not None
        assert "Fill cell A1 with 10" in result["prompt"]
        assert "1_42_input.xlsx" in result["prompt"]
        assert result["init_rollout_args"] is not None
        assert result["init_rollout_args"]["input_src_path"].endswith("1_42_input.xlsx")
        assert result["answer_position"] == "A1"
        assert result["output_filename"] == "1_42_output.xlsx"

    def test_missing_fields_raise(self, excel_env: ExcelEnv) -> None:
        """Missing required fields should raise ValueError."""
        example: Dict[str, Any] = {"id": "1", "instruction": "do"}

        with pytest.raises(ValueError):
            excel_env.dataset_preprocess(example)

    def test_non_string_spreadsheet_path_raises(
        self, mock_dataset: Path
    ) -> None:
        """Non-string spreadsheet_path should raise ValueError."""
        example: Dict[str, Any] = {
            "id": "1",
            "spreadsheet_path": 123,  # invalid type
            "instruction": "x",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }

        with pytest.raises(TypeError):
            ExcelEnv.dataset_preprocess(example, dataset_path=mock_dataset)

    def test_missing_spreadsheet_folder_raises(
        self, tmp_path: Path
    ) -> None:
        """If the spreadsheet folder does not exist under the dataset path, raise FileNotFoundError."""
        dataset_path = tmp_path / "some_other_root"
    
        example: Dict[str, Any] = {
            "id": "99",
            "spreadsheet_path": "no_such_folder",
            "instruction": "x",
            "instruction_type": "Cell-Level Manipulation",
            "answer_position": "A1",
        }

        with pytest.raises(FileNotFoundError):
            ExcelEnv.dataset_preprocess(example, dataset_path=dataset_path)


class TestRewardComputation:
    """Test reward computation for spreadsheet comparison."""

    def test_exact_match_returns_one(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        # Patch compare_excel_cells to return match=True
        with patch(
            "benchmax.envs.excel.workdir.reward_fn.compare_excel_cells",
            return_value=(True, None),
        ):
            score = spreadsheet_comparison_reward(
                completion="irrelevant",
                ground_truth={},
                mcp_client=mock_mcp_client,
                workspace=mock_workspace,
                answer_position="A1",
                output_filename="out.xlsx",
                ground_truth_filename="gt.xlsx",
            )

        assert score == 1.0

    def test_mismatch_returns_zero(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        with patch(
            "benchmax.envs.excel.workdir.reward_fn.compare_excel_cells",
            return_value=(False, None),
        ):
            score = spreadsheet_comparison_reward(
                completion="irrelevant",
                ground_truth={},
                mcp_client=mock_mcp_client,
                workspace=mock_workspace,
                answer_position="A1",
                output_filename="out.xlsx",
                ground_truth_filename="gt.xlsx",
            )

        assert score == 0.0

    def test_missing_kwargs_raises(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        with pytest.raises(ValueError):
            spreadsheet_comparison_reward(
                completion="x",
                ground_truth={},
                mcp_client=mock_mcp_client,
                workspace=mock_workspace,
                # missing answer_position, output_filename, ground_truth_filename
            )

    def test_compare_raises_returns_zero(
        self, mock_mcp_client: Mock, mock_workspace: Path
    ) -> None:
        with patch(
            "benchmax.envs.excel.workdir.reward_fn.compare_excel_cells",
            side_effect=Exception("boom"),
        ):
            score = spreadsheet_comparison_reward(
                completion="irrelevant",
                ground_truth={},
                mcp_client=mock_mcp_client,
                workspace=mock_workspace,
                answer_position="A1",
                output_filename="out.xlsx",
                ground_truth_filename="gt.xlsx",
            )

        assert score == 0.0
