from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, TypedDict


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None

class RewardFunction(Protocol):
    """Function that evaluates model interactions"""
    def __call__(
        self,
        prompt: str,         # Input prompt given to the model
        completion: str,     # Model's generated completion/response
        ground_truth: Any,   # Expected/correct output to compare against
        workspace: Path,     # Path to rollout's workspace with tool outputs
        **kwargs: Any        # Additional context for reward computation
    ) -> float:             # Reward score (typically in range [0, 1])
        ...

class DatasetProcResult(TypedDict, total=False):
    rollout_init_args: Dict[str, Any]
    # other possible keys can be added here as needed

class DatasetProcFunction(Protocol):
    """Function that processes a dataset example"""
    def __call__(self, example: Dict[str, Any]) -> DatasetProcResult:
        ...
