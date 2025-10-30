from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


class StandardizedExample(TypedDict):
    prompt: str
    ground_truth: Any
    init_rollout_args: Optional[Dict[str, Any]]


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
