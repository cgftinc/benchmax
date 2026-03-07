from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

Completion = List[Dict[str, Any]]


class StandardizedExample(TypedDict):
    prompt: str | List[Dict[str, Any]]
    ground_truth: Any
    init_rollout_args: Optional[Dict[str, Any]]


@dataclass
class ToolDefinition:
    """Definition of a tool's interface"""

    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
