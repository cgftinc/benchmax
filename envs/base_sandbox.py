from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Protocol
from pathlib import Path

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

class BaseSandbox(ABC):
    """Base sandbox for tool execution and reward computation"""
    
    def __init__(self) -> None:
        # Private attributes for internal use only
        self._reward_funcs: List[RewardFunction] = []
    
    @abstractmethod
    def init_tools(self, config: Dict[str, Any], allowed_tools: Optional[List[str]] = None) -> None:
        """Set up tool environment with given config and optional tool allowlist"""
        pass

    @abstractmethod
    def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        pass
    
    @abstractmethod
    def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        pass

    @abstractmethod
    def init_rollout(self, rollout_id: str) -> None:
        """Initialize resources for a new rollout"""
        pass
    
    @abstractmethod
    def cleanup_rollout(self, rollout_id: str) -> None:
        """Clean up resources for a rollout"""
        pass

    @abstractmethod
    def get_rollout_workspace(self, rollout_id: str) -> Path:
        """Get the workspace path for a specific rollout"""
        pass

    def add_reward_func(self, func: RewardFunction) -> None:
        """Add a reward computation function"""
        self._reward_funcs.append(func)
    
    def compute_reward(
        self,
        rollout_id: str,
        prompt: str,
        completion: str, 
        ground_truth: Any,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Compute rewards using registered functions
        
        Returns dict mapping reward function names to their computed scores.
        """
        workspace = self.get_rollout_workspace(rollout_id)
        if workspace is None:
            raise ValueError(f"No workspace found for rollout {rollout_id}")
            
        results: Dict[str, float] = {}
        for func in self._reward_funcs:
            try:
                # Get function name, falling back to string representation if not available
                func_name = getattr(func, "__name__", str(func))
                results[func_name] = func(
                    prompt=prompt,
                    completion=completion,
                    ground_truth=ground_truth,
                    workspace=workspace,
                    **kwargs
                )
            except Exception as e:
                # Use same function name resolution
                func_name = getattr(func, "__name__", str(func))
                results[func_name] = float('nan')
                print(f"[WARN] reward {func_name} failed: {e}")
        return results
