from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

from envs.types import RewardFunction, ToolDefinition, StandardizedExample
from prompts.tools import render_tools_prompt

class BaseEnv(ABC):
    """Base benchmax environment for tool execution and reward computation"""
    
    system_prompt: str = ""
    reward_funcs: List[RewardFunction] = []

    # Override this method if your example does not match the default structure
    def dataset_preprocess(self, example: Any) -> StandardizedExample:
        """
        Preprocess a single dataset example into a dict with keys:
        - "prompt": str
        - "ground_truth": Any
        - "init_rollout_args": Optional[Dict[str, Any]]
        """
        return {
            "prompt": example.get("prompt", ""),
            "ground_truth": example.get("ground_truth", None),
            "init_rollout_args": example.get("init_rollout_args", {})
        }

    @abstractmethod
    def list_tools(self) -> List[ToolDefinition]:
        """Return list of available tools"""
        pass
    
    @abstractmethod
    def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        """Execute named tool in rollout context with given arguments"""
        pass

    @abstractmethod
    def init_rollout(self, rollout_id: str, **rollout_args) -> None:
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
        for func in self.reward_funcs:
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
    
    def get_system_prompt(self, add_tool_defs: bool = False) -> str:
        """Get system prompt. To add tool definitions, set add_tool_defs to True."""
        if add_tool_defs:
            return render_tools_prompt(self.list_tools(), self.system_prompt or "")
        else:
            return self.system_prompt
