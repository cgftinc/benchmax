from __future__ import annotations
from pathlib import Path
import ray
from ray.actor import ActorClass, ActorProxy
from typing import Dict, List, Any, Optional, Type, Union

from benchmax.envs.base_env import BaseEnv

# 5 minutes timeout in seconds
RAY_GET_TIMEOUT = 300


class BenchmaxEnv:
    """
    Async Ray actor that instantiates Benchmax env and exposes its async API.
    All methods are async def so other Ray actors can await them.
    """

    def __init__(self, env_cls: Type[BaseEnv], env_kwargs: dict):
        self._env = env_cls(**env_kwargs)

    @ray.method
    async def shutdown(self) -> None:
        return await self._env.shutdown()

    @ray.method
    async def list_tools(self) -> List[Any]:
        return await self._env.list_tools()

    @ray.method
    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        return await self._env.run_tool(
            rollout_id=rollout_id, tool_name=tool_name, **tool_args
        )

    @ray.method
    async def init_rollout(self, rollout_id: str, **rollout_args: Any) -> None:
        return await self._env.init_rollout(rollout_id=rollout_id, **rollout_args)

    @ray.method
    async def release_rollout(self, rollout_id: str) -> None:
        return await self._env.release_rollout(rollout_id)

    @ray.method
    async def copy_to_workspace(
        self, rollout_id: str, src_path: Path, dst_filename: Optional[str] = None
    ) -> None:
        return await self._env.copy_to_workspace(
            rollout_id=rollout_id, src_path=src_path, dst_filename=dst_filename
        )

    @ray.method
    async def copy_content_to_workspace(
        self, rollout_id: str, src_content: Union[str, bytes], dst_filename: str
    ) -> None:
        return await self._env.copy_content_to_workspace(
            rollout_id=rollout_id, src_content=src_content, dst_filename=dst_filename
        )

    @ray.method
    async def copy_from_workspace(
        self, rollout_id: str, src_filename: str, dst_path: Path
    ) -> None:
        return await self._env.copy_from_workspace(
            rollout_id=rollout_id, src_filename=src_filename, dst_path=dst_path
        )

    @ray.method
    async def compute_reward(
        self, rollout_id: str, completion: str, ground_truth: Any, **kwargs: Any
    ) -> Dict[str, float]:
        return await self._env.compute_reward(
            rollout_id=rollout_id,
            completion=completion,
            ground_truth=ground_truth,
            **kwargs,
        )

    @ray.method
    async def get_system_prompt(self, add_tool_defs: bool = True) -> str:
        return await self._env.get_system_prompt(add_tool_defs=add_tool_defs)


# Create the actor class using ray.remote() instead of @ray.remote decorator
BenchmaxEnvActor: ActorClass[BenchmaxEnv] = ray.remote(BenchmaxEnv)


class BenchmaxEnvWrapper:
    """
    Wrapper around a BenchmaxEnvActor Ray actor.
    Exposes both async and sync methods for flexibility.

    Async methods can be awaited: await wrapper.list_tools()
    Sync methods block until complete: wrapper.list_tools_sync()
    """

    def __init__(self, actor: ActorProxy[BenchmaxEnv]):
        self._actor = actor

    def get_system_prompt_sync(self, add_tool_defs: bool = True) -> str:
        """Sync method to get system prompt with options."""
        obj_ref: ray.ObjectRef[str] = self._actor.get_system_prompt.remote(
            add_tool_defs=add_tool_defs  # type: ignore
        )
        return ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    async def get_system_prompt(self, add_tool_defs: bool = True) -> str:
        """Async method to get system prompt."""
        obj_ref: ray.ObjectRef[str] = self._actor.get_system_prompt.remote(
            add_tool_defs=add_tool_defs  # type: ignore
        )
        return await obj_ref

    # === List Tools ===
    async def list_tools(self) -> List[Any]:
        """Async method to list available tools."""
        obj_ref: ray.ObjectRef[List[Any]] = self._actor.list_tools.remote()  # type: ignore
        return await obj_ref

    def list_tools_sync(self) -> List[Any]:
        """Sync method to list available tools."""
        obj_ref: ray.ObjectRef[List[Any]] = self._actor.list_tools.remote()  # type: ignore
        return ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Shutdown ===
    async def shutdown(self) -> None:
        """Async method to shutdown the environment."""
        obj_ref: ray.ObjectRef[Any] = self._actor.shutdown.remote()
        await obj_ref

    def shutdown_sync(self) -> None:
        """Sync method to shutdown the environment."""
        obj_ref: ray.ObjectRef[Any] = self._actor.shutdown.remote()
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Rollout Lifecycle ===
    async def init_rollout(self, rollout_id: str, **rollout_args: Any) -> None:
        """Async method to initialize a rollout."""
        obj_ref: ray.ObjectRef[Any] = self._actor.init_rollout.remote(
            rollout_id, **rollout_args
        )
        await obj_ref

    def init_rollout_sync(self, rollout_id: str, **rollout_args: Any) -> None:
        """Sync method to initialize a rollout."""
        obj_ref: ray.ObjectRef[Any] = self._actor.init_rollout.remote(
            rollout_id, **rollout_args
        )
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    async def release_rollout(self, rollout_id: str) -> None:
        obj_ref: ray.ObjectRef[Any] = self._actor.release_rollout.remote(rollout_id)
        await obj_ref

    def release_rollout_sync(self, rollout_id: str) -> None:
        obj_ref: ray.ObjectRef[Any] = self._actor.release_rollout.remote(rollout_id)
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Run Tool ===
    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        """Async method to run a tool."""
        obj_ref: ray.ObjectRef[Any] = self._actor.run_tool.remote(
            rollout_id, tool_name, **tool_args
        )
        return await obj_ref

    def run_tool_sync(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        """Sync method to run a tool."""
        obj_ref: ray.ObjectRef[Any] = self._actor.run_tool.remote(
            rollout_id, tool_name, **tool_args
        )
        return ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Copy to Workspace ===
    async def copy_to_workspace(
        self,
        rollout_id: str,
        src_path: Union[str, Path],
        dst_filename: Optional[str] = None,
    ) -> None:
        """Async method to copy file to workspace."""
        obj_ref: ray.ObjectRef[str] = self._actor.copy_to_workspace.remote(
            rollout_id,
            Path(src_path),
            dst_filename=dst_filename,  # type: ignore
        )
        await obj_ref

    def copy_to_workspace_sync(
        self,
        rollout_id: str,
        src_path: Union[str, Path],
        dst_filename: Optional[str] = None,
    ) -> None:
        """Sync method to copy file to workspace."""
        obj_ref: ray.ObjectRef[None] = self._actor.copy_to_workspace.remote(
            rollout_id,
            Path(src_path),
            dst_filename=dst_filename,  # type: ignore
        )
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Copy Content to Workspace ===
    async def copy_content_to_workspace(
        self,
        rollout_id: str,
        src_content: Union[str, bytes],
        dst_filename: str,
    ) -> None:
        """Async method to copy content to workspace."""
        obj_ref: ray.ObjectRef[None] = self._actor.copy_content_to_workspace.remote(
            rollout_id,
            src_content,
            dst_filename=dst_filename,  # type: ignore
        )
        await obj_ref

    def copy_content_to_workspace_sync(
        self,
        rollout_id: str,
        src_content: Union[str, bytes],
        dst_filename: str,
    ) -> None:
        """Sync method to copy content to workspace."""
        obj_ref: ray.ObjectRef[None] = self._actor.copy_content_to_workspace.remote(
            rollout_id,
            src_content,
            dst_filename=dst_filename,  # type: ignore
        )
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Copy from Workspace ===
    async def copy_from_workspace(
        self,
        rollout_id: str,
        src_filename: str,
        dst_path: Union[str, Path],
    ) -> None:
        """Async method to copy file from workspace."""
        obj_ref: ray.ObjectRef[Any] = self._actor.copy_from_workspace.remote(
            rollout_id, src_filename, Path(dst_path)
        )
        await obj_ref

    def copy_from_workspace_sync(
        self,
        rollout_id: str,
        src_filename: str,
        dst_path: Union[str, Path],
    ) -> None:
        """Sync method to copy file from workspace."""
        obj_ref: ray.ObjectRef[Any] = self._actor.copy_from_workspace.remote(
            rollout_id, src_filename, Path(dst_path)
        )
        ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)

    # === Compute Reward ===
    async def compute_reward(
        self,
        rollout_id: str,
        completion: str,
        ground_truth: Any,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Async method to compute reward."""
        obj_ref: ray.ObjectRef[Dict[str, float]] = self._actor.compute_reward.remote(
            rollout_id, completion, ground_truth, **kwargs
        )  # type: ignore
        return await obj_ref

    def compute_reward_sync(
        self,
        rollout_id: str,
        completion: str,
        ground_truth: Any,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Sync method to compute reward."""
        obj_ref: ray.ObjectRef[Dict[str, float]] = self._actor.compute_reward.remote(
            rollout_id, completion, ground_truth, **kwargs
        )  # type: ignore
        return ray.get(obj_ref, timeout=RAY_GET_TIMEOUT)
