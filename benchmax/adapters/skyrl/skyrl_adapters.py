"""Module providing integration between Benchmax environment and SkyRL Gym via Ray actors."""
import inspect
from typing import Any, Dict, Optional
import uuid
from omegaconf import DictConfig
from skyrl_gym import Env
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from benchmax.prompts.tools import parse_hermes_tool_call

import ray


def load_benchmax_env_skyrl(**kwargs) -> Env[str, str]:
    """
    Factory function to create a SkyRL Gym environment that wraps a Benchmax task via a Ray actor.

    Parameters:
        actor_name (str, optional): Name of the remote BenchmaxEnvService actor to contact. Defaults to None.
        env_config (DictConfig): Configuration for the SkyRL env (injected by SkyRL Gym internally).
        extras (dict): Additional parameters, including:
            - max_turns (int): Maximum dialogue turns before forcing termination (default: 4).
            - init_rollout_args (dict): Arguments to initialize the rollout in the remote actor.
            - task (str): Name/identifier of the Benchmax task.
            - ground_truth (Any): Ground truth data for reward computation.

    Returns:
        BenchmaxSkyRLEnv: A BaseTextEnv-compatible environment.
    """
    benchmax_actor_name = kwargs.pop("actor_name", None)

    class BenchmaxSkyRLEnv(BaseTextEnv):
        """
        Text-based RL environment integrating Benchmax via a remote Ray actor.
        Inherits from BaseTextEnv and operates on string-based observations/actions.
        """
        def __init__(
            self, env_config: DictConfig, extras: Dict[str, Any] = {}
        ):
            """
            Initialize the BenchmaxSkyRLEnv.

            Args:
                env_config (DictConfig): Configuration passed by SkyRL Gym.
                extras (dict): Extra settings for initialization and reward calculation.
            """
            self.turns = 0
            self.max_turns = extras.get("max_turns", 4)
            self.extras = extras
            self.chat_history: ConversationType = []
            self.rollout_id = uuid.uuid4().hex
            self.benchmax_env = RemoteBaseEnvProxy(actor_name=benchmax_actor_name)
            self.benchmax_env.init_rollout(**extras["init_rollout_args"])

        def _get_reward(self, action: str) -> float:
            """
            Compute total reward by summing the values returned from the remote Benchmax actor.

            Args:
                action (str): The final text action or answer.

            Returns:
                float: Sum of reward components.
            """
            reward_dict = self.benchmax_env.compute_reward(
                self.extras["task"], action, self.extras["ground_truth"]
            )
            return sum(reward_dict.values())

        def _call_tool(self, tool_call: dict, max_chars: int = 10000) -> str:
            """
            Execute a parsed tool call via the remote Benchmax actor and return its string output.

            Args:
                tool_call (dict): Dictionary with 'name' and 'arguments' keys.
                max_chars (int): Maximum length of the returned string (truncated if exceeded).

            Returns:
                str: The tool output or an error message.
            """
            try:
                if not isinstance(tool_call, dict):
                    return "Error: Tool command must be a JSON object."

                tool_name = tool_call.get("name")
                if not tool_name:
                    return "Error: Missing 'name' field in tool command."

                args = tool_call.get("arguments", {})
                if not isinstance(args, dict):
                    return f"Error: 'arguments' for {tool_name} must be a JSON object."

                result = self.benchmax_env.run_tool(tool_name, **args)
                result_str = str(result)
                if 0 < max_chars < len(result_str):
                    result_str = result_str[:max_chars] + "..."
                return result_str
            except Exception as exc:
                return f"Error: {exc}"

        def step(self, action: str) -> BaseTextEnvStepOutput:
            """
            Process one step of the RL environment: parse tool calls or finalize with reward.

            Args:
                action (str): The assistant's text output.

            Returns:
                BaseTextEnvStepOutput: Contains new observations, reward, done flag, and metadata.
            """
            self.turns += 1
            self.chat_history.append({"role": "assistant", "content": action})
            tool_calls = parse_hermes_tool_call(action)
            error = None
            done = not tool_calls
            reward = 0.0

            if done:
                reward = self._get_reward(action)
                return BaseTextEnvStepOutput(
                    observations=[], reward=reward, done=done,
                    metadata={}, postprocessed_action=action
                )

            # Only first tool call is executed
            try:
                tool_call = tool_calls[0]
                observation = self._call_tool(tool_call)
            except Exception as e:
                error = str(e)
                observation = None

            # Wrap the observation as a user message or error
            if observation:
                new_obs = {"role": "user", "content": observation}
            elif error:
                new_obs = {"role": "user", "content": error}
            else:
                new_obs = None

            if new_obs:
                self.chat_history.append(new_obs)

            return BaseTextEnvStepOutput(
                observations=[new_obs] if new_obs else [],
                reward=reward, done=done,
                metadata={}, postprocessed_action=action
            )

    return BenchmaxSkyRLEnv(**kwargs)


class RemoteBaseEnvProxy:
    """
    Proxy for invoking a remote BenchmaxEnvService Ray actor via synchronous calls.

    Attributes:
        rollout_id (str): Unique identifier for the current rollout session.
        auto_cleanup (bool): Whether to automatically cleanup the remote workspace.
    """
    def __init__(
        self,
        actor_name: str = "BenchmaxEnvService",
        rollout_id: Optional[str] = None,
        auto_cleanup: bool = True,
    ):
        self._actor = ray.get_actor(actor_name)
        self.rollout_id = rollout_id or uuid.uuid4().hex
        self.auto_cleanup = auto_cleanup

    def __getattr__(self, name: str):
        """
        Forward method calls to the remote actor, injecting rollout_id when needed.
        """
        def _rpc(*args, **kwargs):
            if name in ("list_tools", "ping", "shutdown"):
                return ray.get(getattr(self._actor, f"{name}").remote(*args, **kwargs))
            return ray.get(getattr(self._actor, f"{name}").remote(self.rollout_id, *args, **kwargs))
        return _rpc

    def cleanup_rollout(self, keep_workspace: bool = True) -> bool:
        """
        Cleanup the remote rollout workspace.

        Args:
            keep_workspace (bool): If False, delete the workspace; else keep it.

        Returns:
            bool: Success of cleanup operation.
        """
        return self.__getattr__("cleanup_rollout")(keep_workspace=keep_workspace)


def expose(attr_name: str):
    """
    Decorator to expose methods of BenchmaxEnvActor by forwarding to self.env.
    """
    async def wrapper(self, *args, **kwargs):
        method = getattr(self.env, attr_name)
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
    return wrapper


@ray.remote(name="BenchmaxEnvService", lifetime="detached", num_cpus=1)
class BenchmaxEnvActor:
    """
    Ray actor that wraps a local environment class and exposes its methods remotely.
    """
    def __init__(self, env_cls: type, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env = env_cls(**(env_kwargs or {}))

    # Expose a fixed list of methods via the expose decorator
    for _m in (
        "list_tools",
        "init_rollout",
        "run_tool",
        "cleanup_rollout",
        "get_rollout_workspace",
        "shutdown",
        "ping",
        "dataset_preprocess",
        "load_dataset",
        "compute_reward",
        "get_system_prompt",
    ):
        locals()[_m] = expose(_m)