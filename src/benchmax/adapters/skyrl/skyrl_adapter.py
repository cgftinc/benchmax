"""Module providing integration between Benchmax environment and SkyRL Gym via Ray actors."""

import logging
import traceback
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, cast
import uuid
from omegaconf import DictConfig
import ray
from skyrl_gym import Env
from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from ray.actor import ActorProxy

from benchmax.adapters.benchmax_wrapper import (
    BenchmaxEnv,
    BenchmaxEnvActor,
    BenchmaxEnvWrapper,
)
from benchmax.envs.base_env import BaseEnv
from benchmax.prompts.tools import parse_hermes_tool_call

logger = logging.getLogger(__name__)


# Main entrypoint to load benchmax in SkyRL Gym
def load_benchmax_env_skyrl(actor: ActorProxy[BenchmaxEnv], **kwargs) -> Env[str, str]:
    """
    Factory function to create a SkyRL Gym environment that wraps a Benchmax task via a Ray actor.

    Parameters:
        actor (ActorProxy[BenchmaxEnv]): Ray actor proxy for the Benchmax environment
        **kwargs: Additional keyword arguments passed to BenchmaxSkyRLEnv, including:
            - env_config (DictConfig): Configuration passed by SkyRL Gym
            - extras (Dict[str, Any]): Additional parameters such as:
                - max_turns (int): Maximum dialogue turns before forcing termination (default: 4)
                - init_rollout_args (dict): Arguments to initialize the rollout in the remote actor
                - task (str): Name/identifier of the Benchmax task
                - ground_truth (Any): Ground truth data for reward computation

    Returns:
        BenchmaxSkyRLEnv: A BaseTextEnv-compatible environment.
    """
    return BenchmaxSkyRLEnv(actor=actor, **kwargs)


class BenchmaxSkyRLEnv(BaseTextEnv):
    """
    Text-based RL environment integrating Benchmax via a remote Ray actor.
    Inherits from BaseTextEnv and operates on string-based observations/actions.
    """

    def __init__(
        self,
        actor: ActorProxy[BenchmaxEnv],
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        """
        Initialize the BenchmaxSkyRLEnv.

        Args:
            actor (ActorProxy[BenchmaxEnv]): Ray actor proxy for Benchmax environment
            env_config (DictConfig): Configuration passed by SkyRL Gym.
            extras (Dict[str, Any]): Additional parameters, including:
                - max_turns (int): Maximum dialogue turns before forcing termination (default: 4).
                - init_rollout_args (dict): Arguments to initialize the rollout in the remote actor.
                - task (str): Name/identifier of the Benchmax task.
                - ground_truth (Any): Ground truth data for reward computation.
        """
        self._turns = 0
        self._extras = extras
        self._max_turns = extras.get("max_turns", 4)
        self._ground_truth = extras.get("ground_truth")
        self._init_rollout_args = extras.get("init_rollout_args") or {}
        self._encountered_error = False
        self._chat_history: ConversationType = []
        self._rollout_id = uuid.uuid4().hex
        self._benchmax_env = BenchmaxEnvWrapper(actor)

        logger.debug(f"Initialized BenchmaxSkyRLEnv with rollout_id={self._rollout_id}")

    def _get_reward(self, action: str) -> float:
        """
        Compute total reward by summing the values returned from the remote Benchmax actor.

        Args:
            action (str): The final text action or answer.

        Returns:
            float: Sum of reward components.
        """
        logger.debug(f"Computing reward for rollout_id={self._rollout_id}")
        filtered_extras = {
            k: v
            for k, v in self._extras.items()
            if k not in ("rollout_id", "completion", "ground_truth")
        }
        reward_dict = self._benchmax_env.compute_reward_sync(
            rollout_id=self._rollout_id,
            completion=action,
            ground_truth=self._ground_truth,
            **filtered_extras,
        )
        total_reward = sum(reward_dict.values())
        logger.debug(
            f"Computed reward for rollout_id={self._rollout_id}: {total_reward} (components: {reward_dict})"
        )
        return total_reward

    def _call_tool(self, tool_call: dict, max_chars: int = 10000) -> str:
        """
        Execute a parsed tool call via the remote Benchmax actor and return its string output.

        Args:
            tool_call (dict): Dictionary with 'name' and 'arguments' keys.
            max_chars (int): Maximum length of the returned string (truncated if exceeded).

        Returns:
            str: The tool output or an error message.
        """
        if not isinstance(tool_call, dict):
            logger.warning(f"Tool call is not a dict: {type(tool_call)}")
            return "Error: Tool command must be a JSON object."

        tool_name = tool_call.get("name")
        if not tool_name:
            logger.warning(f"Tool call missing 'name' field: {tool_call}")
            return "Error: Missing 'name' field in tool command."

        tool_args = tool_call.get("arguments", {})
        if not isinstance(tool_args, dict):
            logger.warning(
                f"Tool arguments for {tool_name} is not a dict: {type(tool_args)}"
            )
            return f"Error: 'arguments' for {tool_name} must be a JSON object."

        logger.debug(
            f"Calling tool {tool_name} with args {tool_args} for rollout_id={self._rollout_id}"
        )
        result = self._benchmax_env.run_tool_sync(
            rollout_id=self._rollout_id, tool_name=tool_name, **tool_args
        )
        result_str = str(result)

        if 0 < max_chars < len(result_str):
            logger.debug(
                f"Tool {tool_name} output truncated from {len(result_str)} to {max_chars} chars"
            )
            result_str = result_str[:max_chars] + "..."

        logger.debug(f"Tool {tool_name} completed for rollout_id={self._rollout_id}")
        return result_str

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        logger.debug(
            f"Initializing rollout_id={self._rollout_id} with args: {self._init_rollout_args}"
        )
        try:
            self._benchmax_env.init_rollout_sync(
                rollout_id=self._rollout_id, **self._init_rollout_args
            )
        except Exception:
            self._encountered_error = True
        return super().init(prompt)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process one step of the RL environment: parse tool calls or finalize with reward.

        Args:
            action (str): The assistant's text output.

        Returns:
            BaseTextEnvStepOutput: Contains new observations, reward, done flag, and metadata.
        """
        # If rollout has encountered error, do not proceed
        if self._encountered_error:
            logger.warning(
                f"Step called on rollout_id={self._rollout_id} after error encountered, returning done"
            )
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={},
                postprocessed_action=action,
            )

        self._turns += 1
        logger.debug(
            f"Step {self._turns}/{self._max_turns} for rollout_id={self._rollout_id}"
        )

        self._chat_history.append({"role": "assistant", "content": action})
        tool_calls = parse_hermes_tool_call(action)
        done = not tool_calls
        reward = 0.0

        try:
            if done:
                logger.debug(
                    f"No tool calls found, finalizing rollout_id={self._rollout_id}"
                )
                reward = self._get_reward(action)
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=reward,
                    done=True,
                    metadata={},
                    postprocessed_action=action,
                )

            # Only first tool call is executed
            tool_call = tool_calls[0]
            if len(tool_calls) > 1:
                logger.debug(
                    f"Multiple tool calls detected ({len(tool_calls)}), executing only the first"
                )

            observation = self._call_tool(tool_call)

        except Exception as e:
            self._encountered_error = True
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(
                f"Exception in step for rollout_id={self._rollout_id}:\n{tb_str}"
            )
            self.close()
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={},
                postprocessed_action=action,
            )

        # Wrap the observation as a user message or error
        new_obs = {"role": "user", "content": observation}
        self._chat_history.append({"role": "user", "content": observation})

        logger.debug(
            f"Step {self._turns} completed for rollout_id={self._rollout_id}, continuing"
        )
        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=done,
            metadata={},
            postprocessed_action=action,
        )

    def close(self):
        logger.debug(f"Closing rollout_id={self._rollout_id}")
        self._benchmax_env.release_rollout_sync(self._rollout_id)


T = TypeVar("T", bound=BaseEnv)


def get_or_create_benchmax_env_actor(
    env_cls: Type[T],
    env_kwargs: Optional[Dict[str, Any]] = None,
    actor_name: str = "BenchmaxEnvService",
    num_cpus: int = 3,
) -> ActorProxy[BenchmaxEnv]:
    """
    Create the benchmax BaseEnv actor if not already running, or return the existing one.

    Args:
        env_cls: The benchmax BaseEnv subclass to wrap (e.g., MathEnv).
        env_kwargs: Keyword args to pass to the env constructor.
        actor_name: Name of the actor in the Ray namespace.
        num_cpus: Number of CPUs to allocate to the actor.

    Returns:
        A handle to the BaseEnv actor.
    """
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        # Create a new detached actor
        return cast(
            ActorProxy[BenchmaxEnv],
            BenchmaxEnvActor.options(
                name=actor_name,
                lifetime="detached",
                get_if_exists=True,
                num_cpus=num_cpus,
                runtime_env={"py_executable": "uv run --isolated --group skypilot"},
            ).remote(env_cls, env_kwargs or {}),
        )


def cleanup_actor(actor: Optional[ActorProxy[BenchmaxEnv]]) -> None:
    """Shutdown and kill a benchmax actor."""
    if actor is None:
        return

    try:
        obj_ref: ray.ObjectRef[Any] = actor.shutdown.remote()
        ray.get(obj_ref, timeout=60)
    except Exception as e:
        logger.error(f"Actor shutdown failed: {e}")

    try:
        ray.kill(actor)
    except Exception as e:
        logger.error(f"Actor kill failed: {e}")
