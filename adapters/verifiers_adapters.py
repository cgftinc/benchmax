from __future__ import annotations
from functools import wraps

from openai import OpenAI

from envs.base_sandbox import BaseSandbox

from contextvars import ContextVar
import uuid
from typing import Any, Callable, Dict, List, Tuple, Union

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers import Parser, RewardFunc
from verifiers.rubrics import Rubric

from envs.types import RewardFunction
from prompts.tools import parse_hermes_tool_call

_CURRENT_ROLLOUT_ID: ContextVar[str | None] = ContextVar("current_rollout_id", default=None)

def wrapped_reward_func(reward_func: RewardFunction) -> Callable:
    @wraps(reward_func)
    def wrapped_func(completion, answer, **kwargs):
        prompt = kwargs.pop("prompt", "")
        info = kwargs.pop("info", {})
        workspace = kwargs.pop("state", {}).get("workspace", None)
        kwargs = {**kwargs, **info}
        return reward_func(prompt, completion[-1]["content"], answer, workspace, **kwargs)
    return wrapped_func

def verifiers_dataset_mapper(dataset: Dataset) -> Dataset:
    """Map a dataset to the expected format for the Verifiers environment."""
    def map_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.pop("prompt", "")
        ground_truth = example.pop("ground_truth", "")
        return {
            "question": prompt,
            "answer": ground_truth,
            **example,  # Include any other fields as is
        }
    return dataset.map(map_example)

class SandboxRubric(Rubric):
    """Wrap all reward functions registered in a ``BaseSandbox`` so they are
    compatible with the Verifiers evaluation pipeline.

    By default each sandbox reward gets weight **1.0**
    """

    def __init__(
        self,
        sandbox: BaseSandbox,
        *,
        parser: Parser = Parser(),
        extra_funcs: List[RewardFunc] | None = None,
        extra_weights: List[float] | None = None,
    ) -> None:
        super().__init__(funcs=[], weights=[], parser=parser)

        # Add sandbox‑native rewards
        for sandbox_reward_fn in sandbox.reward_funcs:
            self.add_reward_func(wrapped_reward_func(sandbox_reward_fn), weight=1.0)

        # Add any caller‑supplied extra rewards
        extra_funcs = extra_funcs or []
        extra_weights = extra_weights or []
        for fn, w in zip(extra_funcs, extra_weights, strict=False):
            self.add_reward_func(fn, weight=w)

def get_verifiers_environment(
    sandbox: BaseSandbox,
    parser: Parser = Parser(),
    max_turns: int = 10,
    **kwargs
) -> MultiTurnEnv:
    class SandboxVerifierEnv(MultiTurnEnv):
        """MultiTurnEnv-Compatible adapter around *sandbox*."""

        def __init__(self):
            rubric = SandboxRubric(sandbox=sandbox, parser=parser)
            self.sandbox = sandbox

            super().__init__(
                system_prompt=sandbox.get_system_prompt(add_tool_defs=True),
                parser=parser,
                rubric=rubric,
                max_turns=max_turns,
                **kwargs
            )

        def call_tool(self, tool_call: dict, max_chars: int = 10000, **kwargs: Any) -> str:
            try:
                if not isinstance(tool_call, dict):
                    return "Error: Tool command must be a JSON object."

                tool_name = tool_call.get("name")
                if not tool_name:
                    return "Error: Missing 'name' field in tool command."

                args = tool_call.get("arguments", {})
                if not isinstance(args, dict):
                    return f"Error: 'arguments' for {tool_name} must be a JSON object."

                # Retrieve rollout id from context
                rid = _CURRENT_ROLLOUT_ID.get() or "[DEFAULT_ROLLOUT_ID]"
                print(f"Running tool {tool_name} with args: {args} in rollout {rid}")
                result = sandbox.run_tool(rid, tool_name, **args)

                result_str = str(result)
                if 0 < max_chars < len(result_str):
                    result_str = result_str[:max_chars] + "..."
                return result_str
            except Exception as exc:
                return f"Error: {exc}"

        def rollout(
            self,
            client: OpenAI,
            model: str,
            prompt: Union[str, List[Dict[str, Any]]],
            answer: str,
            task: str = "default",
            info: Dict[str, Any] = {},
            sampling_args: Dict[str, Any] = {},
            **kwargs: Any
        ):
            rid = kwargs.pop("rollout_id", None) or str(uuid.uuid4())
            token = _CURRENT_ROLLOUT_ID.set(rid)
            init_rollout_args = info.pop("init_rollout_args", {})
            print(f"Starting rollout {rid} with args: {init_rollout_args} {kwargs}")
            self.sandbox.init_rollout(rid, **init_rollout_args)
            workspace = self.sandbox.get_rollout_workspace(rid) or None
            try:
                completion, state = super().rollout(client, model, prompt, answer, task, info, sampling_args, **kwargs)
                state["workspace"] = workspace
            finally:
                _CURRENT_ROLLOUT_ID.reset(token)
            return completion, state
        
        def is_completed(
            self,
            messages: List[Dict[str, str]],
            state: Dict[str, Any],
            **kwargs: Any
        ) -> bool:
            if messages[-1]['role'] != "assistant":
                return False
            tool_calls = parse_hermes_tool_call(messages[-1]['content'])
            return not tool_calls

        def env_response(self,
            messages: List[Dict[str, str]], 
            state: Dict[str, Any],
            **kwargs: Any
        ) -> Tuple[Dict[str, str], Dict[str, Any]]:
            try:
                tool_call = parse_hermes_tool_call(messages[-1]['content'])[0]
                return {
                    "role": "user",
                    "content": self.call_tool(tool_call),
                }, {}
            except Exception:
                pass
            return {'role': 'user', 'content': "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, {}

        def format_dataset(
            self,
            dataset: Dataset,
            system_prompt: str | None = None,
            few_shot: List[ChatMessage] | None = None,
            question_key: str = "question",
            answer_key: str = "answer"
        ) -> Dataset:
            """Format a dataset for the benchmax environment."""

            # Capture the original examples as a list of dicts
            original_examples = list(dataset)

            # Apply the transformation
            new_dataset = super().format_dataset(
                dataset,
                system_prompt=system_prompt,
                few_shot=few_shot,
                question_key=question_key,
                answer_key=answer_key
            )
            # Inject the missing keys into "info"
            def merge_info(new_item, idx):
                original = original_examples[idx]
                info = {
                    k: v for k, v in original.items()
                    if k not in ["prompt", "answer"]
                }
                return {
                    **new_item,
                    "info": info
                }

            new_dataset = new_dataset.map(merge_info, with_indices=True)
            return new_dataset
    
    return SandboxVerifierEnv()
