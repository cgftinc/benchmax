"""Starter castform RL environment — a single-turn QA task.

`castform setup` drops this in so your very first `castform validate` is green out
of the box. The reward is discriminating: a rollout scores 1.0 only when the
model's final answer contains the row's `ground_truth` string, else 0.0 — so the
cheap validation model nails the easy rows and misses the hard ones. That spread
is the point: a reward that never varies gives training no gradient to learn from.

The shipped dataset is a deliberate easy/hard mix (the hard rows are large exact
multiplications the cheap model gets wrong) so the reward varies. To pick hard
rows for YOUR task, filter candidates by what the model already gets wrong — see
the difficulty-filtering recipe in the generate-data skill.

Single-turn, no tools, cheap-model friendly. Edit the system prompt, the reward,
and train_dataset.jsonl / eval_dataset.jsonl to fit your task — see the
design-environment and generate-data skills.
"""

from __future__ import annotations

from benchmax.envs.base_env import BaseEnv


def _last_answer(messages) -> str:
    """The model's final text answer (last assistant turn).

    Inlined on purpose — there is NO importable `last_answer` helper, so
    `from … import last_answer` would ImportError. For multi-turn rollouts that
    span several assistant turns, use
    `from benchmax.envs.reward_helpers import extract_completion_text` instead.
    """
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("content"):
            return m["content"]
    return ""


class StarterEnv(BaseEnv):
    system_prompt = (
        "You are a precise assistant. Answer concisely and exactly as asked."
    )

    async def list_tools(self):
        return []  # [] = single-turn, no tools

    async def run_tool(self, rollout_id, tool_name, **tool_args):
        # Never reached while list_tools returns []; add real tools to go multi-turn.
        raise NotImplementedError("StarterEnv is single-turn (list_tools is []).")

    async def compute_reward(self, rollout_id, messages, task, **kwargs):
        # correct = the row's ground_truth appears in the model's final answer
        # (case-insensitive). Return positive scores only; every component is
        # summed into one scalar. NOTE: under `castform validate --local-only` the
        # simulated final turn IS the ground_truth, so this reads a constant 1.0 —
        # that's expected; real signal/variance comes from the remote rollout path.
        ground_truth = str((task or {}).get("ground_truth", "")).strip().lower()
        answer = _last_answer(messages).lower()
        correct = 1.0 if ground_truth and ground_truth in answer else 0.0
        return {"correct": correct}
