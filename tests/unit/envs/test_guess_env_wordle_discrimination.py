"""GuessEnv fixture — the wordle-discrimination axis of the redesign.

The Example-as-first-class redesign exists primarily to fix one pathology:
envs where every row shares a generic seed prompt template ("guess my
number", "solve the wordle puzzle") but differs in the per-row ground
truth would, under the old prompt_text-hash grouping, collapse into a
single group. The new ``canonical_example_id(prompt_messages, task)`` hash
includes ``task``, so identical seeds with distinct tasks correctly land
in distinct groups.

This file pins that property at the env-level with a fixture env that's
explicitly built around the pattern. It complements the
``canonical_example_id`` golden-hash tests (which prove the hash works in
isolation) by proving the env-level integration: that ``dataset_preprocess``
threads task through correctly so the canonical hash discriminates as
expected.

If this test ever turns red, the wordle case is broken end-to-end and the
CPU/GPU smoke would surface the same bug as "many distinct dataset rows
collapsing into one group_id".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import canonical_example_id, make_example
from benchmax.envs.types import Example, Messages, ToolDefinition


SYSTEM_PROMPT = (
    "I'm thinking of a number between 1 and 100. Try to guess it. "
    "Respond with just the number."
)


class GuessEnv(BaseEnv):
    """Every row shares the same seed prompt; task carries the target.

    Two rows with the same target → same example_id (legitimately the same
    example, deduplicated). Two rows with different targets → different
    example_ids (the redesign's primary correctness property).
    """

    system_prompt: str = SYSTEM_PROMPT

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        # Constant seed across all rows — the wordle pathology.
        return make_example(
            prompt_messages=[
                {"role": "user", "content": "I have a number in mind. What is it?"},
            ],
            task={"target": int(example["target"])},
            system_prompt=cls.system_prompt,
        )

    async def list_tools(self) -> List[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return ""

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        target = (task or {}).get("target")
        if target is None or not messages:
            return {"correctness": 0.0}
        guess_text = messages[-1].get("content", "").strip()
        try:
            return {"correctness": 1.0 if int(guess_text) == target else 0.0}
        except ValueError:
            return {"correctness": 0.0}


# ---------------------------------------------------------------------------
# Discrimination tests
# ---------------------------------------------------------------------------


def test_distinct_targets_produce_distinct_example_ids():
    """The primary redesign property: same seed + different task → different id."""
    rows = [{"target": n} for n in [7, 42, 99]]
    examples = [GuessEnv.dataset_preprocess(r) for r in rows]
    ids = [ex["id"] for ex in examples]

    # All three ids must differ from each other (3 distinct rows).
    assert len(set(ids)) == 3, f"expected 3 distinct ids, got {ids}"

    # And the prompt_messages must in fact be identical across rows —
    # that's what makes this a useful test (the property would be trivial
    # if rows varied in prompt_messages too).
    seed_strings = {tuple((m["role"], m["content"]) for m in ex["prompt_messages"]) for ex in examples}
    assert len(seed_strings) == 1, "prompt_messages should be identical across rows"


def test_prompt_messages_includes_system_prompt():
    """make_example bakes the env's static system_prompt into prompt_messages
    at index 0, so the system prompt is part of the example's identity."""
    ex = GuessEnv.dataset_preprocess({"target": 1})
    seed = ex["prompt_messages"]
    assert len(seed) == 2, f"expected [system, user], got {len(seed)} messages"
    assert seed[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert seed[1]["role"] == "user"


def test_distinct_system_prompts_produce_distinct_example_ids():
    """Same dataset row + different env system prompts → different ids.

    This is the property that would have been broken if prompt_messages had
    stayed user-only: two envs with the same dataset rows but different
    system instructions would have collapsed into the same example_id.
    """

    class GuessEnvVerbose(GuessEnv):
        system_prompt: str = SYSTEM_PROMPT + " Show your reasoning step by step."

    row = {"target": 42}
    ex_default = GuessEnv.dataset_preprocess(row)
    ex_verbose = GuessEnvVerbose.dataset_preprocess(row)

    assert ex_default["id"] != ex_verbose["id"], (
        "envs with different system_prompts must produce distinct example_ids"
    )


def test_same_target_produces_same_example_id():
    """Idempotency: re-preprocessing the same dataset row → same id."""
    row = {"target": 42}
    ex_a = GuessEnv.dataset_preprocess(row)
    ex_b = GuessEnv.dataset_preprocess(row)
    assert ex_a["id"] == ex_b["id"]


def test_example_id_matches_direct_canonical_hash():
    """The id baked into Example.id matches the hash an external caller
    (e.g. external-eval webhook, or the TS port) would compute from the
    same (prompt_messages, task) pair."""
    row = {"target": 17}
    ex = GuessEnv.dataset_preprocess(row)
    expected = canonical_example_id(ex["prompt_messages"], ex["task"])
    assert ex["id"] == expected


def test_task_kept_in_example_not_kwargs():
    """Option A contract: task must land in Example.task, not in **kwargs.

    A regression to the old behaviour — where per-row data was flattened
    into kwargs that compute_reward had to pull out by name — would mean
    Example.task is None/empty here.
    """
    ex = GuessEnv.dataset_preprocess({"target": 5})
    assert ex.get("task") == {"target": 5}


@pytest.mark.asyncio
async def test_compute_reward_reads_task_not_kwargs():
    """End-to-end at the compute_reward boundary: the reward function pulls
    target out of `task` (not kwargs), and a correct guess scores 1.0."""
    env = GuessEnv()
    messages = [
        {"role": "user", "content": "I have a number in mind. What is it?"},
        {"role": "assistant", "content": "42"},
    ]
    correct = await env.compute_reward(
        rollout_id="r1", messages=messages, task={"target": 42}
    )
    assert correct == {"correctness": 1.0}

    wrong = await env.compute_reward(
        rollout_id="r2", messages=messages, task={"target": 7}
    )
    assert wrong == {"correctness": 0.0}


@pytest.mark.asyncio
async def test_compute_reward_ignores_legacy_ground_truth_in_kwargs():
    """A kwarg named 'ground_truth' must NOT be silently picked up — the
    redesign explicitly moved per-example data into `task`, so plumbing
    bugs that stuff ground_truth into kwargs should produce a wrong score
    (since task.target is None), not a misleading correct one."""
    env = GuessEnv()
    messages = [{"role": "assistant", "content": "42"}]
    # ground_truth as kwarg, task is None — env should NOT use the kwarg.
    rewards = await env.compute_reward(
        rollout_id="r1", messages=messages, task=None, ground_truth=42
    )
    assert rewards == {"correctness": 0.0}, (
        "GuessEnv must read target from task only; if kwargs leaked in, this would be 1.0"
    )
