import threading
from datasets import Dataset

from adapters.verifiers.verifiers_adapters import DEFAULT_ROLLOUT_ID, get_verifiers_environment, _CURRENT_ROLLOUT_ID, verifiers_dataset_mapper
from envs.base_env import BaseEnv

class DummyEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.reward_funcs = []
        self.captured_rollout_ids = []

    def list_tools(self):
        return []

    def run_tool(self, rollout_id: str, tool_name: str, **tool_args):
        # record whatever rollout_id we saw
        self.captured_rollout_ids.append(rollout_id)
        return f"tool-{tool_name}-ok"

    def init_rollout(self, rollout_id: str, **rollout_args):
        # no-op
        pass

    def cleanup_rollout(self, rollout_id: str):
        # no-op
        pass

    def get_rollout_workspace(self, rollout_id: str):
        # we don’t actually use workspace here
        return None


class FakeClient:
    """A stand‐in for OpenAI client; unused in these tests."""
    pass


def test_rollout_sets_and_resets_contextvar():
    dummy = DummyEnv()
    dummy_ds = Dataset.from_dict({
        "prompt": ["hello?"] * 5,
        "ground_truth": ["world!"] * 5,
    })
    env = get_verifiers_environment(dummy, dataset=verifiers_dataset_mapper(dummy_ds))
    env.get_model_response = lambda *args, **kwargs: "bye"
    client = FakeClient()

    # before anything, default is None
    assert _CURRENT_ROLLOUT_ID.get() is None

    # do one rollout; this should set _CURRENT_ROLLOUT_ID inside,
    # then reset it on exit
    env.rollout(
        client=client, model="model", # type: ignore
        prompt=[{"prompt": "hi", "role": "user"}],
        answer="hi"
    )

    # after rollout, it must be back to initial state
    assert _CURRENT_ROLLOUT_ID.get() is None

def test_concurrent_rollout_ids_are_isolated():
    dummy = DummyEnv()
    dummy_ds = Dataset.from_dict({
        "prompt": ["hello?"] * 5,
        "ground_truth": ["world!"] * 5,
    })
    env = get_verifiers_environment(dummy, dataset=verifiers_dataset_mapper(dummy_ds))
    env.get_model_response = lambda *args, **kwargs: '<tool_call>{"name": "get_weather", "arguments": {"location": "New York"}}</tool_call>'
    client = FakeClient()

    seen_ids = set()

    def worker():
        env.rollout(
            client=client, # type: ignore
            model="model", prompt=[{"prompt": "hi", "role": "user"}],
            answer="hi"
        )
        # each rollout pushes exactly one id into captured_rollout_ids
        # we pick the last one
        seen_ids.add(dummy.captured_rollout_ids[-1])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Expect 4 distinct rollout IDs
    assert len(seen_ids) == 4
    # And ensure none of them collides with the default
    assert None not in seen_ids

def test_parallel_rollouts_multiple_tool_calls_use_thread_local_ids():
    dummy = DummyEnv()
    dummy_ds = Dataset.from_dict({
        "prompt": ["hey"] * 2,
        "ground_truth": ["yo"] * 2,
    })
    env = get_verifiers_environment(dummy, dataset=verifiers_dataset_mapper(dummy_ds))

    # Track how many times each rollout has been called
    call_counts: dict[str, int] = {}

    def fake_get_model_response(*args, **kwargs):
        rid = _CURRENT_ROLLOUT_ID.get()
        assert rid is not None
        cnt = call_counts.get(rid, 0)
        if cnt == 0:
            call_counts[rid] = 1
            return '<tool_call>{"name": "first_tool", "arguments": {"x": 1}}</tool_call>'
        elif cnt == 1:
            call_counts[rid] = 2
            return '<tool_call>{"name": "second_tool", "arguments": {"y": 2}}</tool_call>'
        else:
            # no more tool calls → rollout will terminate
            return ""

    env.get_model_response = fake_get_model_response
    client = FakeClient()

    def worker():
        env.rollout(
            client=client, # type: ignore
            model="model",
            prompt=[{"prompt": "hi", "role": "user"}],
            answer="ok",
        )

    # Launch 3 parallel rollouts
    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # We should have 3 rollouts × 2 calls each = 6 tool calls recorded
    assert len(dummy.captured_rollout_ids) == 6

    from collections import Counter
    counts = Counter(dummy.captured_rollout_ids)

    # Exactly 3 distinct rollout IDs, each appearing twice
    assert len(counts) == 3
    for rollout_id, count in counts.items():
        assert count == 2
        # And none of them should be the default fallback
        assert rollout_id != DEFAULT_ROLLOUT_ID
