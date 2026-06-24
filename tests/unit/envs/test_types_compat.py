from benchmax.envs.types import PolicyConfig, Trajectory


def test_trajectory_to_sample_dict_preserves_trainer_shape():
    trajectory = Trajectory(
        rollout_id="rollout-1",
        example_id="example-1",
        prompt_messages=[{"role": "user", "content": "hello"}],
        messages=[{"role": "assistant", "content": "world"}],
        task={"answer": "world"},
        prompt_ids=[1, 2],
        completion_ids=[3],
        completion_mask=[1],
        logprobs=[-0.5],
        rewards={"correctness": 1.0},
        metadata={"extra": "kept"},
    )

    assert trajectory.to_sample_dict() == {
        "rollout_id": "rollout-1",
        "example_id": "example-1",
        "prompt_messages": [{"role": "user", "content": "hello"}],
        "messages": [{"role": "assistant", "content": "world"}],
        "task": {"answer": "world"},
        "prompt_ids": [1, 2],
        "prompt_mask": [1, 1],
        "completion_ids": [3],
        "completion_mask": [1],
        "logprobs": [-0.5],
        "rewards": {"correctness": 1.0},
        "truncated": False,
        "extra": "kept",
    }


def test_policy_config_defaults_are_pickle_bundle_compatible():
    config = PolicyConfig()

    assert config.base_url is None
    assert config.model is None
    assert config.api_key is None
    assert config.rollout_func is None
    assert config.generation == {}
