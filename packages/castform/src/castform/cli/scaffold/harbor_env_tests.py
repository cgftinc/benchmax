"""Structural tests for the Harbor seed. Add task-specific verifier tests here."""

import argparse

import pytest
from benchmax.envs.harbor import HarborEnv, ModalCredentials
from main import (
    CustomHarborEnv,
    _constructor_args,
    _parse_verifier_env,
)


def _args(**overrides):
    values = {
        "dataset": "org/task-package",
        "dataset_ref": "latest",
        "modal_token_id": "modal-id",
        "modal_token_secret": "modal-secret",
        "verifier_env": ["OPENAI_API_KEY=judge-key"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_constructor_args_are_explicit_and_build_the_stock_harness() -> None:
    constructor_args = _constructor_args(_args())
    env = CustomHarborEnv(**constructor_args)

    assert isinstance(env, HarborEnv)
    assert constructor_args["dataset_name"] == "org/task-package"
    assert constructor_args["dataset_ref"] == "latest"
    assert isinstance(constructor_args["sandbox_credentials"], ModalCredentials)
    assert constructor_args["verifier_env"] == {"OPENAI_API_KEY": "judge-key"}
    assert env._trial.agent.name == "mini-swe-agent"
    assert env._trial.agent.model_name is None
    assert env._trial.agent.kwargs == {}


def test_stock_harness_has_no_static_model_contract_errors() -> None:
    env = CustomHarborEnv(**_constructor_args(_args()))

    assert env.validation_diagnostics() == ()


def test_verifier_env_accepts_repeated_name_value_arguments() -> None:
    assert _parse_verifier_env(["OPENAI_API_KEY=key", "OPENAI_BASE_URL=https://host/v1"]) == {
        "OPENAI_API_KEY": "key",
        "OPENAI_BASE_URL": "https://host/v1",
    }


@pytest.mark.parametrize(
    "assignments",
    [
        ["OPENAI_API_KEY"],
        ["OPENAI_API_KEY="],
        ["BAD-NAME=value"],
        ["OPENAI_API_KEY=one", "OPENAI_API_KEY=two"],
    ],
)
def test_verifier_env_rejects_ambiguous_values(assignments: list[str]) -> None:
    with pytest.raises(ValueError):
        _parse_verifier_env(assignments)
