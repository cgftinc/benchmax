from pathlib import Path

from benchmax.bundle import dump_bundle, load_bundle
from benchmax.envs.harbor import BundledHarborAgent, ModalCredentials
from harbor import EnvironmentType

from aime_agent import MINI_SWE_AGENT_VERSION
from main import AimeMiniSweHarborEnv


def test_aime_constructor_uses_latest_dataset_and_bundled_agent() -> None:
    credentials = ModalCredentials("modal-id", "modal-secret")

    env = AimeMiniSweHarborEnv(sandbox_credentials=credentials)

    assert env._dataset.name == "aime/aime"
    assert env._dataset.ref == "latest"
    assert env._eval_ratio == 0.1
    assert env._sandbox_credentials is credentials
    assert env.reward_keys == ("reward", "partial_credit")
    trial = env._trial
    assert isinstance(trial.agent, BundledHarborAgent)
    assert trial.agent.config.import_path == "aime_agent:UpstreamMiniSweAgent"
    assert trial.agent.config.kwargs == {"version": MINI_SWE_AGENT_VERSION}
    assert trial.environment.type == EnvironmentType.MODAL
    assert trial.trials_dir == Path("/tmp/castform-aime-harbor-trials")


def test_aime_agent_timeout_flows_into_trial_config() -> None:
    env = AimeMiniSweHarborEnv(
        sandbox_credentials=ModalCredentials("modal-id", "modal-secret"),
        max_agent_timeout_secs=120.0,
    )

    assert env._trial.agent.config.max_timeout_sec == 120.0


def test_aime_bundles_carry_the_fixed_modal_credentials() -> None:
    """Sandbox credentials are fixed keys that ride in bundles."""

    bundle = dump_bundle(
        AimeMiniSweHarborEnv,
        constructor_args={
            "sandbox_credentials": ModalCredentials("modal-id", "modal-secret"),
        },
        pip_dependencies=["harbor[modal]>=0.18.0,<0.19"],
    )
    _, constructor_args = load_bundle(bundle, instantiate=False)
    credentials = constructor_args["sandbox_credentials"]
    assert credentials.token_id == "modal-id"
    assert credentials.token_secret == "modal-secret"
