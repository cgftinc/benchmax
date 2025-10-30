import hydra
import ray
from ray.actor import ActorProxy
from omegaconf import DictConfig
import skyrl_gym
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.config.utils import CONFIG_DIR
from skyrl_gym.envs import register

from benchmax.adapters.skyrl.skyrl_adapter import (
    cleanup_actor,
    get_or_create_benchmax_env_actor,
    load_benchmax_env_skyrl,
)
from benchmax.adapters.benchmax_wrapper import BenchmaxEnv

BENCHMAX_ACTOR_NAME = "BenchmaxEnvService"


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    actor = None
    try:
        # UNCOMMENT the following to run ExcelEnv with skypilot (comment the other)
        from benchmax.envs.excel.excel_env import ExcelEnvSkypilot
        import sky
        
        actor = get_or_create_benchmax_env_actor(
            ExcelEnvSkypilot,
            env_kwargs={
                "cloud": sky.Azure(),
                "num_nodes": 5,
                "servers_per_node": 20,
            },  # samples / prompt * batch size = 100 = 20 * 5
            actor_name=BENCHMAX_ACTOR_NAME,
        )

        # UNCOMMENT the following to run ExcelEnv locally (comment the other)
        # from benchmax.envs.excel.excel_env import ExcelEnvLocal

        # actor = get_or_create_benchmax_env_actor(
        #     ExcelEnvLocal,
        #     env_kwargs={
        #         "num_local_servers": 100
        #     },  # samples / prompt * batch size = 100
        #     actor_name=BENCHMAX_ACTOR_NAME,
        # )

        register(
            id="ExcelEnv",
            entry_point=load_benchmax_env_skyrl,
            kwargs={"actor": actor},
        )
        skyrl_gym.pprint_registry()

        exp = BasePPOExp(cfg)
        exp.run()

    finally:
        cleanup_actor(actor)


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="ppo_base_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    try:
        validate_cfg(cfg)
        initialize_ray(cfg)
        ray.get(skyrl_entrypoint.remote(cfg))
    finally:
        try:
            benchmax_actor: ActorProxy[BenchmaxEnv] = ray.get_actor(BENCHMAX_ACTOR_NAME)
            cleanup_actor(benchmax_actor)
        except Exception:
            pass
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
