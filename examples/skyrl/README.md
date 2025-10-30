# Benchmax Environments with SkyRL

This example directory contains example for both Math environment and Excel environment. The guide below is written for the Math environment but the high-level idea is broadly transferrable.

### **🔍 Quickstart: RL Math Agent with SkyRL + `benchmax`**

We can fine-tune an RL agent with SkyRL using a `benchmax` environment — for example, the `math` environment — and a tool-enabled setup for calculator-like reasoning.

Example script:
`examples/benchmax/run_benchmax_math.sh`

---

## 1. Prepare the dataset

Use `benchmax_data_process.py` to convert a HuggingFace dataset into the multiturn chat format expected by SkyRL rollouts.

Example for `arithmetic50`:

```bash
uv run src/benchmax/adapters/skyrl/benchmax_data_process.py \
  --local_dir ~/data/math \
  --dataset_name dawidmt/arithmetic50 \
  --env_path benchmax.envs.math.math_env.MathEnvLocal
```

**Arguments**:

| Flag             | Required? | Description                                                                                   |
| ---------------- | --------- | --------------------------------------------------------------------------------------------- |
| `--local_dir`    | yes       | Output folder for `train.parquet` & `test.parquet`.                                           |
| `--dataset_name` | yes       | HuggingFace dataset name or path.                                                             |
| `--env_path`     | yes       | Dotted path to the `benchmax` environment class (e.g. `benchmax.envs.math.math_env.MathEnvLocal`). |

---

## 2. Launch training — **Focusing on Environment Arguments**

In `run_benchmax_math.sh`, the environment is configured with:

```bash
ENV_CLASS="MathEnv"
...
environment.env_class=$ENV_CLASS \
```

**How it works**:

* **`environment.env_class`**: This must match the **ID** registered in `skyrl_gym.envs.register(...)`.

  * In `benchmax_math.py`, the registration happens here:

    ```python
    register(
        id="MathEnv",
        entry_point=load_benchmax_env_skyrl,
        kwargs={"actor": actor}
    )
    ```
  * The `id` value (`"MathEnv`) is what `ENV_CLASS` should be set to in the shell script and in the preprocessing step.

* **Ray Actor Creation**: Before registration, the script calls:

  ```python
  get_or_create_benchmax_env_actor(MathEnvLocal)
  ```

  This:

  1. Imports your environment class (`from benchmax.envs.math.math_env import MathEnvLocal`).
  2. Starts a persistent Ray actor (`BenchmaxEnvActor`) that wraps the environment.
  3. Names the actor `BaseEnvService`, so `load_benchmax_env_skyrl` can attach to it when Gym launches the env.

* **Putting it together**:
  `ENV_CLASS` → matches Gym registry `id` → which maps to `load_benchmax_env_skyrl` → which connects to the Ray actor created from your imported environment class.

To run:

```bash
bash examples/skyrl/run_benchmax_math.sh
```

---

## 4. Using Your Own Benchmax Environment with SkyRL

To integrate a new `benchmax` environment:

1. **Create or select your `benchmax` environment class**

   * Must subclass `benchmax.BaseEnv` (or `benchmax.mcp.ParallelMcpEnv` for multi-node support) and implement required methods.
   * Add tools, rewards, and `system_prompt` as needed.

2. **Update the SkyRL entrypoint**

   * In your copy of `benchmax_math.py`, change:

     ```python
     from benchmax.envs.math.math_env import MathEnvLocal
     get_or_create_benchmax_env_actor(MathEnvLocal)
     ```

     to import and use your environment class. Change the env to MathEnvSkypilot to run MCP servers on multiple external nodes

3. **Update the registry ID & shell script**

   * Change the `id` in `register(...)` to match your environment name.
   * Update `ENV_CLASS` in your run script to match this ID.

4. **Update dataset preprocessing**

   * Use `--env_path` pointing to your environment class in the preprocessing command.

---

Once set up, SkyRL will:

* Launch your environment in a Ray actor
* Register it in the SkyRL gym
* Fine-tune your model via PPO with configurable multi-turn RL training