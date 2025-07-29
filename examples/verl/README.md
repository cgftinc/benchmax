# benchmax environments with verl

### **🔍 Quickstart: RL Math Agent with verl + `benchmax`**

We RL fine-tune a RL agent using a calculator MCP server using the `verl` trainer and the benchmax `math`environment. 

The full script can be found here: 

To run it:

- Clone and install `benchmax` with our verl fork (temporary until [PR gets merged](https://github.com/volcengine/verl/pull/2792))
    
    ```bash
    pip install benchmax[verl]
    ```
    
- Prepare the dataset
    
    ```bash
    python benchmax/adapters/verl/benchmax_data_process.py \
      --local_dir ~/data/math \
      --dataset_name dawidmt/arithmetic50 \
      --env_path benchmax.envs.math.math_env.MathEnv
    ```
    
- Run training
    
    ```bash
    sh examples/verl/run_qwen2.5-3b_benchmax_math.sh
    ```
    

---

## How do I use an environment with verl?

To use your custom environment with verl —follow these steps:

1. **Prepare Your Environment & Tools**
    - Implement your environment as a Python class that subclasses **`BaseEnv`** (or **`LocalMCPEnv`** for MCP-based tools).
    - Define:
        - The **`system_prompt`** that guides agent behavior.
        - Your tool or tools: expose each as a **`ToolDefinition`** and register in the environment.
        - Your reward function(s): add them to the **`_reward_funcs`** class variable.
2. **Export Data in Multiturn Format**
    - Use the provided **`benchmax_data_process.py`** script to preprocess HuggingFace datasets into the expected multiturn chat format, suitable for sandbox rollouts.
    - Command example:
        
        `python python benchmax/adapters/verl/benchmax_data_process.py \
          --local_dir ... \ 
          --dataset_name ... \
          --env_path <your.env.ClassPath>`
        
        | Flag | Required? | Default | Description |
        | --- | --- | --- | --- |
        | `--local_dir` | yes | – | **Local directory** where the processed `train.parquet` and `test.parquet` files will be written. |
        | `--dataset_name` | yes | – | **Name of the HuggingFace dataset** to load, e.g. `squad`, `wikitext`, or any HF dataset identifier. |
        | `--env_path` | yes | – | **Dotted path** to the benchmax env (e.g. `benchmax.envs.math.math_env.MathEnv`) used for preprocessing |
3. **Configure Tool Integration**
    - Define your tools in a YAML config (see `examples/verl/config/tool_config/benchmax_math_tool_config.yaml`). Each tool specifies the full class path to your environment and any extra config needed (e.g., API keys). Make sure `type` is set to `benchmax.`Example:
        
        `tools:
          - class_name: benchmax.envs.math.MathEnv
            config:
              type: benchmax
              api_keys:
                - <YOUR_API_KEY>`
        
4. **Add or Update Config for Rollouts**
    - Copy and edit an existing multiturn config (see `examples/verl/config/benchmax_multiturn_grpo.yaml`).
5. **Launch Training**
    - Copy and edit the provided shell script (**`examples/verl/run_qwen2.5-3b_benchmax_math.sh`**) or similar, passing data and configuration file paths. Make sure to edit `TOOL_CONFIG` and `BENCHMAX_CLASS_NAME` to point to the tool config you created above and to the benchmax environment class respectively.
        
        ```bash
        
        TOOL_CONFIG="examples/verl/config/tool_config/benchmax_math_tool_config.yaml"
        BENCHMAX_CLASS_NAME="benchmax.envs.math.MathEnv"
        ```
        
    - Command example:
        
        `sh examples/verl/run_qwen2.5-3b_benchmax_math.sh`
        

Your benchmax environment, with its tools and reward functions, is now directly usable with verl’s RL fine-tuning workflow. Plug in your data, config, and custom environment, and run multiturn RL training out of the box!
