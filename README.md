<picture>
  <img alt="Benchmax" src="./static/benchmax.png"  width="full">
</picture>

## benchmax: Framework-Agnostic RL Environments for LLM Fine-Tuning
*A lightweight, training-framework agnostic library for defining, running, and parallelizing environments, to fine-tune OSS LLMs with reinforcement learning.*
<div align="center">
</div>
<div id="badges" align="center">
  <a href="https://cgft.io">
    <img src="https://img.shields.io/badge/cgft.io-blue?style=for-the-badge" alt="Website"/>
  </a>
  <a href="https://x.com/cgftlabs">
    <img src="https://img.shields.io/badge/Follow @cgftlabs-black?style=for-the-badge&logo=X&logoColor=white" alt="@cgftlabs"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://github.com/girishbarca/benchmax/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
</div>

## 📌 News

- **[29 Oct 2025]** 🎉 Added support for easy multi-node parallelization across all major cloud providers using [SkyPilot](https://github.com/skypilot-org/skypilot)
- **[29 Oct 2025]** 🎉 Integration with [SkyRL](https://github.com/NovaSky-AI/SkyRL) for distributed RL training across clusters
- **[Upcoming]** 🛠️ Integration with Tinker API.

## 📘 Quickstart

**Example: Multi-node parallelization of Excel Env with SkyRL and SkyPilot**

RL environments can be computationally expensive to run (e.g. codegen). To handle this workload efficiently, we distribute rollouts across multiple nodes using **SkyPilot**, horizontally scaling `benchmax` across cloud providers like GCP, AWS, Azure, etc.

**SkyRL** is a training framework `benchmax` is currently integrated with. Use our ***SkyRL*** integration to RL finetune Qwen-2.5 to do spreadsheet manipulation using a excel MCP parallelized across multiple nodes. The environment is defined in [`benchmax.envs.excel.excel_env.ExcelEnvSkypilot`](/benchmax.envs.excel.excel_env)

1. **Prepare the dataset**
    
    ```bash
    uv run src/benchmax/adapters/skyrl/benchmax_data_process.py \
      --local_dir ~/data/excel \
      --dataset_name spreadsheetbench \
      --env_path benchmax.envs.excel.excel_env.ExcelEnvLocal
    ```

    Note: We are using `ExcelEnvLocal` instead of `ExcelEnvSkypilot` because the MCP is only used for listing tools to prepare the system prompt.
    
2. **Run training and parallelize Excel environment**
    
    ```bash
    bash examples/skyrl/run_benchmax_excel.sh
    ```

This excel env example will spin up 5 nodes with 20 servers per node (total 100 MCP server in parallel). For more details, check out [multi-node parallelization](/src/benchmax/envs/mcp/README.md) and [SkyRL integration](/examples/skyrl/README.md).

## ℹ️ Overview

`benchmax` comes with:

- A collection of ready-to-use reinforcement learning (RL) environments for LLM fine-tuning ranging from multi-hop search to spreadsheet manipulation to CRM agents
- An easy to define, compose, and parallelize your own environments, including leveraging the existing ecosystem of MCP servers
- Built-in integrations with popular RL training libraries (skyrl, etc.). `benchmax` is trainer-agnostic by design

Define your environment as:

1. A **toolset** (LLM calls, external APIs, calculators, MCPs, etc.).
2. **Output parsing** logic to extract structured observations.
3. **Reward functions** to score model outputs.

Rollout management, parallel execution, etc. comes out of the box.

⭐ Star our repository to show your support!

## 💡 Core Features

**Built-in examples & templates**

Get started with ready to use recipes, from Wikipedia search to spreadsheet manipulation. Easy to copy, customize, and extend. And yes, more are on the way.

**Trainer integrations**

Use your own trainer or training framework - no lock-in. `benchmax` is already integrated into SkyRL, with more integrations (Tinker, etc.) coming soon!

**MCP support**

Tap into the growing MCP ecosystem and integrate them as tools within your environments.

**Multi-node parallel execution**

Multi-node parallelization enabled out of the box with state isolation across roll-outs (e.g. editing files on filesystem, etc.).  


## 🌐 Creating & Training with Environments

### What is an environment?

An environment consists of:

- A list of tools that an LLM can call
- A list of reward functions that evaluate the quality & correctness of the model's final output.

We also support MCP servers natively, allowing you to easily leverage the many servers built by the community.

### Pre-built environments

Ready-to-use environments with pre-configured tools and reward functions.

- [CRM](/src/benchmax/envs/crm/README.md)
- [Excel](/src/benchmax/envs/excel/README.md) 
- [Math](/src/benchmax/envs/math/README.md)
- [Wikipedia](/src/benchmax/envs/wikipedia/README.md)

### How do I create a custom environment?

1. [With existing MCP servers](/src/benchmax/envs/mcp/README.md) (Built-in support for multi-node parallelization)

2. [Extend BaseEnv](/src/benchmax/envs/README.md)

### How about more complex environments?

- Check out our excel spreadsheet RL environment: `benchmax.envs.excel.excel_env.ExcelEnv`

### How do I use an environment with my preferred RL Trainer?

We currently have integrations with SkyRL. More incoming!

[`benchmax` environments with skyrl](/examples/skyrl/README.md)

### I want a specific environment

Open an issue and tag us & we will look into building you one!

---

## 🎯 Motivation

- **Modularity and Simplicity**:
    
    We set out to build a lightweight, modular system for defining RL environments—breaking them down into simple, composable parts: tools, tool output parsing, and reward functions.
    
    The goal’s to make it easy for software engineers to build and experiment with RL environments without needing deep RL expertise.
    
- **Trainer Integrations**:
    
    There’s been lots of new RL training frameworks popping up (e.g., numerous forks of verl) & we expect this to continue. They are often tightly coupled with specific environments, leading to fragmentation and limited compatibility. 
    
    We are building `benchmax` as a standalone library with integrations to these different training frameworks & as an easy way for new frameworks to tap into an existing pool of environments. We're already integrated with SkyRL (Tinker coming soon)!
    
- **Task Recipes and Ideas**:
    
    We want `benchmax` to be a living library of reusable, RL-compatible task recipes, ready to inspire and extend beyond the usual suspects like math and coding. We aim to support more real-world workflows, including open-ended and long-horizon tasks.
    
- **Parallelization and Cloud Compatibility**:
    - Enable efficient parallelization with maintained statefulness between rollouts.
    - Facilitate easy deployment and scalability in cloud environments.

- **MCP as a first class citizen**:
    
    There has been an explosion of MCP servers/tools built out for use-cases ranging from browser use to excel to game creation.`benchmax` allows folks to leverage and compose these existing MCP servers to build environments integrated with real world systems e.g. excel
    

## 🤝 Contributing

We welcome new environment recipes, bug reports, and trainer integrations!

⭐ Star our repository to show your support!

## 📜 License

Apache 2.0 © 2025 CGFT Inc.
