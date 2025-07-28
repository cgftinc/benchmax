# benchmax environments with verifiers

### **🔍 Quickstart: RL Math Agent with `verifiers` + `benchmax`**

We RL fine-tune a calculator agent using the `verifiers` trainer and the benchmax `math`environment. 

The full script can be found here: 

To run it:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-4B

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/verifiers/verifiers_math_example.py
```

If you want to try something more complex, you can RL fine-tune a spreadsheet agent using the `excel` agent.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-4B

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/verifiers/verifiers_excel_example.py
```

---

### Integration

`benchmax` integrates directly with the [`verifiers`](https://github.com/willcb/verifiers) RL trainer, through the `adapters.verifiers.verifiers_adapter.get_verifiers_environment` function that directly converts any `benchmax` environment into a verifiers compatible sandbox.

### Example Integration

```python
from envs.wikipedia.wiki_env import WikipediaEnv
from adapters.verifiers import get_verifiers_environment

wiki_env = WikipediaEnv() # Create a benchmax sandbox
vf_env = get_verifiers_environment( # Get a verifiers compatible wrapper
    wiki_env,
    max_concurrent=3,        # Number of concurrent rollouts
    max_turns=10,            # Max dialog turns per episode
    dataset=train_ds,        # Your dataset
)
```

**Load your Dataset**

Use `load_dataset` to load the training dataset.

```python
wiki_benchmax_env = WikipediaEnv(api_keys=API_KEYS)
dataset, _ = WikipediaEnv.load_dataset("chiayewken/bamboogle", split="test")
dataset = dataset.map(
    lambda example: wiki_benchmax_env.dataset_preprocess(example),
)
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
```

**Set Up Your Environment**

Instantiate an environment (e.g., `WikipediaEnv`), then wrap it using `get_verifiers_environment` to make it compatible with the Verifiers training loop:

```python
from envs.wikipedia.wiki_env import WikipediaEnv
from adapters.verifiers import get_verifiers_environment

wiki_env = WikipediaEnv()
vf_env = get_verifiers_environment(
    wiki_env,
    max_concurrent=3,        # Number of concurrent rollouts
    max_turns=10,            # Max dialog turns per episode
    dataset=train_ds,        # Your dataset
)

```

**Initialize Your Model & Training Args**

Load a model and tokenizer using the helper, then configure training arguments as you like:

```python
import verifiers as vf

model_name = "willcb/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "bamboogle-wiki-grpo" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 2
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 2
training_args.num_iterations = 1
training_args.num_train_epochs = 5
training_args.max_prompt_length = 1024
training_args.max_completion_length = 4096
training_args.max_steps = 500
training_args.save_steps = 100
training_args.report_to = "none"

```

**Create Trainer and Start Training**

Pass your model, tokenizer, environment, and training arguments into the `GRPOTrainer`, then call `.train()`:

```python
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
```

**Running Training**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-4B

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch adapters/verifiers/verifiers_example_script.py
```

Your benchmax environment, with its tools and reward functions, is now directly usable with verifier’s RL fine-tuning workflow. Plug in your data, config, and custom environment, and run multiturn RL training out of the box!
