import verifiers as vf

from benchmax.adapters.verifiers.verifiers_adapters import get_verifiers_environment
from benchmax.envs.braintrust.braintrust_env import BraintrustEnv
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.environ.get("API_KEY")

"""
Multi-GPU training (single node, 3 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 poetry run vf-vllm --model willcb/Qwen3-4B

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch examples/verifiers/verifiers_braintrust_example.py
"""

dataset, _ = BraintrustEnv.load_dataset(braintrust_api_key=API_KEY, braintrust_dataset_id="0e07f061-1b7c-4a8f-aedb-6f2ca787ccc4")
braintrust_env = BraintrustEnv(braintrust_api_key=API_KEY, braintrust_project_id="89d983a3-7012-4932-a816-67f025318096")
prompt = braintrust_env.prompts["247e1e74-8c32-41f6-97bc-4b998be34e3b"][1]["content"]

dataset = dataset.map(
    lambda example: braintrust_env.dataset_preprocess(example=example, prompt=prompt)
)
splits = dataset.train_test_split(test_size=0.1, seed=42)

train_ds = splits["train"]

vf_env = get_verifiers_environment(
    braintrust_env,
    max_concurrent=3,
    max_turns=3,
    dataset=train_ds,
)

model_name = "willcb/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "verifiers-braintrust" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size=6
training_args.num_generations=12
training_args.gradient_accumulation_steps=2
training_args.num_iterations=1
training_args.num_train_epochs=2
training_args.max_prompt_length=10000
training_args.max_completion_length=4096
training_args.max_steps=500
training_args.save_steps=100
training_args.report_to = "none"
training_args.log_completions = False

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()


