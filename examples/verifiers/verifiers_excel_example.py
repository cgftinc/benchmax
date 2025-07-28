import verifiers as vf

from benchmax.adapters.verifiers.verifiers_adapters import get_verifiers_environment
from benchmax.envs.excel.excel_env import ExcelEnv


"""
Multi-GPU training (single node, 3 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 poetry run vf-vllm --model willcb/Qwen3-4B

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch benchmax/adapters/verifiers/examples/verifiers_excel_example.py
"""

dataset, dataset_path = ExcelEnv.load_dataset()
mcp_benchmax_env = ExcelEnv(dataset_path=dataset_path, pool_size=3)
dataset = dataset.map(
    lambda example: mcp_benchmax_env.dataset_preprocess(example),
)
splits = dataset.train_test_split(test_size=0.1, seed=42)

train_ds = splits["train"]

vf_env = get_verifiers_environment(
    mcp_benchmax_env,
    max_concurrent=3,
    max_turns=3,
    dataset=train_ds,
)

model_name = "willcb/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "verifiers-excel" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size=2
training_args.num_generations=12
training_args.gradient_accumulation_steps=2
training_args.num_iterations=1
training_args.num_train_epochs=5
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
