set -x

# Colocated GRPO training+generation for Qwen2.5-3B-Instruct on a sample benchmax excel environment.
# uv run benchmax/adapters/skyrl/benchmax_data_process.py --local_dir ~/data/excel --dataset_name spreadsheetbench --env_path benchmax.envs.excel.excel_env.ExcelEnvLocal
# bash examples/skyrl/run_benchmax_excel.sh

DATA_DIR="$HOME/data/excel"
NUM_GPUS=2
ENV_CLASS="ExcelEnv"

uv run --isolated --group skyrl --group excel -m examples.skyrl.benchmax_excel \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=20 \
  trainer.policy_mini_batch_size=20 \
  trainer.critic_mini_batch_size=20 \
  trainer.micro_forward_batch_size_per_gpu=10 \
  trainer.micro_train_batch_size_per_gpu=10 \
  trainer.eval_batch_size=20 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=3000 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=$ENV_CLASS \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="benchmax_excel" \
  trainer.run_name="benchmax_excel" \
  $@