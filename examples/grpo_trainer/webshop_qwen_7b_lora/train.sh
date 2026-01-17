#!/bin/bash
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Training configuration
train_data_size=8
val_data_size=32    # can set a larger value for more accurate evaluation
group_size=8
num_cpus_per_env_worker=0.1

# Checkpoint and model paths
EXP_NAME="grpo_qwen2.5_7b_lora"
CKPT_DIR="checkpoints/verl_agent_webshop/${EXP_NAME}"
MERGED_MODEL_DIR="$HOME/models/webshop_qwen2.5_7b_lora_merged"

# Prepare data
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=True

# Find latest checkpoint and merge LoRA adapter with base model
LATEST_CKPT=$(ls -td ${CKPT_DIR}/global_step_*/actor/lora_adapter 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Merging LoRA adapter from: $LATEST_CKPT"
    python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

base_model = 'Qwen/Qwen2.5-7B-Instruct'
adapter_path = '$LATEST_CKPT'
output_path = '$MERGED_MODEL_DIR'

print(f'Loading base model: {base_model}')
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f'Loading LoRA adapter: {adapter_path}')
model = PeftModel.from_pretrained(model, adapter_path)

print('Merging and unloading...')
model = model.merge_and_unload()

print(f'Saving merged model to: {output_path}')
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print('Done!')
"
    echo "Merged model saved to: $MERGED_MODEL_DIR"
else
    echo "No checkpoint found in $CKPT_DIR"
fi
