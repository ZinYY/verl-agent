#!/bin/bash
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
TEACHER_MODEL="$HOME/models/webshop_qwen2.5_1.5b_merged"
OUTPUT_DIR="$HOME/data/verl-agent/webshop_sft"
BATCH_SIZE=64
NUM_BATCHES=10  # Total = BATCH_SIZE * NUM_BATCHES
num_cpus_per_env_worker=0.1

# Generate trajectories in batches (each batch saves to separate dir to avoid overwrite)
for batch_id in $(seq 0 $((NUM_BATCHES - 1))); do
    offset=$((batch_id * BATCH_SIZE))
    echo "=== Batch $batch_id / $NUM_BATCHES, offset=$offset ==="

    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size $BATCH_SIZE \
        --val_data_size $BATCH_SIZE \
        --train_offset $offset \
        --val_offset $offset

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/train.parquet \
        data.train_batch_size=2 \
        data.val_batch_size=$BATCH_SIZE \
        data.max_prompt_length=4096 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$TEACHER_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=2 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.3 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        env.env_name=Webshop \
        env.seed=$batch_id \
        env.max_steps=15 \
        env.rollout.n=1 \
        env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
        trainer.logger=['console'] \
        trainer.project_name='webshop_sft_gen' \
        trainer.experiment_name="generate_batch_${batch_id}" \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        +trainer.save_traj_for_sft=True \
        +trainer.sft_traj_output_dir="${OUTPUT_DIR}/batch_${batch_id}"
done

# Merge all batch parquet files into one
python3 -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
train_dfs, val_dfs = [], []
for batch_dir in sorted(output_dir.glob('batch_*')):
    if (batch_dir / 'train.parquet').exists():
        train_dfs.append(pd.read_parquet(batch_dir / 'train.parquet'))
    if (batch_dir / 'val.parquet').exists():
        val_dfs.append(pd.read_parquet(batch_dir / 'val.parquet'))

if train_dfs:
    pd.concat(train_dfs, ignore_index=True).to_parquet(output_dir / 'train.parquet')
if val_dfs:
    pd.concat(val_dfs, ignore_index=True).to_parquet(output_dir / 'val.parquet')
print(f'Merged {len(train_dfs)} batches: train={sum(len(d) for d in train_dfs)}, val={sum(len(d) for d in val_dfs)}')
"

echo "All trajectories saved to: $OUTPUT_DIR"
