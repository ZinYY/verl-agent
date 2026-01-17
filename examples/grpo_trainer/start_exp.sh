export WANDB_API_KEY=48de847750dd8d971bd7cce0720e512a8e5cc067


# Train and save model
bash examples/grpo_trainer/webshop_qwen_7b_lora/train.sh
bash examples/grpo_trainer/webshop_qwen_1.5b_full/train.sh

# Evaluate trained models
bash examples/grpo_trainer/webshop_qwen_7b_lora/eval.sh
bash examples/grpo_trainer/webshop_qwen_1.5b_full/eval.sh

# Evaluate base model (no training)
bash examples/grpo_trainer/webshop_qwen_7b_base/eval.sh