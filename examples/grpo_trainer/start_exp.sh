# before experiment, set the environment variables
export WANDB_API_KEY=48de847750dd8d971bd7cce0720e512a8e5cc067
export https_proxy=http://ubuntu-lab.zinyy.tech:20171 http_proxy=http://ubuntu-lab.zinyy.tech:20171 all_proxy=http://ubuntu-lab.zinyy.tech:20171
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

# 改动的文件：
# `examples/data_preprocess/prepare.py`: 添加数据集起始位置参数
# `verl/trainer/ppo/ray_trainer.py`: 添加trajectory保存功能


# For qwen_1.5b_full_SFT (train with SFT data, Imitation Learning)
# Step 1: 用大模型生成轨迹数据（保存成功轨迹用于 SFT）
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/generate.sh
# Step 2: SFT 训练小模型
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/train.sh
# Step 3: 评估
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/eval.sh
# Result:



# For qwen_1.5b_full (train with full model, GRPO)
bash examples/grpo_trainer/webshop_qwen_1.5b_full/train.sh
bash examples/grpo_trainer/webshop_qwen_1.5b_full/eval.sh
# Result:
# (TaskRunner pid=1686147) step:0 
# - val/text/test_score:5.546 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.570 
# - val/webshop_task_score (not success_rate):0.740


# For qwen_7b_lora (train with LoRA, GRPO)
bash examples/grpo_trainer/webshop_qwen_7b_lora/train.sh
bash examples/grpo_trainer/webshop_qwen_7b_lora/eval.sh
# Result:


# For qwen_7b_base (only eval, no training)
bash examples/grpo_trainer/webshop_qwen_7b_base/eval.sh
# Result:
# (TaskRunner pid=1811260) step:0 
# - val/text/test_score:0.035 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.004 
# - val/webshop_task_score (not success_rate):0.025
