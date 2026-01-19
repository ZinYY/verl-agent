# 改动的文件：
# `examples/data_preprocess/prepare.py`: 添加数据集起始位置参数
# `verl/trainer/ppo/ray_trainer.py`: 添加trajectory保存功能
# `examples/grpo_trainer`里的多个文件夹：包含：base (不训练，直接测)，SFT (Imitation Learning)，GRPO (全量微调版本)，GRPO（LoRA微调版本）




###################### Base (不训练，直接测) ########################

# For qwen_7b_base (only eval, no training)
bash examples/grpo_trainer/webshop_qwen_7b_base/eval.sh
# Result, pass@1:
# (TaskRunner pid=1811260) step:0 
# - val/text/test_score:0.035 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.004 
# - val/webshop_task_score (not success_rate):0.025




###################### SFT (Imitation Learning) ########################

# For qwen_1.5b_full_SFT (train with SFT data, Imitation Learning)
# Step 1: 用teacher model生成轨迹数据（保存成功轨迹用于 SFT）
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/generate.sh
# Step 2: SFT 训练小模型
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/train.sh
# Step 3: 评估
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/eval.sh
# Result, pass@1:
# (TaskRunner pid=2997566) step:0 
# - val/text/test_score:0.270 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.050 
# - val/webshop_task_score (not success_rate):0.174



###################### GRPO (全量微调版本) ########################

# For qwen_1.5b_full (train with full model, GRPO)
bash examples/grpo_trainer/webshop_qwen_1.5b_full/train.sh
bash examples/grpo_trainer/webshop_qwen_1.5b_full/eval.sh
# Result, pass@1:
# (TaskRunner pid=1686147) step:0 
# - val/text/test_score:5.546 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.570 
# - val/webshop_task_score (not success_rate):0.740

# Result, pass@2:
# (TaskRunner pid=799422) step:0 
# - val/text/test_score:8.380 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.628 
# - val/webshop_task_score (not success_rate):0.775

# Result, pass@4:
# (TaskRunner pid=930943) step:0 
# - val/text/test_score:9.800 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.628 
# - val/webshop_task_score (not success_rate):0.775

# Result, pass@64:
# (TaskRunner pid=1079076) step:0 
# - val/text/test_score:10.000 
# - val/text/tool_call_count/mean:0.000 
# - val/success_rate:0.700 
# - val/webshop_task_score (not success_rate):0.816

###################### GRPO (LoRA微调版本) ########################

# For qwen_7b_lora (train with LoRA, GRPO)
bash examples/grpo_trainer/webshop_qwen_7b_lora/train.sh
bash examples/grpo_trainer/webshop_qwen_7b_lora/eval.sh
# Result:
结果很差，应该是learning rate没调对，或者lora根本不起效果。