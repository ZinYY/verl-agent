# verl-agent WebShop Experiments

本项目基于 [verl-agent](https://github.com/langfengQ/verl-agent) 框架，针对 WebShop 环境实现了多种训练算法的对比实验。

## 项目改动

在 verl-agent 代码仓库的基础上，对以下文件进行了修改：

| 文件 | 改动说明 |
|------|----------|
| `examples/data_preprocess/prepare.py` | 添加数据集起始位置参数 (`train_offset`, `val_offset`)，支持分批生成轨迹数据 |
| `verl/trainer/ppo/ray_trainer.py` | 添加 trajectory 保存功能，支持将成功轨迹保存为 SFT 训练数据 |
| `examples/grpo_trainer/` | 新增多个实验配置文件夹，包含不同算法的训练/评估脚本 |
| `verl/trainer/ppo/ray_trainer.py` | 添加pass@k的evaluation功能 |

## 环境安装

### 1. 创建 Conda 环境

WebShop 需要 Python <= 3.10：

```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

### 2. 安装 WebShop 环境

```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

> **Note**: 如果遇到 gdown 下载问题，可能需要到Google Drive手动下载相关文件。

### 3. 安装 verl-agent

返回项目根目录并安装依赖：

```bash
cd /path/to/verl-agent
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```


## 实现的算法

本项目在 WebShop 环境上实现了以下四种训练方法，以及pass@k的评估：

| 方法 | 说明 | 模型 |
|------|------|------|
| **Base** | 直接评估预训练模型，不进行任何训练 | Qwen2.5-7B-Instruct |
| **SFT (Imitation Learning)** | 使用 teacher model 生成成功轨迹，进行监督微调 | Qwen2.5-1.5B-Instruct |
| **GRPO (Full)** | 全量参数微调的 GRPO 强化学习训练 | Qwen2.5-1.5B-Instruct |
| **GRPO (LoRA)** | 使用 LoRA 的 GRPO 强化学习训练 | Qwen2.5-7B-Instruct |
|**Eval Pass@k** | 评估模型在 WebShop 测试集上 pass@k 的表现, k=1,2,4,64,128 | ALL |

## 运行实验

### 总览
```bash
bash examples/grpo_trainer/start_exp.sh
```

### 方法1: Base 评估 (无训练)

直接评估预训练模型在 WebShop 上的表现：

```bash
bash examples/grpo_trainer/webshop_qwen_7b_base/eval.sh
```

### 方法2: SFT (Imitation Learning)

分三步进行：

```bash
# Step 1: 使用 teacher model 生成轨迹数据
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/generate.sh

# Step 2: SFT 训练
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/train.sh

# Step 3: 评估
bash examples/grpo_trainer/webshop_qwen_1.5b_full_SFT/eval.sh
```

### 方法3: GRPO 全量微调

```bash
# 训练
bash examples/grpo_trainer/webshop_qwen_1.5b_full/train.sh

# 评估
bash examples/grpo_trainer/webshop_qwen_1.5b_full/eval.sh
```

### 方法4: GRPO LoRA 微调

注意调整学习率。

```bash
# 训练
bash examples/grpo_trainer/webshop_qwen_7b_lora/train.sh

# 评估
bash examples/grpo_trainer/webshop_qwen_7b_lora/eval.sh
```

## 实验结果

在 WebShop 测试集 (500 tasks) 上的评估结果：

| 方法 | Success Rate | Task Score | Test Score |
|------|--------------|------------|------------|
| Base (Qwen2.5-7B) | 0.4% | 0.025 | 0.035 |
| SFT (Qwen2.5-1.5B) | 5.0% | 0.174 | 0.270 |
| GRPO Full (Qwen2.5-1.5B) pass@1 | **57.0%** | **0.740** | **5.546** |
| GRPO Full (Qwen2.5-1.5B) pass@2 | **62.8%** | **0.775** | **8.380** |
| GRPO Full (Qwen2.5-1.5B) pass@4 | **62.8%** | **0.775** | **9.800** |
| GRPO Full (Qwen2.5-1.5B) pass@64 | **70.0%** | **0.816** | **10.000** |
| GRPO Full (Qwen2.5-1.5B) pass@128 | **67.6%** | **0.801** | **10.000** |
| GRPO LoRA (Qwen2.5-7B) | 效果很差 | 估计学习率没调对 | 或者LoRA没用 |


## 项目结构

```
examples/grpo_trainer/
├── start_exp.sh                    # 实验总览脚本
├── webshop_qwen_7b_base/           # Base 评估
│   └── eval.sh
├── webshop_qwen_1.5b_full_SFT/     # SFT 训练
│   ├── generate.sh                 # 生成轨迹数据
│   ├── train.sh                    # SFT 训练
│   └── eval.sh                     # 评估
├── webshop_qwen_1.5b_full/         # GRPO 全量微调
│   ├── train.sh
│   └── eval.sh
└── webshop_qwen_7b_lora/           # GRPO LoRA 微调
    ├── train.sh
    └── eval.sh
```

