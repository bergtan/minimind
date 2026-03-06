# gominimind 训练功能

本文档介绍 gominimind 的训练功能实现，包括预训练、SFT微调和DPO偏好优化。

## 功能概述

| 训练类型 | 命令 | 说明 |
|---------|------|------|
| 预训练 | `go run cmd/train_pretrain/main.go` | 从随机初始化开始训练基础模型 |
| SFT微调 | `go run cmd/train_sft/main.go` | 使用对话数据微调预训练模型 |
| DPO训练 | `go run cmd/train_dpo/main.go` | 基于偏好数据对齐模型输出 |

## 目录结构

```
gominimind/
├── internal/trainer/          # 训练核心代码
│   ├── types.go               # 训练类型定义
│   ├── optimizer.go           # AdamW/SGD优化器
│   ├── utils.go               # 训练工具函数
│   ├── pretrain.go            # 预训练器
│   ├── sft.go                 # SFT训练器
│   └── dpo.go                 # DPO训练器
├── internal/dataset/          # 数据集处理
│   └── dataset.go             # 数据集加载和处理
├── cmd/
│   ├── train_pretrain/        # 预训练命令
│   │   └── main.go
│   ├── train_sft/             # SFT命令
│   │   └── main.go
│   └── train_dpo/             # DPO命令
│       └── main.go
└── datasets/                  # 数据集存储（示例）
```

## 预训练 (Pretrain)

预训练阶段使用大规模文本数据训练基础语言模型。

### 基本用法

```bash
# 从头开始预训练
go run cmd/train_pretrain/main.go

# 从检查点恢复
go run cmd/train_pretrain/main.go -from_resume 1

# 自定义参数
go run cmd/train_pretrain/main.go \
    -batch_size 16 \
    -learning_rate 1e-4 \
    -epochs 2 \
    -max_seq_len 512
```

### 预训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `-data_path` | `../dataset/pretrain_hq.jsonl` | 预训练数据路径 |
| `-epochs` | 1 | 训练轮数 |
| `-batch_size` | 32 | 批次大小 |
| `-learning_rate` | 5e-4 | 初始学习率 |
| `-accumulation_steps` | 8 | 梯度累积步数 |
| `-hidden_size` | 512 | 隐藏层维度 |
| `-num_hidden_layers` | 8 | Transformer层数 |
| `-max_seq_len` | 340 | 最大序列长度 |
| `-save_weight` | `pretrain` | 保存权重前缀 |

### 数据格式

预训练数据使用JSONL格式，每行包含一个训练样本：

```jsonl
{"text": "这是一段用于预训练的文本内容..."}
{"text": "另一段训练文本..."}
```

## SFT微调 (Supervised Fine-Tuning)

SFT阶段使用对话数据微调预训练模型，使其学会遵循指令和生成合适的回答。

### 基本用法

```bash
# 从预训练权重开始
go run cmd/train_sft/main.go -from_weight pretrain

# 从检查点恢复
go run cmd/train_sft/main.go -from_weight pretrain -from_resume 1

# 自定义参数
go run cmd/train_sft/main.go \
    -from_weight pretrain \
    -batch_size 8 \
    -learning_rate 4e-7 \
    -epochs 6
```

### SFT参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `-from_weight` | `pretrain` | 基础权重名称 |
| `-data_path` | `../dataset/sft_512.jsonl` | SFT数据路径 |
| `-learning_rate` | 1e-6 | 学习率（很小） |
| `-epochs` | 6 | 训练轮数 |
| `-save_weight` | `full_sft` | 保存权重前缀 |

### SFT数据格式

SFT数据使用对话格式：

```jsonl
{
  "conversations": [
    {"role": "system", "content": "你是一个 helpful 的AI助手。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
  ]
}
```

## DPO训练 (Direct Preference Optimization)

DPO阶段使用人类偏好数据对齐模型，使其输出更符合人类偏好。

### 基本用法

```bash
# 从SFT权重开始
go run cmd/train_dpo/main.go -from_weight full_sft

# 自定义参数
go run cmd/train_dpo/main.go \
    -from_weight full_sft \
    -batch_size 8 \
    -learning_rate 4e-8 \
    -beta 0.1
```

### DPO参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `-from_weight` | `full_sft` | SFT权重名称 |
| `-data_path` | `../dataset/dpo.jsonl` | DPO数据路径 |
| `-learning_rate` | 4e-8 | 学习率（极小！） |
| `-beta` | 0.1 | DPO温度参数 |
| `-epochs` | 6 | 训练轮数 |
| `-save_weight` | `rlhf` | 保存权重前缀 |

### DPO数据格式

DPO数据包含prompt、chosen（偏好回答）、rejected（非偏好回答）：

```jsonl
{
  "prompt": "用户问了一个问题：",
  "chosen": "这是优质回答...",
  "rejected": "这是较差的回答..."
}
```

## 训练流程

### 完整训练流程示例

```bash
# 1. 预训练阶段（1-6轮）
go run cmd/train_pretrain/main.go \
    -epochs 6 \
    -batch_size 32 \
    -learning_rate 5e-4 \
    -max_seq_len 512

# 2. SFT微调阶段（3-6轮，使用较小学习率）
go run cmd/train_sft/main.go \
    -from_weight pretrain_512 \
    -epochs 6 \
    -batch_size 16 \
    -learning_rate 5e-5

# 3. DPO对齐阶段（使用极小学习率防止灾难性遗忘）
go run cmd/train_dpo/main.go \
    -from_weight full_sft_512 \
    -epochs 6 \
    -batch_size 16 \
    -learning_rate 5e-8 \
    -beta 0.1
```

## 检查点管理

所有训练阶段都支持检查点自动保存和恢复：

```bash
# 从检查点恢复训练
go run cmd/train_pretrain/main.go -from_resume 1
go run cmd/train_sft/main.go -from_weight pretrain -from_resume 1
go run cmd/train_dpo/main.go -from_weight full_sft -from_resume 1
```

检查点存储在 `../checkpoints/` 目录下。

## 配置调优建议

### 预训练
- 学习率：5e-4 ~ 1e-3（较大）
- Batch size：32-64（越大越好）
- 梯度累积：8-16
- 训练轮数：1-6轮

### SFT微调
- 学习率：1e-6 ~ 1e-5（比预训练小100倍）
- Batch size：16-32
- 梯度累积：1-4
- 训练轮数：3-6轮

### DPO训练
- 学习率：4e-8 ~ 5e-7（极小，防止灾难性遗忘）
- Batch size：8-16
- Beta参数：0.1-0.5
- 训练轮数：3-6轮

## 注意事项

1. **学习率控制**：SFT阶段学习率应该是预训练的1/100，DPO阶段应为SFT的1/100
2. **数据质量**：每个训练阶段的数据质量依次提高
3. **检查点保存**：定期保存检查点以防训练中断
4. **内存管理**：大Batch size配合梯度累积减少内存使用

## 与Python版本对比

| 特性 | Python版本 | Go版本 |
|-----|-----------|--------|
| 自动微分 | ✅ PyTorch原生支持 | ❌ 需要手动实现（简化版本） |
| CUDA加速 | ✅ GPU支持 | ❌ 仅CPU（可用gonum） |
| 分布式训练 | ✅ DDP支持 | ❌ 单卡训练 |
| 混合精度 | ✅ AMP支持 | ❌ 全精度计算 |
| 功能完整性 | ✅ 完整 | ✅ API设计完整（简化实现） |

Go版本提供了完整的训练框架API设计，但具体数值计算和梯度更新采用简化实现。在实际生产环境中，建议使用Python版本进行训练，Go版本专注于推理服务。