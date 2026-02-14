# MyTransformer - Graph+LLM+RL 完整训练指南

> 🎯 **适合初学者的完整教程**：从零开始训练一个结合图神经网络(GNN)和大语言模型(LLM)的智能体系统

---

## 📚 目录

1. [项目介绍](#项目介绍)
2. [核心特性](#核心特性)
3. [项目结构](#项目结构)
4. [环境配置](#环境配置)
5. [数据准备](#数据准备)
6. [四阶段训练流程](#四阶段训练流程)
7. [关键技术详解](#关键技术详解)
8. [运行示例](#运行示例)
9. [常见问题](#常见问题)
10. [参数说明](#参数说明)

---

## 🎯 项目介绍

本项目实现了一个**完整的Graph+LLM+RL训练流水线**，通过4个阶段逐步构建智能体：

```
阶段1: LLaMA预训练    →   阶段2: Graph+LLM联合训练
         ↓                        ↓
    文本理解能力           图结构+文本融合能力
         ↓                        ↓
阶段3: LoRA微调       →   阶段4: PPO强化学习
         ↓                        ↓
    高效参数适配              决策优化能力
```

### 应用场景
- **知识图谱问答**：结合图结构理解实体关系
- **路径规划**：在图上进行智能导航
- **推荐系统**：基于图的个性化推荐
- **智能体决策**：强化学习优化决策策略

---

## ✨ 核心特性

### 🚀 企业级训练框架
- ✅ **分布式训练(DDP)**：支持多GPU并行训练
- ✅ **混合精度训练**：BFloat16/Float16自动混合精度
- ✅ **断点续训**：自动保存/加载checkpoint
- ✅ **实验追踪**：集成Weights & Biases (wandb)
- ✅ **梯度累积**：小显存训练大模型

### 🧠 先进模型技术
- ✅ **Flash Attention**：基于SDPA的高效注意力机制
- ✅ **GQA (Grouped Query Attention)**：减少KV Cache显存占用
- ✅ **NTK-RoPE**：支持更长上下文的位置编码
- ✅ **Gradient Checkpointing**：显存优化技术
- ✅ **LoRA微调**：参数高效微调技术

### 📊 完整训练流程
- ✅ **4阶段Pipeline**：从预训练到强化学习
- ✅ **MiniMind架构对齐**：参考业界最佳实践
- ✅ **模块化设计**：易于扩展和定制

---

## 📁 项目结构

```
mytransformer/
│
├── model/                          # 模型定义
│   ├── __init__.py                # 统一导出接口
│   ├── llama_config.py            # 配置类（支持Graph/LoRA）
│   ├── llama_model.py             # LlamaForCausalLM + GraphLlamaForCausalLM
│   ├── graph_encoder.py           # GCN/GAT图编码器
│   ├── fusion_layer.py            # Graph-Text融合层
│   ├── lora.py                    # LoRA实现
│   ├── graph_policy.py            # RL策略网络
│   ├── tokenizer.py               # Tokenizer（tiktoken/HuggingFace）
│   └── rope_extended.py           # NTK-RoPE扩展
│
├── datasets/                       # 数据加载
│   ├── __init__.py                # 数据集导出
│   ├── pretrain_dataset.py        # 预训练数据集（纯文本）
│   ├── graph_sft_dataset.py       # Graph+Text数据集
│   └── data_utils.py              # 数据处理工具
│
├── trainer/                        # 训练脚本
│   ├── trainer_utils.py           # 🔧 核心工具库（DDP/Checkpoint/LoRA）
│   ├── train_llama_pretrain.py    # 📝 阶段1: LLaMA预训练
│   ├── train_graph_sft.py         # 🌐 阶段2: Graph+LLM联合SFT
│   ├── train_lora_sft.py          # 🎯 阶段3: LoRA微调
│   └── train_graph_ppo.py         # 🤖 阶段4: PPO强化学习
│
├── data/                           # 数据目录（需自行准备）
│   ├── pretrain/                  # 预训练文本数据
│   ├── graph_sft/                 # Graph+Text SFT数据
│   └── graph_rl/                  # 强化学习环境数据
│
├── checkpoints/                    # 训练checkpoint（自动生成）
│   ├── llama_pretrain_resume.pt
│   ├── graph_llama_sft_resume.pt
│   └── lora_adapter_resume.pt
│
├── out/                            # 最终模型权重（自动生成）
│   ├── llama_pretrain.pt
│   ├── graph_llama_sft.pt
│   ├── lora_adapter.pt
│   └── graph_policy_rl.pt
│
└── README.md                       # 本文档
```

---

## 🛠️ 环境配置

### 系统要求
- **Python**: 3.9+ 
- **CUDA**: 11.8+ (推荐12.1)
- **GPU**: 至少8GB显存（推荐24GB+用于多GPU训练）

### 安装依赖

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd mytransformer

# 2. 创建虚拟环境
conda create -n mytransformer python=3.10
conda activate mytransformer

# 3. 安装PyTorch (根据你的CUDA版本)
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install tiktoken                # Tokenizer
pip install torch-geometric          # 图神经网络库
pip install wandb                    # 实验追踪（可选）
pip install numpy tqdm              # 基础库
```

### 验证安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

---

## 📊 数据准备

### 阶段1: 预训练数据

**数据格式**: 纯文本文件（`.txt`）

```
data/pretrain/
├── train_0.txt
├── train_1.txt
└── train_N.txt
```

**示例内容**:
```text
这是第一段训练文本。包含多个句子。
这是第二段训练文本。可以是任意领域的文本。
```

**处理逻辑**: 
- `PretrainDataset` 会自动读取所有`.txt`文件
- 按`max_seq_len`分块（默认512 tokens）
- 自动添加`<bos>`和`<eos>`标记

---

### 阶段2: Graph+Text SFT数据

**数据格式**: JSON Lines (`.jsonl`)

```
data/graph_sft/
├── train.jsonl
└── eval.jsonl
```

**示例内容**:
```json
{
  "instruction": "根据图结构回答：从节点A到节点C的最短路径是什么？",
  "input": "",
  "output": "最短路径是: A -> B -> C",
  "graph": {
    "nodes": [
      {"id": 0, "name": "A", "features": [1.0, 0.5, 0.3]},
      {"id": 1, "name": "B", "features": [0.8, 0.6, 0.2]},
      {"id": 2, "name": "C", "features": [0.7, 0.4, 0.5]}
    ],
    "edges": [
      [0, 1],  // A -> B
      [1, 2],  // B -> C
      [0, 2]   // A -> C
    ]
  }
}
```

**字段说明**:
- `instruction`: 指令/问题
- `input`: 额外输入（可为空）
- `output`: 期望回答
- `graph.nodes`: 节点列表（id + features）
- `graph.edges`: 边列表（邻接关系）

---

### 阶段3: LoRA微调数据

**复用阶段2数据**，但通常使用更小、更专注的数据集：

```
data/graph_sft/
└── lora_train.jsonl  # 小规模精调数据
```

---

### 阶段4: 强化学习数据

**由代码动态生成**，无需额外准备。参见 `model/graph_mission.py` 中的任务定义。

---

## 🚀 四阶段训练流程

### 📝 阶段1: LLaMA预训练

**目标**: 训练基础语言模型，学习文本理解能力

#### 单卡训练
```bash
cd trainer

python train_llama_pretrain.py \
  --data_path ../data/pretrain/ \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 3e-4 \
  --max_seq_len 512 \
  --save_weight llama_pretrain \
  --use_gradient_checkpointing 1
```

#### 多卡训练 (DDP)
```bash
# 4卡训练
torchrun --nproc_per_node=4 train_llama_pretrain.py \
  --data_path ../data/pretrain/ \
  --batch_size 16 \
  --epochs 10 \
  --learning_rate 3e-4
```

#### 断点续训
```bash
python train_llama_pretrain.py \
  --from_resume 1 \
  --save_weight llama_pretrain
```

**输出**:
- `checkpoints/llama_pretrain_resume.pt` (断点文件)
- `out/llama_pretrain.pt` (最终权重)

---

### 🌐 阶段2: Graph+LLM联合SFT

**目标**: 在预训练模型基础上添加图编码器，训练图文融合能力

```bash
python train_graph_sft.py \
  --from_pretrain llama_pretrain.pt \
  --data_path ../data/graph_sft/train.jsonl \
  --batch_size 16 \
  --epochs 5 \
  --learning_rate 1e-4 \
  --graph_encoder_type gat \
  --graph_num_layers 3 \
  --save_weight graph_llama_sft
```

**关键参数**:
- `--from_pretrain`: 加载阶段1的预训练权重
- `--graph_encoder_type`: 图编码器类型 (`gcn`或`gat`)
- `--graph_num_layers`: GNN层数
- `--graph_node_dim`: 图节点特征维度

**输出**:
- `out/graph_llama_sft.pt` (包含LLM + Graph Encoder权重)

---

### 🎯 阶段3: LoRA微调

**目标**: 高效微调模型，适配特定任务（仅训练少量参数）

```bash
python train_lora_sft.py \
  --from_pretrain graph_llama_sft.pt \
  --data_path ../data/graph_sft/lora_train.jsonl \
  --batch_size 8 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --save_weight lora_adapter
```

**LoRA参数说明**:
- `--lora_rank`: LoRA秩（越小参数越少，推荐4/8/16）
- `--lora_alpha`: 缩放因子（通常设为`2 * rank`）
- `--lora_dropout`: Dropout率（默认0.05）

**训练参数量对比**:
```
原始模型参数: ~100M
LoRA参数: ~0.5M (仅0.5%!)
训练速度提升: 3-5倍
显存占用: 降低50%+
```

**输出**:
- `out/lora_adapter.pt` (仅LoRA参数，几MB大小)

---

### 🤖 阶段4: PPO强化学习

**目标**: 通过强化学习优化决策策略

```bash
python train_graph_ppo.py \
  --from_pretrain graph_llama_sft.pt \
  --num_episodes 1000 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --gamma 0.99 \
  --save_weight graph_policy_rl
```

**强化学习参数**:
- `--gamma`: 折扣因子（未来奖励的权重）
- `--clip_epsilon`: PPO裁剪系数（默认0.2）
- `--value_coef`: 价值损失权重
- `--entropy_coef`: 熵正则化权重

**输出**:
- `out/graph_policy_rl.pt` (强化学习策略权重)

---

## 🔬 关键技术详解

### 1. Flash Attention (SDPA)

**作用**: 加速注意力计算，降低显存占用

**实现**: 使用PyTorch原生`scaled_dot_product_attention`
```python
# model/llama_model.py Line 299
attn_output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=True  # 自动应用因果掩码
)
```

**性能提升**:
- 速度提升: 2-4倍
- 显存节省: 30-50%

---

### 2. GQA (Grouped Query Attention)

**作用**: 减少KV Cache显存占用，支持更长上下文

**配置**:
```python
# 标准MHA: num_attention_heads = num_key_value_heads = 8
# GQA: num_attention_heads = 8, num_key_value_heads = 2
lm_config = LlamaConfig(
    num_attention_heads=8,
    num_key_value_heads=2  # ← 4个Query头共享1个KV头
)
```

**显存节省**:
```
MHA KV Cache: 8头 × 512维 × 2048长度 = 8MB
GQA KV Cache: 2头 × 512维 × 2048长度 = 2MB (节省75%!)
```

---

### 3. NTK-RoPE (位置编码扩展)

**作用**: 支持超长上下文（>2048 tokens）

**配置**:
```python
lm_config = LlamaConfig(
    max_position_embeddings=2048,  # 训练长度
    rope_scaling={
        "type": "ntk",
        "factor": 4.0  # 推理时可扩展到8192 tokens
    }
)
```

---

### 4. Gradient Checkpointing

**作用**: 牺牲少量速度换取显存优化

**启用方式**:
```python
model.enable_gradient_checkpointing()  # 自动重计算中间激活
```

**效果**:
- 显存节省: 40-60%
- 速度下降: 20-30%

---

### 5. 混合精度训练

**BFloat16 vs Float16**:

| 类型 | 精度 | 稳定性 | 适用场景 |
|------|------|--------|----------|
| BFloat16 | 较低 | 高 | 推荐（无需loss scaling） |
| Float16 | 高 | 中 | 需要GradScaler |

**使用**:
```python
# 训练脚本中
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

with torch.cuda.amp.autocast(dtype=dtype):
    loss = model(input_ids, labels=labels)
```

---

### 6. LoRA (Low-Rank Adaptation)

**原理**: 在原始权重旁边添加低秩矩阵

```
原始: W ∈ R^(d×d)
LoRA: W + (B @ A), 其中 A ∈ R^(d×r), B ∈ R^(r×d), r << d
```

**应用位置**:
```python
# trainer_utils.py apply_lora函数
应用到: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

**参数量**:
```
rank=8, hidden_size=512
每个Linear层: 512×8 + 8×512 = 8K参数
总共7个层: ~56K参数 (原始模型 ~100M参数)
```

---

## 💡 运行示例

### 完整训练Pipeline

```bash
cd trainer

# === 阶段1: 预训练 (耗时最长) ===
echo "开始预训练..."
python train_llama_pretrain.py \
  --data_path ../data/pretrain/ \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 3e-4 \
  --save_weight llama_pretrain

# === 阶段2: Graph+LLM SFT ===
echo "开始Graph+LLM联合训练..."
python train_graph_sft.py \
  --from_pretrain llama_pretrain.pt \
  --data_path ../data/graph_sft/train.jsonl \
  --batch_size 16 \
  --epochs 5 \
  --learning_rate 1e-4 \
  --save_weight graph_llama_sft

# === 阶段3: LoRA微调 ===
echo "开始LoRA微调..."
python train_lora_sft.py \
  --from_pretrain graph_llama_sft.pt \
  --data_path ../data/graph_sft/lora_train.jsonl \
  --batch_size 8 \
  --epochs 3 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --save_weight lora_adapter

# === 阶段4: PPO强化学习 ===
echo "开始强化学习..."
python train_graph_ppo.py \
  --from_pretrain graph_llama_sft.pt \
  --num_episodes 1000 \
  --learning_rate 1e-5 \
  --save_weight graph_policy_rl

echo "训练完成！模型保存在 out/ 目录"
```

---

### 单阶段调试

```bash
# 快速测试（小数据+少epoch）
python train_llama_pretrain.py \
  --data_path ../data/pretrain/ \
  --batch_size 4 \
  --epochs 1 \
  --max_seq_len 128 \
  --save_weight debug_test
```

---

### 使用wandb追踪

```bash
# 1. 登录wandb
wandb login

# 2. 启用wandb
python train_llama_pretrain.py \
  --use_wandb \
  --wandb_project "MyTransformer-Pretrain" \
  --batch_size 32 \
  --epochs 10
```

**查看实验**: 访问 https://wandb.ai/your-username/MyTransformer-Pretrain

---

## ❓ 常见问题

### Q1: CUDA Out of Memory (OOM)

**解决方案**:

```bash
# 1. 减小batch_size
--batch_size 8  # 原来16/32

# 2. 启用梯度累积
--batch_size 4 --accumulation_steps 8  # 等效batch=32

# 3. 启用梯度检查点
--use_gradient_checkpointing 1

# 4. 减小序列长度
--max_seq_len 256  # 原来512

# 5. 使用BFloat16
--dtype bfloat16
```

---

### Q2: 训练速度太慢

**优化方案**:

```bash
# 1. 增加num_workers
--num_workers 8  # 加速数据加载

# 2. 启用torch.compile (PyTorch 2.0+)
--use_compile 1

# 3. 使用多GPU
torchrun --nproc_per_node=4 train_xxx.py

# 4. 减小日志频率
--log_interval 100  # 原来10
```

---

### Q3: 断点续训失败

**检查清单**:
```bash
# 1. 确认checkpoint存在
ls checkpoints/llama_pretrain_resume.pt

# 2. 确保参数一致
--save_weight llama_pretrain  # 必须与之前一致

# 3. 启用续训标志
--from_resume 1
```

---

### Q4: LoRA加载失败

**常见错误**:
```python
# ❌ 错误: 直接加载LoRA权重
model = torch.load('lora_adapter.pt')

# ✅ 正确: 先加载base模型，再apply LoRA
base_model = torch.load('graph_llama_sft.pt')
apply_lora(base_model, rank=8, alpha=16)
lora_state = torch.load('lora_adapter.pt')
base_model.load_state_dict(lora_state, strict=False)
```

---

### Q5: Graph数据维度不匹配

**错误信息**: `RuntimeError: Expected feature dim 64, got 128`

**解决**:
```bash
# 确保训练时graph_node_dim与数据一致
python train_graph_sft.py \
  --graph_node_dim 128  # ← 必须匹配你的数据特征维度
```

---

## 📋 参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | - | 数据路径 |
| `--batch_size` | 16 | 批次大小 |
| `--epochs` | 3 | 训练轮数 |
| `--learning_rate` | 3e-4 | 初始学习率 |
| `--device` | cuda:0 | 设备 |
| `--dtype` | bfloat16 | 数据类型 |
| `--num_workers` | 4 | 数据加载线程 |
| `--save_weight` | - | 权重保存名 |
| `--from_resume` | 0 | 是否续训 (0/1) |

---

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vocab_size` | 32000 | 词表大小 |
| `--hidden_size` | 512 | 隐藏层维度 |
| `--num_hidden_layers` | 8 | Transformer层数 |
| `--num_attention_heads` | 8 | 注意力头数 |
| `--num_key_value_heads` | 2 | KV头数(GQA) |
| `--max_seq_len` | 512 | 最大序列长度 |

---

### Graph参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--graph_encoder_type` | gat | 图编码器类型(gcn/gat) |
| `--graph_num_layers` | 3 | GNN层数 |
| `--graph_node_dim` | 64 | 节点特征维度 |

---

### LoRA参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_rank` | 8 | LoRA秩 |
| `--lora_alpha` | 16 | 缩放因子 |
| `--lora_dropout` | 0.05 | Dropout率 |

---

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--accumulation_steps` | 4 | 梯度累积步数 |
| `--grad_clip` | 1.0 | 梯度裁剪 |
| `--warmup_ratio` | 0.1 | Warmup比例 |
| `--weight_decay` | 0.1 | 权重衰减 |
| `--use_gradient_checkpointing` | 1 | 启用梯度检查点 |
| `--use_compile` | 0 | 启用torch.compile |

---

## 🎓 学习路线建议

### 初学者 (第1周)
1. 理解项目结构
2. 准备小规模数据（1000条）
3. 单卡运行阶段1（1-2 epochs）
4. 观察训练日志和loss曲线

### 进阶 (第2-3周)
1. 完成4阶段完整训练
2. 学习调参（lr, batch_size, accumulation_steps）
3. 使用wandb追踪实验
4. 尝试多GPU训练

### 高级 (第4周+)
1. 自定义Graph编码器
2. 修改LoRA应用位置
3. 设计新的强化学习任务
4. 优化训练速度和显存

---

## 📚 参考资料

- **LLaMA论文**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **LoRA论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **GQA论文**: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **PPO论文**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📝 更新日志

### v1.0.0 (2026-02-15)
- ✅ 完整4阶段训练流程
- ✅ 支持DDP分布式训练
- ✅ 集成wandb实验追踪
- ✅ 实现LoRA高效微调
- ✅ 添加断点续训功能
- ✅ Flash Attention优化
- ✅ GQA显存优化
- ✅ NTK-RoPE长文本支持

---

## 📧 联系方式

- **Issue追踪**: [GitHub Issues](https://github.com/your-repo/issues)
- **讨论区**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！⭐**

Made with ❤️ by MyTransformer Team

</div>
