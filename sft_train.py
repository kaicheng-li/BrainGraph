# sft_train.py
"""
Supervised Fine-Tuning (SFT) 框架
论文：Training language models to follow instructions with human feedback (InstructGPT, Ouyang et al., 2022)

训练流程：
1. 数据格式：{"prompt": "...", "completion": "..."}
2. 损失函数：只计算completion部分的交叉熵
3. 支持LoRA微调（可选）
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from model.llama_model import LlamaForCausalLM
from model.llama_config import LlamaConfig
from model.lora import mark_only_lora_as_trainable
from typing import List, Dict
import math

def create_sft_sample_graph_task():
    """
    创建图推理的SFT样本
    格式：Instruction + Graph Description → Reasoning Path
    """
    samples = [
        {
            "prompt": "Graph: 节点0连接到节点1, 节点1连接到节点4, 节点4连接到节点5. 任务: 从节点0到节点5的最短路径是什么?",
            "completion": "最短路径是: 0 -> 1 -> 4 -> 5. 理由: 这条路径总共3步，没有更短的路径。"
        },
        {
            "prompt": "Graph: 边列表 [(0,1), (1,2), (2,3), (0,3)]. 任务: 判断节点0和节点3之间有多少条路径?",
            "completion": "有2条路径: 路径1是 0->1->2->3 (3步), 路径2是 0->3 (1步). 最短路径是 0->3."
        },
        {
            "prompt": "Graph: 节点[0,1,2,3,4], 边[(0,1),(1,2),(2,3),(3,4),(0,4)]. 从节点0到节点4，走最短路径。",
            "completion": "最短路径: 0 -> 4 (直达边，1步)."
        }
    ]
    return samples

def prepare_sft_batch(samples: List[Dict], tokenizer=None, max_length: int = 512):
    """
    准备SFT训练批次
    关键：只计算completion部分的损失（prompt部分用-100 mask掉）
    
    返回：
    - input_ids: [batch, seq_len]
    - labels: [batch, seq_len]，prompt部分=-100
    """
    if tokenizer is None:
        # 简化版：用字符级tokenizer演示
        def simple_tokenize(text):
            # 这里应该用真实的tokenizer（如sentencepiece）
            # 为了演示，用字符ASCII码
            return [ord(c) % 1000 for c in text[:max_length]]
        
        batch_input_ids = []
        batch_labels = []
        
        for sample in samples:
            prompt_tokens = simple_tokenize(sample["prompt"])
            completion_tokens = simple_tokenize(sample["completion"])
            
            # 拼接
            input_ids = prompt_tokens + completion_tokens
            
            # labels：prompt部分用-100标记（不计算损失）
            labels = [-100] * len(prompt_tokens) + completion_tokens
            
            # Padding到max_length
            if len(input_ids) < max_length:
                pad_len = max_length - len(input_ids)
                input_ids += [0] * pad_len
                labels += [-100] * pad_len
            
            batch_input_ids.append(input_ids[:max_length])
            batch_labels.append(labels[:max_length])
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }
    else:
        # TODO: 使用真实tokenizer的实现
        raise NotImplementedError("需要实现真实的tokenizer")

def sft_train(
    model: LlamaForCausalLM,
    samples: List[Dict],
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    use_lora: bool = False,
    use_amp: bool = True
):
    """
    SFT训练主循环
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 如果使用LoRA，只训练LoRA参数
    if use_lora:
        mark_only_lora_as_trainable(model, bias='none')
        print(f"LoRA模式：只训练 {sum(p.numel() for p in model.parameters() if p.requires_grad)} 个参数")
    else:
        print(f"全量微调：训练 {sum(p.numel() for p in model.parameters())} 个参数")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    scaler = GradScaler() if use_amp else None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 简单的批次循环（实际应该用DataLoader）
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch = prepare_sft_batch(batch_samples)
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if use_amp:
                with autocast():
                    logits, _ = model(input_ids)  # [B, T, vocab]
                    
                    # 计算损失（只对非-100的位置）
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(samples) / batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def main():
    """
    演示：在图推理任务上做SFT
    """
    print("=== SFT训练演示 ===\n")
    
    # 1. 配置（启用LoRA）
    config = LlamaConfig(
        vocab_size=1000,  # 简化演示
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"]  # 只对Q和V做LoRA
    )
    
    # 2. 创建模型
    model = LlamaForCausalLM(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    
    # 3. 准备数据
    samples = create_sft_sample_graph_task()
    print(f"训练样本数: {len(samples)}")
    print(f"样本示例:\nPrompt: {samples[0]['prompt'][:50]}...\nCompletion: {samples[0]['completion'][:50]}...\n")
    
    # 4. 训练
    model = sft_train(
        model=model,
        samples=samples,
        num_epochs=10,
        batch_size=2,
        learning_rate=1e-4,
        use_lora=True,
        use_amp=False  # CPU演示关闭AMP
    )
    
    print("\n SFT训练完成！")
    print("下一步：可以用这个SFT模型初始化RL训练（PPO）")

if __name__ == "__main__":
    main()