"""
LLaMA预训练脚本（Qwen2.5/GPT-4级优化）
 Flash Attention 2
 GQA (Grouped Query Attention)
 梯度检查点 (Gradient Checkpointing)
 混合精度训练 (AMP FP16/BF16)
 梯度累积 (Gradient Accumulation)
 Warmup + Cosine LR Schedule
 梯度裁剪 (Gradient Clipping)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import math

from model import LlamaConfig, LlamaForCausalLM, get_tokenizer
from datasets import PretrainDataset

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Warmup + Cosine Decay (Qwen2.5风格)"""
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def train_llama_pretrain(
    data_path="./data/pretrain/",
    checkpoint_dir="./checkpoints",
    # === 模型配置 ===
    vocab_size=32000,
    hidden_size=2048,              # Qwen2.5-7B: 2048
    num_layers=24,                 # Qwen2.5-7B: 24层
    num_heads=16,
    num_kv_heads=4,                # GQA: 4个KV头
    max_seq_len=4096,              # 支持4K上下文
    # === 训练配置 ===
    batch_size=16,
    gradient_accumulation_steps=4,  # 等效batch=64
    num_epochs=3,
    learning_rate=3e-4,
    min_lr=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
    # === 优化配置 ===
    use_flash_attn=True,
    use_gradient_checkpointing=True,  # 节省70%显存
    use_amp=True,                      # 混合精度
    amp_dtype="bfloat16",              # bfloat16 > float16 (Qwen2.5用法)
):
    """
    Qwen2.5/GPT-4级别的预训练脚本
    """
    print("\n" + "="*70)
    print(" LLaMA预训练 (Qwen2.5优化)")
    print("="*70)
    
    # ========== 1. 配置 ==========
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=hidden_size * 4,  # SwiGLU: 4x
        max_position_embeddings=max_seq_len,
        use_flash_attn=use_flash_attn,
        rope_scaling={"type": "ntk", "factor": 2.0},  # NTK-RoPE扩展上下文
    )
    
    print(f"📐 模型配置:")
    print(f"  - 隐藏层: {hidden_size}d × {num_layers}层")
    print(f"  - 注意力: {num_heads}头 (GQA: {num_kv_heads} KV头)")
    print(f"  - 上下文: {max_seq_len} tokens")
    print(f"  - Flash Attention: {use_flash_attn}")
    
    # ========== 2. 数据 ==========
    tokenizer = get_tokenizer(tokenizer_type="tiktoken", model_name="gpt-4")
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq_len
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # 加速GPU传输
    )
    
    effective_batch = batch_size * gradient_accumulation_steps
    print(f"\n📊 数据配置:")
    print(f"  - 训练样本: {len(dataset):,}")
    print(f"  - Batch size: {batch_size} × {gradient_accumulation_steps} = {effective_batch}")
    print(f"  - 每轮steps: {len(dataloader) // gradient_accumulation_steps}")
    
    # ========== 3. 模型 ==========
    model = LlamaForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 启用梯度检查点（节省显存）
    if use_gradient_checkpointing and torch.cuda.is_available():
        model.enable_gradient_checkpointing()
        print(f"\n💾 梯度检查点: 已启用 (节省~70%显存)")
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 总参数: {total_params/1e9:.2f}B")
    
    # ========== 4. 优化器 (AdamW with Qwen2.5配置) ==========
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.95),      # Qwen2.5/Llama3: β2=0.95
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    # ========== 5. 混合精度 ==========
    scaler = None
    dtype = torch.float32
    if use_amp and torch.cuda.is_available():
        if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print(f"🔥 混合精度: BFloat16 (Qwen2.5标准)")
        else:
            dtype = torch.float16
            scaler = GradScaler()
            print(f"⚡ 混合精度: Float16 + GradScaler")
    
    # ========== 6. 学习率调度 ==========
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)
    
    print(f"\n📈 训练配置:")
    print(f"  - 总steps: {total_steps:,}")
    print(f"  - Warmup: {warmup_steps} steps ({warmup_ratio*100:.0f}%)")
    print(f"  - LR: {learning_rate} → {min_lr} (Cosine)")
    print(f"  - 梯度裁剪: {max_grad_norm}")
    
    # ========== 7. 训练循环 ==========
    print(f"\n{'='*70}")
    print(" 开始训练")
    print(f"{'='*70}\n")
    
    global_step = 0
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # === 前向传播 (混合精度) ===
            with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                loss = loss / gradient_accumulation_steps  # 梯度累积归一化
            
            # === 反向传播 ===
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # === 梯度累积 ===
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 动态学习率
                lr = get_lr(global_step, warmup_steps, total_steps, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 梯度裁剪
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                # 日志
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Step {global_step}/{total_steps} | "
                          f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | "
                          f"LR: {lr:.6f}")
            
            epoch_loss += loss.item() * gradient_accumulation_steps
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n Epoch {epoch+1} 完成 | 平均Loss: {avg_loss:.4f}\n")
        
        # 保存checkpoint
        Path(checkpoint_dir).mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'loss': avg_loss,
        }, f"{checkpoint_dir}/llama_pretrain_epoch{epoch+1}.pt")
    
    # ========== 8. 保存最终模型 ==========
    final_path = f"{checkpoint_dir}/llama_pretrain.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'training_args': {
            'num_epochs': num_epochs,
            'total_steps': global_step,
            'final_loss': avg_loss,
        }
    }, final_path)
    
    print(f"\n{'='*70}")
    print(f" 预训练完成: {final_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train_llama_pretrain()