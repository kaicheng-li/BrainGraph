"""
LoRA微调脚本 (Qwen2.5优化)
 LoRA高效微调 (QLoRA支持)
 混合精度训练
 梯度检查点
 只训练0.1%参数
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import math

from model import LlamaConfig, GraphLlamaForCausalLM, get_tokenizer
from model.llama_model import LoRALinear
from datasets import GraphSFTDataset, collate_graph_batch

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Warmup + Cosine Decay"""
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None):
    """
    给模型添加LoRA适配器
    默认应用到: Q, K, V, O投影层
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    lora_params = []
    replaced_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in target_modules):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            # 创建LoRALinear层
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha,
                bias=module.bias is not None
            )
            # 复制原始权重
            lora_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.linear.bias.data = module.bias.data.clone()
            
            # 替换模块
            parent = model
            for attr in parent_name.split('.'):
                if attr:
                    parent = getattr(parent, attr)
            setattr(parent, attr_name, lora_layer)
            
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            replaced_count += 1
    
    print(f" LoRA应用: 替换了 {replaced_count} 个线性层")
    return lora_params

def train_lora_sft(
    base_model_path="checkpoints/graph_llama_sft.pt",
    data_path="./data/graph_sft/",
    output_path="checkpoints/lora_adapter.pt",
    # === LoRA配置 ===
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,  # None = Q,K,V,O
    # === 训练配置 ===
    batch_size=16,
    gradient_accumulation_steps=2,
    num_epochs=3,
    learning_rate=1e-4,
    min_lr=1e-5,
    warmup_ratio=0.03,
    # === 优化配置 ===
    use_gradient_checkpointing=True,
    use_amp=True,
    amp_dtype="bfloat16",
):
    """LoRA微调 (Qwen2.5优化)"""
    print("\n" + "="*70)
    print(" LoRA微调 (Qwen2.5优化)")
    print("="*70)
    
    # ========== 1. 加载基础模型 ==========
    print(f"\n🔄 加载基础模型: {base_model_path}")
    checkpoint = torch.load(base_model_path, map_location="cpu")
    config_dict = checkpoint['config']
    config = LlamaConfig(**config_dict)
    
    model = GraphLlamaForCausalLM(config, graph_hidden_size=config.graph_node_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(" 基础模型加载完成")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 2. 应用LoRA ==========
    print(f"\n🔧 应用LoRA (rank={lora_rank}, alpha={lora_alpha})")
    lora_params = apply_lora_to_model(model, rank=lora_rank, alpha=lora_alpha, target_modules=target_modules)
    
    # 冻结主模型，只训练LoRA
    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # 梯度检查点
    if use_gradient_checkpointing and torch.cuda.is_available():
        if hasattr(model, 'llama'):
            model.llama.enable_gradient_checkpointing()
        else:
            model.enable_gradient_checkpointing()
        print("💾 梯度检查点: 已启用")
    
    model.to(device)
    
    # ========== 3. 数据 ==========
    tokenizer = get_tokenizer(tokenizer_type="tiktoken", model_name="gpt-4")
    dataset = GraphSFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_graph_batch,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n📊 数据: {len(dataset)} 样本")
    
    # ========== 4. 优化器 ==========
    optimizer = torch.optim.AdamW(
        lora_params, 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # ========== 5. 混合精度 ==========
    scaler = None
    dtype = torch.float32
    if use_amp and torch.cuda.is_available():
        if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("🔥 混合精度: BFloat16")
        else:
            dtype = torch.float16
            scaler = GradScaler()
            print("⚡ 混合精度: Float16")
    
    # ========== 6. 训练循环 ==========
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)
    
    print(f"\n📈 训练: {total_steps:,} steps (warmup: {warmup_steps})")
    print(f"{'='*70}\n")
    
    global_step = 0
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_map = batch['batch_map'].to(device)
            
            # 前向 (混合精度)
            with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    node_features=node_features,
                    edge_index=edge_index,
                    batch_map=batch_map
                )
                loss = outputs["loss"] / gradient_accumulation_steps
            
            # 反向
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                lr = get_lr(global_step, warmup_steps, total_steps, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 5 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                          f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | LR: {lr:.6f}")
            
            epoch_loss += loss.item() * gradient_accumulation_steps
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n Epoch {epoch+1} 完成 | Loss: {avg_loss:.4f}\n")
    
    # ========== 7. 保存LoRA适配器 ==========
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                'lora_A': module.lora_A.data.cpu(),
                'lora_B': module.lora_B.data.cpu()
            }
    
    Path(output_path).parent.mkdir(exist_ok=True)
    torch.save({
        'lora_state': lora_state,
        'lora_config': {
            'rank': lora_rank, 
            'alpha': lora_alpha,
            'target_modules': target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        },
        'base_model': base_model_path,
        'training_args': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'final_loss': avg_loss,
        }
    }, output_path)
    
    print(f"\n{'='*70}")
    print(f" LoRA微调完成: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train_lora_sft()