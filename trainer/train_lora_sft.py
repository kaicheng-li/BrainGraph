"""
LoRA微调脚本（Graph+LLM）
基础模型：checkpoints/graph_llama_sft.pt
训练方式：只训练LoRA参数，冻结主模型
输出：checkpoints/lora_adapter.pt
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from model import LlamaConfig, GraphLlamaForCausalLM, get_tokenizer
from model.llama_model import LoRALinear
from datasets import GraphSFTDataset, collate_graph_batch

def apply_lora_to_model(model, rank=8, alpha=16):
    """
    给模型的所有Linear层添加LoRA
    LoRA只在Attention的Q,K,V,O上
    """
    lora_params = []
    
    for name, module in model.named_modules():
        # 只对Attention层的投影矩阵应用LoRA
        if isinstance(module, nn.Linear) and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            # 替换为LoRALinear
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha,
                bias=module.bias is not None
            )
            lora_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.linear.bias.data = module.bias.data.clone()
            
            # 设置到模型中
            parent = model
            for attr in parent_name.split('.'):
                if attr:
                    parent = getattr(parent, attr)
            setattr(parent, attr_name, lora_layer)
            
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
    
    return lora_params

def train_lora_sft():
    # ========== 加载SFT模型 ==========
    checkpoint = torch.load("checkpoints/graph_llama_sft.pt", map_location="cpu")
    config_dict = checkpoint['config']
    config = LlamaConfig(**config_dict)
    
    model = GraphLlamaForCausalLM(config, graph_hidden_size=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ========== 应用LoRA ==========
    lora_params = apply_lora_to_model(model, rank=8, alpha=16)
    
    # 冻结主模型，只训练LoRA
    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 LoRA参数：{trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ========== 数据 ==========
    tokenizer = get_tokenizer(tokenizer_type="tiktoken", model_name="gpt-4")
    dataset = GraphSFTDataset(
        data_path="./data/graph_sft/",
        tokenizer=tokenizer,
        max_length=256
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_graph_batch,
        num_workers=2
    )
    
    # ========== 优化器（只优化LoRA参数）==========
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    
    # ========== 训练 ==========
    print(f"🚀 开始LoRA微调：{len(dataset)} 样本")
    
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_map = batch['batch_map'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                node_features=node_features,
                edge_index=edge_index,
                batch_map=batch_map
            )
            loss = outputs["loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f" Epoch {epoch+1} 完成 | 平均Loss: {avg_loss:.4f}")
    
    # ========== 保存LoRA适配器 ==========
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                'lora_A': module.lora_A.data,
                'lora_B': module.lora_B.data
            }
    
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'lora_state': lora_state,
        'config': {'rank': 8, 'alpha': 16},
        'base_model': 'checkpoints/graph_llama_sft.pt'
    }, "checkpoints/lora_adapter.pt")
    
    print(" LoRA微调完成：checkpoints/lora_adapter.pt")

if __name__ == "__main__":
    train_lora_sft()