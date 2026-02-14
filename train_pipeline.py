"""
完整训练流程：SFT → RL（PPO）
真正的知识迁移：GraphLlama的graph_encoder → GraphPolicy

流程：
1. Stage 1：SFT训练GraphLlamaForCausalLM（学习图推理的文本描述）
2. Stage 2：提取graph_encoder权重
3. Stage 3：用SFT的graph_encoder初始化RL的policy，继续PPO优化
"""
import torch
import torch.nn.functional as F
from model.llama_model import GraphLlamaForCausalLM
from model.llama_config import LlamaConfig
from model.lora import mark_only_lora_as_trainable, LoRALinear, get_lora_state_dict
from model.graph_policy import GraphPolicy
from model.graph_mission import build_path_task
from model.graph_utils import build_adj_list
from torch_geometric.data import Data
import os
from typing import Optional
from model.tokenizer import get_tokenizer

# ============================================
# Stage 1: SFT训练函数（使用GraphLlama）
# ============================================

def create_graph_sft_data_with_structure():
    """
    加载图推理训练数据（JSONL格式）
    
    返回格式：List[Dict]，每个字典包含：
    {
        "prompt": str,           # 问题文本
        "completion": str,       # 答案文本
        "graph": Data           # PyG图对象 (x, edge_index)
    }
    """
    import json
    
    # ============================================
    # 👇 修改这里：你的JSONL文件路径
    # ============================================
    data_path = "./data/graph_dataset.jsonl"
    
    # ============================================
    # 加载JSONL文件
    # ============================================
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                samples.append(json.loads(line))
    
    print(f"加载了 {len(samples)} 条数据")
    
    # ============================================
    # 👇 数据格式转换（根据你的JSONL字段调整）
    # ============================================
    processed_samples = []
    for item in samples:
        processed_samples.append({
            "prompt": item['prompt'],              # 改成你数据的字段名
            "completion": item['completion'],      # 改成你数据的字段名
            "graph": Data(
                x=torch.tensor(item['node_features'], dtype=torch.float),
                edge_index=torch.tensor(item['edge_index'], dtype=torch.long)
            )
        })
    
    return processed_samples

def tokenize_sft(prompt: str, completion: str, tokenizer, max_len: int = 128):
    """简化token化"""
    # 编码
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    
    # 拼接
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    
    # Truncate
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    
    # Padding
    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

def sft_train_stage(
    config: LlamaConfig,
    tokenizer,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    checkpoint_dir: str = "./checkpoints"
):

    """Stage 1: SFT训练GraphLlamaForCausalLM"""
    print("\n" + "="*60)
    print(" Stage 1: SFT训练 (GraphLlama)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建GraphLlama模型（重点：有graph_encoder）
    model = GraphLlamaForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: GraphLlamaForCausalLM")
    print(f"总参数: {total_params/1e6:.2f}M")
    
    # 2. 应用LoRA
    if use_lora:
        apply_lora_to_model(model, config)
        mark_only_lora_as_trainable(model, bias='none')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA可训练: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
    
    # 3. 准备数据（带图结构）
    samples = create_graph_sft_data_with_structure()
    print(f"训练样本: {len(samples)}")
    
    # 4. 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 5. 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            
            # 构造batch
            batch_data = [
                tokenize_sft(s["prompt"], s["completion"], tokenizer)  # ← 传tokenizer
                for s in batch_samples
            ]
            input_ids = torch.stack([d["input_ids"] for d in batch_data]).to(device)
            labels = torch.stack([d["labels"] for d in batch_data]).to(device)
            
            # 图数据（简化：用第一个样本的图）
            graph_data = {
                "node_features": batch_samples[0]["graph"].x.to(device),
                "edge_index": batch_samples[0]["graph"].edge_index.to(device)
            }
            
            # 前向+反向
            optimizer.zero_grad()
            loss, logits = model(input_ids, labels=labels, graph_data=graph_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 6. 保存完整模型（包含graph_encoder）
    model.eval()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config
    }
    save_path = os.path.join(checkpoint_dir, "sft_graphllama.pt")
    
    torch.save(checkpoint, save_path)
    print(f"\n SFT完成，checkpoint: {save_path}")
    print(f"   包含graph_encoder权重（将迁移到RL）\n")
    
    return model, save_path

# ============================================
# Stage 2: RL训练（继承graph_encoder）
# ============================================

def rl_train_stage(
    sft_checkpoint_path: str,
    num_episodes: int = 300,
    learning_rate: float = 1e-4,
    use_sft_init: bool = True
):
    """Stage 2: RL训练（继承SFT的graph_encoder）"""
    print("\n" + "="*60)
    print(" Stage 2: RL训练 (PPO)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建策略网络
    policy = GraphPolicy(
        node_dim=64,
        hidden_size=128,
        num_layers=2,
        encoder_type="gat"
    ).to(device)
    
    # 2. 加载SFT的graph_encoder权重（核心！）
    if use_sft_init and os.path.exists(sft_checkpoint_path):
        print(f"加载SFT checkpoint: {sft_checkpoint_path}")
        checkpoint = torch.load(sft_checkpoint_path, map_location=device, weights_only=False)
        sft_state = checkpoint['model_state_dict']
        
        # 提取graph_encoder权重
        policy_state = policy.state_dict()
        transferred = 0
        
        for name, param in sft_state.items():
            # GraphLlama的graph_encoder → Policy的encoder
            if 'graph_encoder' in name:
                # 去掉前缀 'graph_encoder.' 映射到 policy 的 'encoder.'
                new_name = name.replace('graph_encoder', 'encoder')
                if new_name in policy_state and param.shape == policy_state[new_name].shape:
                    policy_state[new_name] = param.clone()
                    transferred += 1
        
        if transferred > 0:
            policy.load_state_dict(policy_state)
            print(f" 成功迁移 {transferred} 个graph_encoder参数")
            print(f"   Policy现在具备SFT学到的图理解能力！")
        else:
            print("  未找到匹配的graph_encoder权重，使用随机初始化")
    else:
        print("使用随机初始化（无SFT预训练）")
    
    # 3. PPO训练
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    
    task = build_path_task()
    graph = task["graph"]
    num_nodes = graph.x.size(0)
    adj = build_adj_list(graph.edge_index, num_nodes)
    
    clip_epsilon = 0.2
    ppo_epochs = 4
    
    print(f"\n开始PPO训练，episodes: {num_episodes}")
    
    for episode in range(num_episodes):
        current = task["start"]
        goal = task["goal"]
        max_steps = task["max_steps"]
        
        states, actions, old_log_probs, rewards = [], [], [], []
        visited = set([current])
        path_nodes = [current]
        
        # 收集轨迹
        for step in range(max_steps):
            logits = policy(graph.x, graph.edge_index, current, goal)
            
            mask = torch.full_like(logits, float("-inf"))
            for n in adj[current]:
                mask[n] = 0.0
            masked_logits = logits + mask
            
            probs = torch.softmax(masked_logits, dim=-1)
            next_node = torch.multinomial(probs, num_samples=1).item()
            old_log_prob = torch.log(probs[next_node] + 1e-9)
            
            reward = -0.01
            if next_node == goal:
                reward += 1.0
            if next_node in visited:
                reward -= 0.05
            visited.add(next_node)
            
            states.append((current, goal))
            actions.append(next_node)
            old_log_probs.append(old_log_prob.detach())
            rewards.append(reward)
            
            path_nodes.append(next_node)
            current = next_node
            
            if current == goal:
                break
        
        # 路径对齐奖励
        gold_path = [int(x.strip()) for x in task["gold_path"].split("->")]
        if path_nodes == gold_path:
            rewards[-1] += 0.5
        
        # 计算returns和advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # PPO更新
        for _ in range(ppo_epochs):
            total_loss = 0
            for t in range(len(states)):
                curr, g = states[t]
                action = actions[t]
                old_lp = old_log_probs[t]
                adv = advantages[t]
                
                logits = policy(graph.x, graph.edge_index, curr, g)
                mask = torch.full_like(logits, float("-inf"))
                for n in adj[curr]:
                    mask[n] = 0.0
                masked_logits = logits + mask
                
                probs = torch.softmax(masked_logits, dim=-1)
                new_log_prob = torch.log(probs[action] + 1e-9)
                
                ratio = torch.exp(new_log_prob - old_lp)
                clipped = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
                loss = -torch.min(ratio * adv, clipped * adv)
                
                total_loss += loss
            
            if total_loss > 0:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Steps: {len(path_nodes)}, "
                  f"Goal: {current == goal}, Reward: {sum(rewards):.2f}")
    
    print("\n RL训练完成\n")
    return policy

# ============================================
# LoRA辅助函数
# ============================================

def apply_lora_to_model(model, config: LlamaConfig):
    """应用LoRA到模型"""
    if not config.use_lora:
        return
    
    target_modules = config.lora_target_modules or ["q_proj", "v_proj"]
    
    def replace_linear(module, name=""):
        for attr_name in dir(module):
            try:
                target_attr = getattr(module, attr_name)
            except:
                continue
                
            if isinstance(target_attr, torch.nn.Linear):
                if any(target in attr_name for target in target_modules):
                    in_features = target_attr.in_features
                    out_features = target_attr.out_features
                    bias = target_attr.bias is not None
                    
                    lora_layer = LoRALinear(
                        in_features=in_features,
                        out_features=out_features,
                        r=config.lora_r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        bias=bias
                    )
                    
                    lora_layer.weight.data = target_attr.weight.data.clone()
                    if bias:
                        lora_layer.bias.data = target_attr.bias.data.clone()
                    
                    setattr(module, attr_name, lora_layer)
        
        for child_name, child_module in module.named_children():
            replace_linear(child_module, f"{name}.{child_name}" if name else child_name)
    
    replace_linear(model)

# ============================================
# 主流程
# ============================================

def main():
    """完整训练流程：SFT → RL 知识迁移"""
    print("\n" + "="*60)
    print("AI+RL+Graph 完整训练流程")
    print("知识迁移: SFT的graph_encoder → RL的policy")
    print("="*60)
    
    # 配置
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,  # 与graph_encoder对齐
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj"],
        tokenizer_type="tiktoken",      # 或 "huggingface"
        tokenizer_name="cl100k_base",   # 或 "Qwen/Qwen-7B" 或 "gpt2"
        # ← 新增Graph配置
        use_graph=True,
        graph_node_dim=64,
        graph_num_layers=2,
        graph_encoder_type="gat"
    )

    # 2. 根据config创建tokenizer
    print(f"\n初始化 {config.tokenizer_type} Tokenizer...")
    
    if config.tokenizer_type == "tiktoken":
        tokenizer = get_tokenizer("tiktoken", encoding_name=config.tokenizer_name)
    elif config.tokenizer_type == "huggingface":
        tokenizer = get_tokenizer("huggingface", model_name=config.tokenizer_name)
    else:
        raise ValueError(f"不支持的tokenizer类型: {config.tokenizer_type}")
    
    # 3. 同步vocab_size到config
    config.vocab_size = tokenizer.vocab_size
    print(f" Tokenizer加载完成，vocab_size={config.vocab_size}")
    
    # Stage 1: SFT（训练GraphLlama）
    model, sft_path = sft_train_stage(
        config=config,
        tokenizer=tokenizer,
        num_epochs=5,
        batch_size=2,
        learning_rate=3e-4,
        use_lora=True
    )
    
    # Stage 2: RL（继承graph_encoder）
    policy = rl_train_stage(
        sft_checkpoint_path=sft_path,
        num_episodes=200,
        learning_rate=1e-4,
        use_sft_init=True
    )
    
    # 保存最终策略
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict()
    }, "./checkpoints/final_policy.pt")
    
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print("\n文件:")
    print("  - SFT: ./checkpoints/sft_graphllama.pt")
    print("  - RL:  ./checkpoints/final_policy.pt")
    print("\n下一步: python eval_complete.py\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "v_proj"],
            tokenizer_type="tiktoken",
            tokenizer_name="cl100k_base",
            # Graph配置
            use_graph=True,
            graph_node_dim=64,
            graph_num_layers=2,
            graph_encoder_type="gat"
        )
        
        if mode == "sft":
            print("\n🎯 模式：仅SFT\n")
            # 创建tokenizer
            tokenizer = get_tokenizer(
                config.tokenizer_type, 
                encoding_name=config.tokenizer_name if config.tokenizer_type == "tiktoken" else None,
                model_name=config.tokenizer_name if config.tokenizer_type != "tiktoken" else None
            )
            config.vocab_size = tokenizer.vocab_size
            sft_train_stage(config, tokenizer, num_epochs=5)
            
        elif mode == "rl":
            print("\n🎯 模式：仅RL\n")
            sft_path = "./checkpoints/sft_graphllama.pt"
            rl_train_stage(sft_path, num_episodes=200)
            
        else:
            print(f"用法: python train_pipeline.py [sft|rl]")
    else:
        main()