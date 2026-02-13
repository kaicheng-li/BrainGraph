"""
完整训练流程：SFT → RL（PPO）
支持LoRA微调 + NTK-RoPE长上下文

流程：
1. Stage 1：SFT预训练（图推理任务监督学习）
2. Stage 2：保存SFT checkpoint
3. Stage 3：加载SFT模型，用PPO优化策略
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from model.llama_model import LlamaForCausalLM, GraphLlamaForCausalLM
from model.llama_config import LlamaConfig
from model.lora import mark_only_lora_as_trainable, LoRALinear, get_lora_state_dict
from model.graph_policy import GraphPolicy
from model.graph_mission import build_path_task
from model.graph_utils import build_adj_list
import os
from typing import Optional

# ============================================
# Stage 1: SFT训练函数
# ============================================

def create_graph_sft_data():
    """生成图推理的SFT训练数据"""
    samples = [
        {
            "prompt": "图结构: 节点0→1→4→5, 节点0→2→5. 问题: 从0到5的最短路径?",
            "completion": "分析: 路径1是0→1→4→5(3步), 路径2是0→2→5(2步). 答案: 0→2→5"
        },
        {
            "prompt": "图: 边[(0,1),(1,2),(2,3),(0,3)]. 问: 节点0到3有几条路径?",
            "completion": "有2条路径: ①0→1→2→3(3步) ②0→3(1步，最短)"
        },
        {
            "prompt": "图: 5节点完全图. 问: 从节点0出发，访问所有节点的最短步数?",
            "completion": "完全图中任意两点直连，访问5个节点最少需要4步(从0到1,2,3,4)"
        },
        # 添加更多样本提高泛化
        {
            "prompt": "图: 链状0-1-2-3-4. 问: 0到4的路径?",
            "completion": "唯一路径: 0→1→2→3→4(4步)"
        },
        {
            "prompt": "图: 环状0→1→2→3→0. 问: 从0回到0需要几步?",
            "completion": "顺时针: 0→1→2→3→0(4步), 这是唯一路径"
        }
    ]
    return samples * 20  # 复制20次增加训练数据量

def tokenize_sft_sample(prompt: str, completion: str, vocab_size: int, max_len: int = 128):
    """简化的token化（实际应用需要真实tokenizer）"""
    # 字符级编码
    def encode(text):
        return [min(ord(c) % vocab_size, vocab_size-1) for c in text]
    
    prompt_tokens = encode(prompt)[:max_len//2]
    completion_tokens = encode(completion)[:max_len//2]
    
    input_ids = prompt_tokens + completion_tokens
    labels = [-100] * len(prompt_tokens) + completion_tokens
    
    # Padding
    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids += [0] * pad_len
        labels += [-100] * pad_len
    
    return {
        "input_ids": torch.tensor(input_ids[:max_len], dtype=torch.long),
        "labels": torch.tensor(labels[:max_len], dtype=torch.long)
    }

def sft_train_stage(
    config: LlamaConfig,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    use_lora: bool = True,
    checkpoint_dir: str = "./checkpoints"
):
    """Stage 1: SFT训练"""
    print("\n" + "="*60)
    print("📚 Stage 1: Supervised Fine-Tuning (SFT)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建模型
    model = LlamaForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params/1e6:.2f}M")
    
    # 2. 应用LoRA（如果启用）
    if use_lora:
        # 替换Linear为LoRALinear
        apply_lora_to_model(model, config)
        mark_only_lora_as_trainable(model, bias='none')
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA可训练参数: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")
    
    # 3. 准备数据
    samples = create_graph_sft_data()
    print(f"训练样本数: {len(samples)}")
    
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
                tokenize_sft_sample(s["prompt"], s["completion"], config.vocab_size)
                for s in batch_samples
            ]
            input_ids = torch.stack([d["input_ids"] for d in batch_data]).to(device)
            labels = torch.stack([d["labels"] for d in batch_data]).to(device)
            
            # 前向+反向
            optimizer.zero_grad()
            loss, logits = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 6. 保存checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    if use_lora:
        # 只保存LoRA权重
        checkpoint = {
            "lora_state_dict": get_lora_state_dict(model),
            "config": config
        }
        save_path = os.path.join(checkpoint_dir, "sft_lora.pt")
    else:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config
        }
        save_path = os.path.join(checkpoint_dir, "sft_full.pt")
    
    torch.save(checkpoint, save_path)
    print(f"\n[OK] SFT训练完成，checkpoint保存至: {save_path}\n")
    
    return model, save_path

# ============================================
# Stage 2: RL（PPO）训练函数
# ============================================

def rl_train_stage(
    sft_checkpoint_path: str,
    config: LlamaConfig,
    num_episodes: int = 500,
    learning_rate: float = 1e-4,
    use_lora: bool = True
):
    """Stage 2: 基于SFT模型的RL训练"""
    print("\n" + "="*60)
    print(" Stage 2: Reinforcement Learning (PPO)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载SFT checkpoint
    print(f"加载SFT checkpoint: {sft_checkpoint_path}")
    checkpoint = torch.load(sft_checkpoint_path, map_location=device)
    
    # 2. 创建策略网络
    policy = GraphPolicy(
        node_dim=64,
        hidden_size=128,
        num_layers=2,
        encoder_type="gat"
    ).to(device)
    
    # 可选：用SFT的graph encoder初始化policy
    # 这里简化处理，实际可以共享编码器权重
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    
    # 3. PPO训练
    task = build_path_task()
    graph = task["graph"]
    num_nodes = graph.x.size(0)
    adj = build_adj_list(graph.edge_index, num_nodes)
    
    clip_epsilon = 0.2
    ppo_epochs = 4
    
    print(f"开始PPO训练，总episodes: {num_episodes}")
    
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
            
            # Mask非邻居节点
            mask = torch.full_like(logits, float("-inf"))
            for n in adj[current]:
                mask[n] = 0.0
            masked_logits = logits + mask
            
            probs = torch.softmax(masked_logits, dim=-1)
            next_node = torch.multinomial(probs, num_samples=1).item()
            old_log_prob = torch.log(probs[next_node] + 1e-9)
            
            # 奖励
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
        
        # 额外奖励（路径对齐）
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
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Steps: {len(path_nodes)}, "
                  f"Reached goal: {current == goal}, "
                  f"Total reward: {sum(rewards):.2f}")
    
    print("\n[OK] RL训练完成\n")
    return policy

# ============================================
# LoRA应用辅助函数
# ============================================

def apply_lora_to_model(model: LlamaForCausalLM, config: LlamaConfig):
    """将模型的Linear层替换为LoRALinear"""
    if not config.use_lora:
        return
    
    target_modules = config.lora_target_modules or ["q_proj", "v_proj"]
    
    def replace_linear(module, name=""):
        for attr_name in dir(module):
            target_attr = getattr(module, attr_name)
            if isinstance(target_attr, torch.nn.Linear):
                # 检查是否在目标模块列表中
                if any(target in attr_name for target in target_modules):
                    in_features = target_attr.in_features
                    out_features = target_attr.out_features
                    bias = target_attr.bias is not None
                    
                    # 创建LoRALinear替换
                    lora_layer = LoRALinear(
                        in_features=in_features,
                        out_features=out_features,
                        r=config.lora_r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        bias=bias
                    )
                    
                    # 复制原权重
                    lora_layer.weight.data = target_attr.weight.data.clone()
                    if bias:
                        lora_layer.bias.data = target_attr.bias.data.clone()
                    
                    setattr(module, attr_name, lora_layer)
                    print(f"  ✓ 替换 {name}.{attr_name} 为 LoRALinear")
        
        # 递归处理子模块
        for child_name, child_module in module.named_children():
            replace_linear(child_module, f"{name}.{child_name}" if name else child_name)
    
    print("应用LoRA到模型:")
    replace_linear(model)

# ============================================
# 主流程
# ============================================

def main():
    """完整训练流程"""
    print("\n" + "="*60)
    print(" AI+RL+Graph 完整训练流程")
    print("="*60)
    
    # 配置
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        # NTK-RoPE配置
        rope_scaling={"type": "ntk", "factor": 2.0},
        # LoRA配置
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj", "o_proj"]
    )
    
    # Stage 1: SFT
    model, sft_path = sft_train_stage(
        config=config,
        num_epochs=5,
        batch_size=4,
        learning_rate=3e-4,
        use_lora=True
    )
    
    # Stage 2: RL
    policy = rl_train_stage(
        sft_checkpoint_path=sft_path,
        config=config,
        num_episodes=300,
        learning_rate=1e-4,
        use_lora=True
    )
    
    # 保存最终模型
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "config": config
    }, "./checkpoints/final_policy.pt")
    
    print("\n" + "="*60)
    print(" 训练流程全部完成！")
    print("="*60)
    print("\n模型文件:")
    print("  - SFT checkpoint: ./checkpoints/sft_lora.pt")
    print("  - Final policy: ./checkpoints/final_policy.pt")
    print("\n下一步:")
    print("  1. 运行评估脚本测试性能")
    print("  2. 可视化训练曲线")
    print("  3. 尝试更复杂的图任务（如旅行商问题TSP）\n")

if __name__ == "__main__":
    main()