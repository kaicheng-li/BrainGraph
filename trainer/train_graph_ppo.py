"""
Graph PPO训练 (Qwen2.5优化)
 PPO算法 (Proximal Policy Optimization)
 知识迁移 (从SFT的graph_encoder)
 混合精度训练
 优势函数归一化
 梯度裁剪
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.graph_policy import GraphPolicy
from model.graph_mission import build_path_task
from model.graph_utils import build_adj_list

def train_graph_ppo(
    sft_checkpoint_path="checkpoints/graph_llama_sft.pt",
    output_path="checkpoints/graph_policy_rl.pt",
    # === PPO配置 ===
    num_episodes=500,
    max_steps_per_episode=20,
    ppo_epochs=4,
    clip_epsilon=0.2,
    gamma=0.99,               # 折扣因子
    gae_lambda=0.95,          # GAE参数
    value_loss_coef=0.5,      # Value loss权重
    entropy_coef=0.01,        # 熵正则化
    # === 训练配置 ===
    learning_rate=3e-4,
    batch_size=32,
    max_grad_norm=0.5,
    # === 优化配置 ===
    use_amp=True,
    amp_dtype="bfloat16",
):
    """PPO强化学习 (Qwen2.5优化)"""
    print("\n" + "="*70)
    print(" Graph PPO训练 (Qwen2.5优化)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # ========== 1. 加载SFT checkpoint并迁移权重 ==========
    if os.path.exists(sft_checkpoint_path):
        print(f"\n🔄 加载SFT checkpoint: {sft_checkpoint_path}")
        sft_ckpt = torch.load(sft_checkpoint_path, map_location=device, weights_only=False)
        
        # 创建Policy
        policy = GraphPolicy(
            node_dim=64,
            hidden_size=128,
            num_layers=2,
            encoder_type="gat"
        ).to(device)
        
        # 迁移graph_encoder权重
        sft_state = sft_ckpt["model_state_dict"]
        graph_encoder_state = {
            k.replace("graph_encoder.", ""): v 
            for k, v in sft_state.items() 
            if k.startswith("graph_encoder.")
        }
        policy.graph_encoder.load_state_dict(graph_encoder_state, strict=False)
        print(" 成功迁移SFT的graph_encoder权重")
    else:
        print(f"⚠️ 未找到SFT checkpoint，从零开始训练")
        policy = GraphPolicy(
            node_dim=64,
            hidden_size=128,
            num_layers=2,
            encoder_type="gat"
        ).to(device)
    
    # ========== 2. 创建图任务 ==========
    print("\n📊 创建图任务...")
    task = build_path_task()
    graph = task["graph"]
    num_nodes = graph.x.size(0)
    adj = build_adj_list(graph.edge_index, num_nodes)
    
    print(f"  - 节点数: {num_nodes}")
    print(f"  - 边数: {graph.edge_index.size(1)}")
    print(f"  - 起点: {task['start']} → 终点: {task['goal']}")
    print(f"  - 标准路径: {task['answer']}")
    
    # ========== 3. 优化器 ==========
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # ========== 4. 混合精度 ==========
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
    
    # ========== 5. PPO训练循环 ==========
    print(f"\n{'='*70}")
    print(" 开始PPO训练")
    print(f"{'='*70}\n")
    
    best_reward = -float('inf')
    success_count = 0
    
    for episode in range(num_episodes):
        current = task["start"]
        goal = task["goal"]
        
        # 收集轨迹
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        visited = set([current])
        path_nodes = [current]
        
        # === Episode采样 ===
        policy.eval()
        with torch.no_grad():
            for step in range(max_steps_per_episode):
                # 计算策略和价值
                with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    logits = policy(graph.x, graph.edge_index, current, goal)
                    value = policy.get_value(graph.x, graph.edge_index, current, goal) if hasattr(policy, 'get_value') else 0.0
                
                # Mask非邻居节点
                mask = torch.full_like(logits, float("-inf"))
                for n in adj[current]:
                    mask[n] = 0.0
                masked_logits = logits + mask
                
                # 采样动作
                probs = torch.softmax(masked_logits, dim=-1)
                next_node = torch.multinomial(probs, num_samples=1).item()
                old_log_prob = torch.log(probs[next_node] + 1e-9)
                
                # 计算奖励
                reward = -0.01  # 步数惩罚
                if next_node == goal:
                    reward += 10.0  # 到达目标
                if next_node in visited:
                    reward -= 0.1   # 重复访问惩罚
                visited.add(next_node)
                
                # 保存
                states.append((current, goal))
                actions.append(next_node)
                old_log_probs.append(old_log_prob.detach())
                rewards.append(reward)
                values.append(value if isinstance(value, float) else value.item())
                
                current = next_node
                path_nodes.append(current)
                
                if current == goal:
                    success_count += 1
                    break
        
        # === 计算优势函数 (GAE) ===
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.stack(old_log_probs).to(device)
        
        # 优势函数归一化 (重要！稳定训练)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === PPO更新 ===
        policy.train()
        for ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad()
            
            # 重新计算log_probs
            new_log_probs = []
            new_values = []
            entropies = []
            
            with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                for (s_current, s_goal), action in zip(states, actions):
                    logits = policy(graph.x, graph.edge_index, s_current, s_goal)
                    value = policy.get_value(graph.x, graph.edge_index, s_current, s_goal) if hasattr(policy, 'get_value') else torch.tensor(0.0, device=device)
                    
                    mask = torch.full_like(logits, float("-inf"))
                    for n in adj[s_current]:
                        mask[n] = 0.0
                    masked_logits = logits + mask
                    
                    probs = torch.softmax(masked_logits, dim=-1)
                    new_log_probs.append(torch.log(probs[action] + 1e-9))
                    new_values.append(value)
                    
                    # 熵正则化 (鼓励探索)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum()
                    entropies.append(entropy)
                
                new_log_probs_tensor = torch.stack(new_log_probs)
                new_values_tensor = torch.stack(new_values) if len(new_values) > 0 else torch.zeros_like(returns)
                entropy_tensor = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)
                
                # PPO Clipped Objective
                ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (MSE)
                value_loss = F.mse_loss(new_values_tensor.squeeze(), returns)
                
                # 总Loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_tensor
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
        
        # === 日志 ===
        total_reward = sum(rewards)
        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % 20 == 0:
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} (Best: {best_reward:.2f}) | "
                  f"Steps: {len(path_nodes)} | "
                  f"Success: {success_rate:.1f}% | "
                  f"Path: {path_nodes[:5]}{'...' if len(path_nodes) > 5 else ''}")
    
    # ========== 6. 保存模型 ==========
    Path(output_path).parent.mkdir(exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'training_args': {
            'num_episodes': num_episodes,
            'best_reward': best_reward,
            'success_rate': success_count / num_episodes,
        }
    }, output_path)
    
    print(f"\n{'='*70}")
    print(f" PPO训练完成: {output_path}")
    print(f"  - 成功率: {success_count/num_episodes*100:.1f}%")
    print(f"  - 最佳奖励: {best_reward:.2f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train_graph_ppo()