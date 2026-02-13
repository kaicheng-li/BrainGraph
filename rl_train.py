import torch
import torch.nn.functional as F
from model.graph_mission import build_path_task
import torch
from model.graph_mission import build_path_task
from model.graph_policy import GraphPolicy
from model.graph_utils import build_adj_list

# 在main()之前添加

def build_shortest_path_reward(path_nodes, gold_path_str):
    """
    对齐最短路径的额外奖励
    path_nodes: 实际走过的节点列表，如 [0, 1, 4, 5]
    gold_path_str: 标准答案 "0 -> 1 -> 4 -> 5"
    """
    gold = [int(x.strip()) for x in gold_path_str.split("->")]
    
    # 完全匹配
    if path_nodes == gold:
        return 0.5
    
    # 部分匹配：按重合比例给分
    overlap = len(set(path_nodes) & set(gold))
    return 0.5 * overlap / max(1, len(gold))

def main():
    # 1) 任务 + 图
    task = build_path_task()
    graph = task["graph"]

    num_nodes = graph.x.size(0)
    adj = build_adj_list(graph.edge_index, num_nodes)

    # 2) 策略网络
    policy = GraphPolicy(node_dim=64, hidden_size=128, num_layers=2, encoder_type="gat")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    
    #新增PPO#
    clip_epsilon = 0.2 # 裁剪范围
    ppo_epochs = 4  # 每条轨迹重复训练次数

    # 3) 训练循环
    for episode in range(200):
        current = task["start"]
        goal = task["goal"]
        max_steps = task["max_steps"]
        
        #额外补充收集轨迹
        states = []        # 保存(current, goal)
        actions = []       # 保存选择的节点
        old_log_probs = [] # 保存旧策略的log概率
        rewards = []
        visited = set([current])
        path_nodes = [current]

        for step in range(max_steps):
            # 3.1 计算策略logits
            logits = policy(graph.x, graph.edge_index, current, goal)

            # 3.2 只允许走邻居
            mask = torch.full_like(logits, float("-inf"))
            for n in adj[current]:
                mask[n] = 0.0
            masked_logits = logits + mask

            # 3.3 采样动作
            probs = torch.softmax(masked_logits, dim=-1)
            next_node = torch.multinomial(probs, num_samples=1).item()
            old_log_prob = torch.log(probs[next_node] + 1e-9)

            # 3.4 奖励规则
            reward = -0.01            # 每走一步小惩罚
            if next_node == goal:
                reward += 1.0         # 到达终点大奖励
            if next_node in visited:
                reward -= 0.05        # 重复节点惩罚
            visited.add(next_node)

            # 保存
            states.append((current, goal))
            actions.append(next_node)
            old_log_probs.append(old_log_prob.detach())  # detach很重要！
            rewards.append(reward)

            current = next_node
            path_nodes.append(current)

            if current == goal:
                break

        # 4) 额外的最短路径奖励
        extra_reward = build_shortest_path_reward(path_nodes, task["answer"])
        rewards = [r + extra_reward/len(rewards) for r in rewards]
        advantages = torch.tensor(rewards, dtype=torch.float32)

        # 5) REINFORCE更新改为PPO更新
        #loss = 0.0
        #for log_prob, r in zip(old_log_probs, rewards):
        #    loss += -log_prob * r
        
        for _ in range(ppo_epochs):
            total_loss = 0.0

            for i, ((cur, g), action, old_log_prob, adv) in enumerate(
                zip(states, actions, old_log_probs, advantages)
            ):
                # 重新计算logits
                logits = policy(graph.x, graph.edge_index, cur, g)

                mask = torch.full_like(logits, float("-inf"))
                for n in adj[cur]:
                    mask[n] = 0.0
                masked_logits = logits + mask

                probs = torch.softmax(masked_logits, dim=-1)
                new_log_prob = torch.log(probs[action] + 1e-9)

                # === PPO核心：计算ratio并裁剪 ===
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
                
                # 取min（保守更新）
                loss = -torch.min(ratio * adv, clipped_ratio * adv)
                total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 20 == 0:
            path_str = " -> ".join(map(str, path_nodes))
            print(f"episode {episode} | steps={len(rewards)} | path={path_str} | reward={sum(rewards):.3f}")
            
if __name__ == "__main__":
    main()