"""
完整评估脚本：测试SFT+RL的效果

评估指标：
1. 最短路径准确率（Exact Match）
2. 路径长度与最优解的比值（Path Efficiency）
3. 到达目标的成功率（Success Rate）
4. 平均步数（Average Steps）
"""
import torch
import torch.nn.functional as F
from model.graph_policy import GraphPolicy
from model.graph_mission import build_path_task
from model.graph_utils import build_adj_list
from model.llama_model import LlamaForCausalLM
from model.llama_config import LlamaConfig
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

# ============================================
# 测试任务生成器
# ============================================

def create_test_graphs():
    """创建多个测试图任务"""
    import torch_geometric
    from torch_geometric.data import Data
    
    test_cases = []
    
    # Case 1: 简单链状图
    edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)
    x = torch.randn(5, 64)
    test_cases.append({
        "name": "链状图(5节点)",
        "graph": Data(x=x, edge_index=edge_index),
        "start": 0,
        "goal": 4,
        "gold_path": "0 -> 1 -> 2 -> 3 -> 4",
        "optimal_length": 4
    })
    
    # Case 2: 有捷径的图
    edge_index = torch.tensor([
        [0,1,1,2,0],
        [1,2,4,3,4]
    ], dtype=torch.long)
    x = torch.randn(5, 64)
    test_cases.append({
        "name": "捷径图(5节点)",
        "graph": Data(x=x, edge_index=edge_index),
        "start": 0,
        "goal": 4,
        "gold_path": "0 -> 4",  # 直达最短
        "optimal_length": 1
    })
    
    # Case 3: 稍复杂的图
    edge_index = torch.tensor([
        [0,0,1,1,2,3,4,5],
        [1,2,3,4,5,4,5,6]
    ], dtype=torch.long)
    x = torch.randn(7, 64)
    test_cases.append({
        "name": "分支图(7节点)",
        "graph": Data(x=x, edge_index=edge_index),
        "start": 0,
        "goal": 6,
        "gold_path": "0 -> 1 -> 4 -> 5 -> 6",
        "optimal_length": 4
    })
    
    return test_cases

# ============================================
# 评估函数
# ============================================

def evaluate_policy(
    policy: GraphPolicy,
    test_cases: List[Dict],
    max_steps: int = 20,
    num_trials: int = 10
) -> Dict:
    """
    评估策略在多个测试案例上的表现
    
    返回：
    - exact_match: 完全匹配最优路径的比例
    - success_rate: 到达目标的比例
    - avg_steps: 平均步数
    - efficiency: 路径效率（最优长度/实际长度）
    """
    policy.eval()
    device = next(policy.parameters()).device
    
    results = {
        "exact_matches": [],
        "successes": [],
        "steps": [],
        "efficiencies": []
    }
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        graph = case["graph"]
        start = case["start"]
        goal = case["goal"]
        optimal_length = case["optimal_length"]
        gold_path = [int(x.strip()) for x in case["gold_path"].split("->")]
        
        num_nodes = graph.x.size(0)
        adj = build_adj_list(graph.edge_index, num_nodes)
        
        case_exact = 0
        case_success = 0
        case_steps = []
        case_eff = []
        
        # 多次试验
        for trial in range(num_trials):
            current = start
            visited = set([current])
            path = [current]
            
            for step in range(max_steps):
                with torch.no_grad():
                    logits = policy(graph.x, graph.edge_index, current, goal)
                    
                    # Mask
                    mask = torch.full_like(logits, float("-inf"))
                    for n in adj[current]:
                        mask[n] = 0.0
                    masked_logits = logits + mask
                    
                    # Greedy选择（评估时不采样）
                    next_node = torch.argmax(masked_logits).item()
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
                    if current == goal:
                        break
            
            # 统计
            success = (current == goal)
            exact = (path == gold_path)
            steps = len(path) - 1  # 边数
            efficiency = optimal_length / max(steps, 1) if success else 0
            
            if exact:
                case_exact += 1
            if success:
                case_success += 1
            case_steps.append(steps)
            case_eff.append(efficiency)
        
        # 汇总
        results["exact_matches"].append(case_exact / num_trials)
        results["successes"].append(case_success / num_trials)
        results["steps"].append(np.mean(case_steps))
        results["efficiencies"].append(np.mean(case_eff))
        
        print(f"  精确匹配率: {100*case_exact/num_trials:.1f}%")
        print(f"  成功率: {100*case_success/num_trials:.1f}%")
        print(f"  平均步数: {np.mean(case_steps):.2f} (最优: {optimal_length})")
        print(f"  路径效率: {100*np.mean(case_eff):.1f}%")
    
    # 总体指标
    overall = {
        "avg_exact_match": np.mean(results["exact_matches"]),
        "avg_success_rate": np.mean(results["successes"]),
        "avg_steps": np.mean(results["steps"]),
        "avg_efficiency": np.mean(results["efficiencies"])
    }
    
    return results, overall

# ============================================
# 可视化
# ============================================

def plot_training_comparison(baseline_rewards, rl_rewards, save_path="training_curve.png"):
    """对比baseline和RL的训练曲线"""
    plt.figure(figsize=(10, 6))
    
    episodes_baseline = np.arange(len(baseline_rewards))
    episodes_rl = np.arange(len(rl_rewards))
    
    plt.plot(episodes_baseline, baseline_rewards, label='Baseline (Random)', alpha=0.6)
    plt.plot(episodes_rl, rl_rewards, label='PPO Training', alpha=0.8, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Progress: Baseline vs PPO', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存: {save_path}")
    plt.close()

def plot_evaluation_results(results, case_names, save_path="eval_results.png"):
    """可视化评估结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.arange(len(case_names))
    width = 0.6
    
    # 1. 精确匹配率
    axes[0, 0].bar(x, [r*100 for r in results["exact_matches"]], width, color='steelblue')
    axes[0, 0].set_ylabel('Exact Match (%)', fontsize=11)
    axes[0, 0].set_title('精确匹配率', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(case_names, rotation=15, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 成功率
    axes[0, 1].bar(x, [r*100 for r in results["successes"]], width, color='seagreen')
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=11)
    axes[0, 1].set_title('到达目标成功率', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(case_names, rotation=15, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 平均步数
    axes[1, 0].bar(x, results["steps"], width, color='coral')
    axes[1, 0].set_ylabel('Average Steps', fontsize=11)
    axes[1, 0].set_title('平均步数', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(case_names, rotation=15, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. 路径效率
    axes[1, 1].bar(x, [r*100 for r in results["efficiencies"]], width, color='mediumpurple')
    axes[1, 1].set_ylabel('Path Efficiency (%)', fontsize=11)
    axes[1, 1].set_title('路径效率', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(case_names, rotation=15, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"评估结果已保存: {save_path}")
    plt.close()

# ============================================
# 主评估流程
# ============================================

def main():
    print("\n" + "="*60)
    print("📊 模型评估 - Graph RL Policy")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载训练好的策略
    checkpoint_path = "./checkpoints/final_policy.pt"
    if os.path.exists(checkpoint_path):
        print(f"\n加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        policy = GraphPolicy(
            node_dim=64,
            hidden_size=128,
            num_layers=2,
            encoder_type="gat"
        ).to(device)
        
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print("✓ 模型加载成功")
    else:
        print(f"\n警告: 未找到checkpoint {checkpoint_path}")
        print("使用随机初始化的策略进行演示...")
        policy = GraphPolicy(
            node_dim=64,
            hidden_size=128,
            num_layers=2,
            encoder_type="gat"
        ).to(device)
    
    # 2. 创建测试集
    test_cases = create_test_graphs()
    print(f"\n测试案例数: {len(test_cases)}")
    
    # 3. 评估
    print("\n" + "="*60)
    print("开始评估...")
    print("="*60)
    
    results, overall = evaluate_policy(
        policy=policy,
        test_cases=test_cases,
        max_steps=20,
        num_trials=10
    )
    
    # 4. 总体结果
    print("\n" + "="*60)
    print("总体评估结果:")
    print("="*60)
    print(f"平均精确匹配率: {100*overall['avg_exact_match']:.1f}%")
    print(f"平均成功率: {100*overall['avg_success_rate']:.1f}%")
    print(f"平均步数: {overall['avg_steps']:.2f}")
    print(f"平均路径效率: {100*overall['avg_efficiency']:.1f}%")
    
    # 5. 可视化
    os.makedirs("./results", exist_ok=True)
    case_names = [case["name"] for case in test_cases]
    plot_evaluation_results(results, case_names, "./results/eval_results.png")
    
    print("\n✅ 评估完成！结果已保存到 ./results/ 目录")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()