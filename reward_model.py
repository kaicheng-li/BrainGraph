"""
Reward Function for Graph Reasoning

理论参考：
- Sutton & Barto, Reinforcement Learning (1998) — REINFORCE
- GreaseLM (ICLR 2022) — 图推理评价思想
"""

def path_reward(pred_path: str, gold_path: str) -> float:
    """
    简单奖励：
    - 完全正确：+1.0
    - 部分正确：按节点匹配比例给分
    """
    pred = [x.strip() for x in pred_path.split("->")]
    gold = [x.strip() for x in gold_path.split("->")]

    if pred == gold:
        return 1.0

    # 部分匹配：重合节点比例
    overlap = len(set(pred) & set(gold))
    return overlap / max(1, len(gold))