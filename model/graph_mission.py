import torch
from torch_geometric.data import Data

def build_path_task():
    """
    图推理任务：最短路径
    图结构：
        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
    """
    edge_index = torch.tensor([
        [0,1, 1,2, 0,3, 1,4, 2,5, 3,4, 4,5],
        [1,0, 2,1, 3,0, 4,1, 5,2, 4,3, 5,4]
    ], dtype=torch.long)

    node_features = torch.randn(6, 64)  # 节点特征（可替换成更真实的特征）

    graph = Data(x=node_features, edge_index=edge_index)

    task = {
        "graph": graph,
        "question": "从节点0到节点5的最短路径？",
        "answer": "0 -> 1 -> 4 -> 5",
        "reasoning_steps": [
            "从0开始",
            "邻居是1和3",
            "选择1",
            "从1的邻居是0,2,4",
            "选择4",
            "从4的邻居是1,3,5",
            "到达5"
        ],
        "start": 0,       # ✅ 起点
        "goal": 5,        # ✅ 终点
        "max_steps": 6    # ✅ 最多允许走几步（防止死循环）
    }
    return task