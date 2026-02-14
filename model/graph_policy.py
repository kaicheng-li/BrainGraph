import torch
import torch.nn as nn
from .graph_encoder import GraphEncoder

class GraphPolicy(nn.Module):
    """
    图策略网络：
    - 先用GraphEncoder得到每个节点的向量
    - 用“当前节点向量 + 目标节点向量”作为状态
    - 输出对所有节点的logits（表示下一步去哪）
    """

    def __init__(self, node_dim=64, hidden_size=128, num_layers=2, encoder_type="gat"):
        super().__init__()
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            encoder_type=encoder_type
        )
        # 把“当前节点 + 目标节点”的拼接向量投影
        self.policy_head = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, node_features, edge_index, current_node, goal_node):
        """
        返回对所有节点的logits
        """
        node_emb, _ = self.encoder(node_features, edge_index)  # [N, H]

        cur_emb = node_emb[current_node]   # 当前节点表示 [H]
        goal_emb = node_emb[goal_node]     # 目标节点表示 [H]

        state = torch.cat([cur_emb, goal_emb], dim=-1)  # [2H]
        state = self.policy_head(state)  # [H]

        # 与每个节点向量做点积，得到对所有节点的“倾向分数”
        logits = torch.matmul(node_emb, state)  # [N]
        return logits