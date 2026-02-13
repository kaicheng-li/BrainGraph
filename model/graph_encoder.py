"""
Graph Encoder Module

论文参考：
1. GCN: Kipf & Welling (ICLR 2017)
2. GAT: Veličković et al. (ICLR 2018)
3. GreaseLM: Zhang et al. (ICLR 2022) - Graph-LM融合架构
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from typing import Optional

# ============================================================
# Part 1: 基础的GCN编码器（Kipf & Welling, ICLR 2017）
# ============================================================

class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network编码器
    
    原理（GCN论文核心公式）：
        H^(l+1) = σ(D^(-1/2) A_hat D^(-1/2) H^(l) W^(l))
    
    其中：
        - H^(l): 第l层的节点特征 [num_nodes, hidden_size]
        - A_hat: 邻接矩阵（加了自环）
        - D: 度矩阵（对角线是每个节点的度数）
        - W^(l): 可学习的权重矩阵
        - σ: 激活函数（ReLU）
    
    直观理解：
        每个节点的新表示 = 自己 + 邻居们的加权平均
    """
    
    def __init__(
        self, 
        node_dim: int,          # 输入节点特征维度
        hidden_size: int,       # 隐藏层维度（要和Llama的hidden_size对齐）
        num_layers: int = 3,    # GCN层数（论文推荐2-3层）
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入投影：把原始节点特征投影到hidden_size
        self.input_proj = nn.Linear(node_dim, hidden_size)
        
        # GCN层堆叠
        # 论文发现：2-3层效果最好，太深会导致over-smoothing（所有节点变得相似）
        self.convs = nn.ModuleList([
            GCNConv(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # 每层后面加Layer Norm（稳定训练）
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,              # 节点特征 [num_nodes, node_dim]
        edge_index: torch.Tensor,     # 边索引 [2, num_edges]
        batch: Optional[torch.Tensor] = None  # batch索引 [num_nodes]
    ):
        """
        前向传播
        
        参数说明：
            x: 节点特征矩阵
                例如：[100, 64] 表示100个节点，每个节点64维特征
            
            edge_index: 边的连接关系（COO格式）
                例如：[[0, 1, 2],    # 源节点
                      [1, 2, 0]]    # 目标节点
                表示：0→1, 1→2, 2→0 三条边
            
            batch: 每个节点属于哪个图（batch处理时用）
                例如：[0,0,0,1,1,1] 表示前3个节点属于图0，后3个属于图1
        """
        
        # Step 1: 投影到hidden_size
        h = self.input_proj(x)  # [num_nodes, hidden_size]
        
        # Step 2: 逐层GCN传播
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GCN更新
            h_new = conv(h, edge_index)  # [num_nodes, hidden_size]
            
            # 激活函数
            h_new = torch.relu(h_new)
            
            # Dropout（防止过拟合）
            h_new = self.dropout(h_new)
            
            # 残差连接（帮助深层网络训练）
            h = norm(h + h_new)  # 残差 + Layer Norm
        
        node_embeddings = h  # [num_nodes, hidden_size]
        
        # Step 3: 全局池化（把所有节点聚合成图级表示）
        if batch is None:
            # 单图模式：直接平均
            graph_embedding = h.mean(dim=0)  # [hidden_size]
        else:
            # 批处理模式：按batch分组平均
            graph_embedding = global_mean_pool(h, batch)  # [batch_size, hidden_size]
        
        return node_embeddings, graph_embedding


# ============================================================
# Part 2: GAT编码器（Veličković et al., ICLR 2018）
# ============================================================

class GATEncoder(nn.Module):
    """
    Graph Attention Network编码器
    
    原理（GAT论文核心）：
        α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
        h_i' = σ(Σ_j α_ij W h_j)
    
    其中：
        - α_ij: 节点i对邻居j的注意力权重（动态计算）
        - W: 线性变换矩阵
        - a: 注意力参数向量
        - ||: 拼接操作
    
    优势：
        相比GCN，GAT能自动学习邻居的重要性
        比如：论文中的社交网络，好友的重要性不同
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_size: int,
        num_layers: int = 3,
        num_heads: int = 4,      # 多头注意力（类似Transformer）
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 输入投影
        self.input_proj = nn.Linear(node_dim, hidden_size)
        
        # GAT层
        # 注意：多头注意力，所以每个头输出 hidden_size // num_heads
        self.convs = nn.ModuleList([
            GATConv(
                hidden_size, 
                hidden_size // num_heads,
                heads=num_heads,       # 多头数量
                dropout=dropout,
                concat=True  # 最后一层average而不是concat
            )
            for i in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        # 投影
        h = self.input_proj(x)
        
        # 逐层GAT传播
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            
            # 残差连接
            h = norm(h + h_new)
        
        node_embeddings = h
        
        # 全局池化
        if batch is None:
            graph_embedding = h.mean(dim=0)
        else:
            graph_embedding = global_mean_pool(h, batch)
        
        return node_embeddings, graph_embedding


# ============================================================
# Part 3: 统一的Graph Encoder接口
# ============================================================

class GraphEncoder(nn.Module):
    """
    统一的图编码器接口
    
    支持两种backbone：
    - GCN (更快，适合大图)
    - GAT (更强，适合复杂关系)
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_size: int,
        num_layers: int = 3,
        encoder_type: str = 'gat',  # 'gcn' or 'gat'
        **kwargs
    ):
        super().__init__()
        
        if encoder_type == 'gcn':
            self.encoder = GCNEncoder(
                node_dim=node_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                **kwargs
            )
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(
                node_dim=node_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, x, edge_index, batch=None):
        return self.encoder(x, edge_index, batch)