"""
测试Graph Encoder是否正常工作
"""

import torch
from torch_geometric.data import Data
from model.graph_encoder import GraphEncoder

def test_graph_encoder():
    print("=== 测试Graph Encoder ===\n")
    
    # 1. 创建一个简单的图
    # 图结构：
    #   0 -- 1 -- 2
    #   |         |
    #   3 ------- 4
    
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 2, 4, 3, 4],  # 源节点
        [1, 0, 2, 1, 3, 0, 4, 2, 4, 3]   # 目标节点
    ], dtype=torch.long)
    
    # 节点特征（5个节点，每个64维）
    node_features = torch.randn(5, 64)
    
    print(f"图信息：")
    print(f"  节点数: {node_features.size(0)}")
    print(f"  边数: {edge_index.size(1) // 2}")  # 无向图，每条边算两次
    print(f"  节点特征维度: {node_features.size(1)}\n")
    
    # 2. 创建Graph Encoder
    encoder = GraphEncoder(
        node_dim=64,
        hidden_size=128,  # 和Llama对齐
        num_layers=2,
        encoder_type='gat'  # 使用GAT
    )
    
    # 3. 编码图
    node_emb, graph_emb = encoder(node_features, edge_index)
    
    print(f"编码结果：")
    print(f"  节点embeddings: {node_emb.shape}")  # 应该是 [5, 128]
    print(f"  图embedding: {graph_emb.shape}")    # 应该是 [128]
    print(f"\n节点embedding示例（节点0的前10维）:")
    print(f"  {node_emb[0, :10]}\n")
    
    # 4. 测试batch处理
    print("=== 测试Batch处理 ===\n")
    
    # 创建两个图的batch
    # 图1: 3个节点
    # 图2: 2个节点
    batch_node_features = torch.randn(5, 64)
    batch_edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4],  # 0-1, 1-2 (图1), 3-4 (图2)
        [1, 0, 2, 1, 4, 3]
    ])
    batch_indicator = torch.tensor([0, 0, 0, 1, 1])  # 前3个节点属于图1，后2个属于图2
    
    node_emb, graph_emb = encoder(
        batch_node_features, 
        batch_edge_index, 
        batch=batch_indicator
    )
    
    print(f"Batch编码结果：")
    print(f"  节点embeddings: {node_emb.shape}")  # [5, 128]
    print(f"  图embeddings: {graph_emb.shape}")   # [2, 128] - 两个图
    print(f"\n图1的embedding（前10维）:")
    print(f"  {graph_emb[0, :10]}")
    print(f"\n图2的embedding（前10维）:")
    print(f"  {graph_emb[1, :10]}")

if __name__ == "__main__":
    test_graph_encoder()