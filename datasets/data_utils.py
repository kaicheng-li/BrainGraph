"""
数据处理通用工具
"""
import torch
from torch_geometric.data import Batch
from typing import List, Dict

def collate_graph_batch(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，用于Graph+Text数据
    
    输入：List of dicts，每个dict包含 input_ids, labels, node_features, edge_index
    输出：Batched dict
    """
    # 文本部分直接stack
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # 图部分需要特殊处理（不同图的节点数可能不同）
    node_features_list = [item["node_features"] for item in batch]
    edge_index_list = [item["edge_index"] for item in batch]
    
    # 创建batch索引
    batch_idx = []
    for i, node_feat in enumerate(node_features_list):
        batch_idx.extend([i] * node_feat.size(0))
    
    # 拼接所有图
    all_node_features = torch.cat(node_features_list, dim=0)
    
    # 边索引需要偏移
    offset = 0
    all_edge_index = []
    for edge_idx in edge_index_list:
        all_edge_index.append(edge_idx + offset)
        offset += node_features_list[len(all_edge_index) - 1].size(0)
    all_edge_index = torch.cat(all_edge_index, dim=1)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_features": all_node_features,
        "edge_index": all_edge_index,
        "batch": torch.tensor(batch_idx, dtype=torch.long)
    }


def collate_pyg_batch(batch: List) -> Dict:
    """
    使用PyG的Batch自动处理（推荐）
    适用于GraphSFTDatasetWithBatch
    """
    # PyG自动处理图的batching
    batched_graph = Batch.from_data_list(batch)
    
    return {
        "input_ids": batched_graph.input_ids,
        "labels": batched_graph.labels,
        "node_features": batched_graph.x,
        "edge_index": batched_graph.edge_index,
        "batch": batched_graph.batch
    }


def prepare_tokenized_batch(texts: List[str], tokenizer, max_length: int = 512) -> Dict:
    """
    快速批量tokenize文本
    
    参数：
        texts: 文本列表
        tokenizer: Tokenizer对象
        max_length: 最大长度
    
    返回：
        {
            "input_ids": [batch, seq_len],
            "labels": [batch, seq_len]
        }
    """
    batch_input_ids = []
    batch_labels = []
    
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        labels = token_ids.copy()
        if len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        batch_input_ids.append(token_ids)
        batch_labels.append(labels)
    
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long)
    }


def create_sample_graph_data(save_path: str = "data/graph_sft/sample.jsonl"):
    """
    创建示例Graph SFT数据（用于测试）
    """
    import json
    from pathlib import Path
    
    # 创建目录
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    samples = [
        {
            "prompt": "图结构：节点0连接到节点1，节点1连接到节点2。请问从节点0到节点2的路径是什么？",
            "completion": "从节点0到节点2的路径是: 0 -> 1 -> 2",
            "node_features": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            "edge_index": [[0, 1], [1, 2]]
        },
        {
            "prompt": "给定图：边列表 [(0,1), (1,2), (2,3)]。节点0到节点3有几条边？",
            "completion": "从节点0到节点3需要经过3条边",
            "node_features": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "edge_index": [[0, 1, 2], [1, 2, 3]]
        }
    ]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f" 示例数据已创建: {save_path}")


if __name__ == "__main__":
    # 测试：创建示例数据
    create_sample_graph_data()