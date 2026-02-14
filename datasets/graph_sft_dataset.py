"""
Graph+Text SFT数据集（适配你的模型）
"""
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Union

class GraphSFTDataset(Dataset):
    """
    Graph+Text SFT数据集
    
    数据格式：
    {
        "prompt": "问题",
        "completion": "答案",
        "node_features": [[...], ...],
        "edge_index": [[...], [...]]
    }
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 512,
        mask_prompt: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.data = []
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
        
        self._load_jsonl(data_path)
        print(f"✅ Graph SFT数据集：{len(self.data)} 条样本")
    
    def _load_jsonl(self, file_path: Path):
        required_fields = {"prompt", "completion", "node_features", "edge_index"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        missing = required_fields - set(item.keys())
                        if missing:
                            print(f"⚠️ 第{line_num}行缺少字段: {missing}")
                            continue
                        self.data.append(item)
                    except json.JSONDecodeError:
                        print(f"⚠️ 第{line_num}行JSON格式错误")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        ✅ 返回字典（适配GraphLlamaForCausalLM）
        """
        item = self.data[idx]
        
        # 1. 文本处理
        prompt = item["prompt"]
        completion = item["completion"]
        
        # ✅ 使用你的tokenizer.encode()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        
        input_ids = prompt_ids + completion_ids
        
        # Labels（mask prompt）
        if self.mask_prompt:
            labels = [-100] * len(prompt_ids) + completion_ids
        else:
            labels = input_ids.copy()
        
        # 截断和padding
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        # 2. 图数据
        node_features = torch.tensor(item["node_features"], dtype=torch.float)
        edge_index = torch.tensor(item["edge_index"], dtype=torch.long)
        
        # ✅ 返回字典（适配你的GraphLlamaForCausalLM.forward()）
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "node_features": node_features,
            "edge_index": edge_index
        }