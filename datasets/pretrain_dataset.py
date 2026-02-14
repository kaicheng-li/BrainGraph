"""
预训练数据集（适配你的Tokenizer）
兼容：TiktokenWrapper 和 HFTokenizerWrapper
"""
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Union

class PretrainDataset(Dataset):
    """
    预训练数据集
    
    兼容你的两种tokenizer：
    1. TiktokenWrapper (默认)
    2. HFTokenizerWrapper
    
    数据格式：{"text": "..."}
    """
    
    def __init__(
        self, 
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        data_path = Path(data_path)
        if data_path.is_file():
            self._load_jsonl(data_path)
        elif data_path.is_dir():
            for jsonl_file in data_path.glob("*.jsonl"):
                self._load_jsonl(jsonl_file)
        else:
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
        
        print(f"✅ 预训练数据集：{len(self.data)} 条样本")
    
    def _load_jsonl(self, file_path: Path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        if 'text' in item and item['text'].strip():
                            self.data.append(item['text'])
                    except json.JSONDecodeError:
                        print(f"⚠️ {file_path.name} 第{line_num}行JSON格式错误")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        ✅ 适配你的tokenizer.encode()接口
        返回：(input_ids, labels) - 与MiniMind格式一致
        """
        text = self.data[idx]
        
        # ✅ 使用你的tokenizer.encode()
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 截断
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Padding
        labels = token_ids.copy()
        if len(token_ids) < self.max_length:
            pad_len = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        # ✅ 返回元组（与MiniMind格式一致）
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return input_ids, labels