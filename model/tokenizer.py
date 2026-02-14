# model/tokenizer.py
"""
标准Tokenizer实现
使用现成的工具：
1. tiktoken (OpenAI的BPE，速度最快) - 推荐
2. transformers.AutoTokenizer (HuggingFace)
3. sentencepiece (Google)
"""
import torch
from typing import List, Dict, Optional, Union

# ============================================
# 方案1: 使用tiktoken (推荐，最快)
# ============================================

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken未安装，使用备选方案")

class TiktokenWrapper:
    """
    基于tiktoken的Tokenizer包装
    OpenAI GPT系列使用的BPE
    
    优点：
    - C++实现，速度极快
    - 支持多语言（包括中文）
    - 直接兼容GPT-2/GPT-3/GPT-4
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        encoding_name选项：
        - "gpt2": GPT-2/GPT-3
        - "cl100k_base": GPT-3.5/GPT-4 (推荐)
        - "p50k_base": Code models
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("需要安装: pip install tiktoken")
        
        self.enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        
        # 特殊token
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|startoftext|>"
        
        # 添加特殊token（如果不存在）
        self.special_tokens = {
            self.pad_token: self.enc.n_vocab,
            self.bos_token: self.enc.n_vocab + 1,
        }
        
        print(f"✓ Tiktoken加载: {encoding_name}, vocab_size={self.vocab_size}")
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """编码文本"""
        ids = self.enc.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """解码token ids"""
        # 过滤特殊token
        ids = [i for i in ids if i < self.enc.n_vocab]
        return self.enc.decode(ids)
    
    def encode_batch(self, texts: List[str], max_length: int = 512,
                     padding: bool = True, truncation: bool = True,
                     add_special_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """批量编码"""
        all_input_ids = []
        all_attention_mask = []
        
        for text in texts:
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            
            # Truncation
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            
            # Attention mask
            attention_mask = [1] * len(ids)
            
            # Padding
            if padding and len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
            
            all_input_ids.append(ids)
            all_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long)
        }
    
    @property
    def vocab_size(self) -> int:
        """词表大小（包含特殊token）"""
        return self.enc.n_vocab + len(self.special_tokens)
    
    @property
    def pad_token_id(self) -> int:
        return self.special_tokens[self.pad_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.enc.eot_token
    
    @property
    def bos_token_id(self) -> int:
        return self.special_tokens[self.bos_token]


# ============================================
# 方案2: 使用HuggingFace Tokenizer
# ============================================

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class HFTokenizerWrapper:
    """
    HuggingFace Tokenizer包装
    可以直接使用Qwen、GPT等预训练tokenizer
    """
    def __init__(self, model_name: str = "Qwen/Qwen-7B"):
        """
        model_name选项：
        - "Qwen/Qwen-7B": Qwen tokenizer
        - "gpt2": GPT-2
        - "facebook/opt-125m": OPT
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装: pip install transformers")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 添加pad token（如果没有）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ HF Tokenizer加载: {model_name}, vocab_size={self.vocab_size}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, ids: List[int]) -> str:
        """解码"""
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    def encode_batch(self, texts: List[str], max_length: int = 512,
                     padding: bool = True, truncation: bool = True,
                     add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """批量编码"""
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length' if padding else False,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors='pt'
        )
        return encoded
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id


# ============================================
# 统一接口
# ============================================

def get_tokenizer(tokenizer_type: str = "tiktoken", **kwargs):
    """
    获取tokenizer
    
    参数:
    - tokenizer_type: "tiktoken" | "huggingface" | "qwen"
    - kwargs: 传递给具体tokenizer的参数
    
    示例:
    >>> tokenizer = get_tokenizer("tiktoken", encoding_name="cl100k_base")
    >>> tokenizer = get_tokenizer("qwen")
    """
    if tokenizer_type == "tiktoken":
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("pip install tiktoken")
        encoding_name = kwargs.get("encoding_name", "cl100k_base")
        return TiktokenWrapper(encoding_name)
    
    elif tokenizer_type in ["huggingface", "hf", "qwen"]:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("pip install transformers")
        if tokenizer_type == "qwen":
            model_name = kwargs.get("model_name", "Qwen/Qwen-7B")
        else:
            model_name = kwargs.get("model_name", "gpt2")
        return HFTokenizerWrapper(model_name)
    
    else:
        raise ValueError(f"未知tokenizer类型: {tokenizer_type}")


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("标准Tokenizer测试")
    print("="*60)
    
    # 测试文本
    text_zh = "图结构: 节点0→1→4→5, 从0到5的最短路径是什么?"
    text_en = "Graph: nodes 0->1->4->5. What is the shortest path?"
    
    # 测试tiktoken
    if TIKTOKEN_AVAILABLE:
        print("\n【Tiktoken测试】")
        tokenizer = TiktokenWrapper("cl100k_base")
        
        ids = tokenizer.encode(text_zh)
        decoded = tokenizer.decode(ids)
        print(f"原文: {text_zh}")
        print(f"Token数: {len(ids)}")
        print(f"解码: {decoded}")
        
        # 批量测试
        batch = tokenizer.encode_batch([text_zh, text_en], max_length=64)
        print(f"Batch shape: {batch['input_ids'].shape}")
    
    # 测试HuggingFace
    if TRANSFORMERS_AVAILABLE:
        print("\n【HuggingFace测试】")
        try:
            tokenizer = get_tokenizer("hf", model_name="gpt2")
            
            ids = tokenizer.encode(text_en)
            decoded = tokenizer.decode(ids)
            print(f"原文: {text_en}")
            print(f"Token数: {len(ids)}")
            print(f"解码: {decoded}")
        except Exception as e:
            print(f"跳过HF测试: {e}")
    
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)