# model/rope_extended.py
"""
NTK-aware RoPE 实现
论文：
1. RoFormer (Su et al., 2021) - 原始RoPE
2. Scaled RoPE (kaiokendev, 2023) - 线性插值
3. NTK-aware (Bloc97, 2023) - 更好的频率缩放
"""
import torch
import torch.nn as nn
import math
from typing import Tuple

class NTKScaledRoPE(nn.Module):
    """
    NTK-aware 位置编码扩展
    - base_theta: 原始频率基数（默认10000）
    - max_position_embeddings: 训练时最大序列长度
    - scaling_factor: 外推倍数（如4.0表示支持4倍长度）
    """
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base_theta: float = 10000.0,
        scaling_factor: float = 1.0,
        device=None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base_theta = base_theta
        self.scaling_factor = scaling_factor
        
        # NTK插值：调整频率基数而非位置索引
        # 公式：theta' = theta * scaling_factor^(d/(d-2))
        # 论文依据：Bloc97的NTK-by-parts方法
        if scaling_factor > 1.0:
            # alpha = scaling_factor^(dim / (dim - 2))
            alpha = scaling_factor ** (dim / (dim - 2))
            adjusted_theta = base_theta * alpha
        else:
            adjusted_theta = base_theta
        
        # 计算频率：1 / (theta^(2i/d))，i=0,1,...,d/2-1
        inv_freq = 1.0 / (
            adjusted_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算cos/sin缓存（可选，加速推理）
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings * scaling_factor,
            device=device or inv_freq.device
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device):
        """预计算位置编码缓存"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # freqs: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # 拼接成 [seq_len, dim]（每个频率复制两次）
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回cos和sin，用于外部旋转
        x: [batch, num_heads, seq_len, head_dim]
        返回: (cos, sin) 各为 [1, 1, seq_len, head_dim]
        """
        # 如果超出缓存长度，重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE旋转辅助函数：将x的后半维度取负并交换
    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码
    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed