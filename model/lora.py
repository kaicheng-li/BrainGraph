# model/lora.py
"""
LoRA (Low-Rank Adaptation) 实现
论文：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022)

核心思想：
- 冻结预训练权重 W
- 添加低秩分解 ΔW = BA，其中 B∈R^(d×r), A∈R^(r×k), r<<min(d,k)
- 前向传播：y = Wx + BAx = Wx + ΔWx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class LoRALinear(nn.Module):
    """
    LoRA增强的线性层
    
    参数：
    - in_features, out_features: 原始Linear层的输入输出维度
    - r: LoRA秩（rank），通常取4/8/16/32，越小参数越少
    - lora_alpha: 缩放因子，控制LoRA的影响强度（通常 = r 或 2r）
    - lora_dropout: LoRA路径的dropout概率
    - merge_weights: 是否将LoRA权重合并到原始权重（推理时加速）
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        bias: bool = False
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merge_weights = merge_weights
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # LoRA权重（可训练）
        # A: [r, in_features]，用高斯初始化
        # B: [out_features, r]，用0初始化（保证初始ΔW=0）
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 缩放因子：alpha / r
        # 论文建议：让LoRA的学习率与秩无关
        self.scaling = lora_alpha / r
        
        # 标记是否已合并
        self.merged = False
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化权重"""
        # 原始权重用标准初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # LoRA A用高斯初始化（类似Kaiming）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # LoRA B用0初始化（保证初始时ΔW=BA=0）
        nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        """训练/推理模式切换"""
        super().train(mode)
        if mode:
            # 训练模式：如果之前合并了，先解除合并
            if self.merged:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # 推理模式：合并权重加速
            if self.merge_weights and not self.merged:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: [..., in_features]
        返回: [..., out_features]
        """
        if self.merged:
            # 已合并：直接用融合后的权重
            return F.linear(x, self.weight, self.bias)
        else:
            # 未合并：原始路径 + LoRA路径
            # 原始: Wx
            result = F.linear(x, self.weight, self.bias)
            
            # LoRA: BAx，带dropout和缩放
            # 1. x @ A^T: [..., in_features] @ [in_features, r] -> [..., r]
            # 2. 结果 @ B^T: [..., r] @ [r, out_features] -> [..., out_features]
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result += lora_out * self.scaling
            
            return result
    
    def extra_repr(self) -> str:
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, ' \
               f'r={self.r}, lora_alpha={self.lora_alpha}, merged={self.merged}'


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    冻结模型所有参数，只保留LoRA参数可训练
    
    参数：
    - model: 要处理的模型
    - bias: 'none'/'lora_only'/'all' - 是否训练bias
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    if bias == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.bias is not None:
                module.bias.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> dict:
    """只保存LoRA权重（用于微调后的模型保存）"""
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}