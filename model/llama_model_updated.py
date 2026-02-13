# model/llama_model_upgraded.py
"""
升级版 LlamaModel：集成NTK-RoPE
在原有llama_model.py基础上修改SelfAttention部分
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .llama_config import LlamaConfig
from .rope_extended import NTKScaledRoPE, apply_rotary_pos_emb

class SelfAttentionUpgraded(nn.Module):
    """
    升级版自注意力：支持NTK-RoPE长上下文扩展
    论文：
    - RoPE (Su et al., 2021)
    - NTK-aware scaling (Bloc97, 2023)
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        kv_hidden_size = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # === NTK-RoPE初始化 ===
        rope_scaling = config.rope_scaling
        if rope_scaling and rope_scaling.get("type") == "ntk":
            scaling_factor = rope_scaling.get("factor", 1.0)
        else:
            scaling_factor = 1.0
        
        self.rotary_emb = NTKScaledRoPE(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base_theta=config.rope_theta,
            scaling_factor=scaling_factor
        )
        
        print(f"  ✓ SelfAttention使用NTK-RoPE，scaling_factor={scaling_factor}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape
        
        # 1. QKV投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Reshape
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用NTK-RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 4. KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None
        
        # 5. GQA：扩展KV头
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).repeat(1, 1, self.num_kv_groups, 1, 1)
            v = v.unsqueeze(2).repeat(1, 1, self.num_kv_groups, 1, 1)
            k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v.reshape(bsz, self.num_heads, -1, self.head_dim)
        
        # 6. Flash Attention
        try:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        except:
            # Fallback
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # 7. 输出
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output, present_key_value