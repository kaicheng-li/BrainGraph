import torch
import torch.nn as nn

class GraphTextFusion(nn.Module):
    """
    Graph-Text Fusion Layer
    作用：让文本token通过Cross-Attention“看到”图节点
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, use_flash_attention: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash_attention = use_flash_attention

        assert hidden_size % num_heads == 0, "hidden_size必须能整除num_heads"

        # Q来自文本，K/V来自图
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb:  [B, T, C]  文本token嵌入
        graph_emb: [B, N, C]  图节点嵌入
        返回:      [B, T, C]  融合后的文本表示
        """
        bsz, seq_len, _ = text_emb.shape
        _, num_nodes, _ = graph_emb.shape

        # 1) 线性投影
        q = self.q_proj(text_emb)
        k = self.k_proj(graph_emb)
        v = self.v_proj(graph_emb)

        # 2) 多头拆分
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(bsz, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) # [B,H,N,D]
        v = v.view(bsz, num_nodes, self.num_heads, self.head_dim).transpose(1, 2) # [B,H,N,D]

        # 3) Cross-Attention: 文本查询图节点
        if self.use_flash_attention:
            try:
                # Flash Attention优化（与Llama保持一致）
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False  # Cross-Attention无因果mask
                )
            except:
                # Fallback to manual implementation
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
        else:
            # Manual attention (原始实现)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,T,N]
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)  # [B,H,T,D]

        # 4) 合并多头
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        # 5) 残差 + LayerNorm
        return self.layer_norm(text_emb + output)