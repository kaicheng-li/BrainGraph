import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .llama_config import LlamaConfig
from .graph_encoder import GraphEncoder
from .fusion_layer import GraphTextFusion
from .rope_extended import NTKScaledRoPE, apply_rotary_pos_emb

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Llama 使用的归一化方式)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 存一个很小的数字，防止除以 0
        self.eps = eps
        # 可学习的缩放参数，维度 = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]

        # 1. 计算每个位置上向量的平方均值：mean(x^2)
        # keepdim=True 保持最后一维，方便后面做广播
        norm = x.pow(2).mean(-1, keepdim=True)

        # 2. 取平方根并加上 eps，得到 RMS
        rms = torch.sqrt(norm + self.eps)

        # 3. 用 x / rms 做归一化，再乘以可学习的 weight
        x_norm = x / rms
        return self.weight * x_norm

class LlamaModel(nn.Module):
    """
    Llama 风格的 Decoder-only Transformer 主体（目前只实现基本结构）
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # === 新增：梯度检查点开关 ===
        self.gradient_checkpointing = False  # 默认关闭，训练长序列时开启

        # 1. token 嵌入层：把 token id 映射到向量
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        # 2. 输入后的 dropout
        self.dropout = nn.Dropout(p=0.0)

        # === 新增：堆叠多层 Decoder 层 ===
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # 3. 最后的 RMSNorm（所有层叠加完再做一次）
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # === 新增：权重初始化（来自GPT标准）===
        self.apply(self._init_weights) #会递归遍历LlamaModel里的所有子模块，并把它们交给_init_weights()处理。

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT/Llama 标准初始化：
        - Linear/Embedding: 正态分布 N(0, 0.02)
        - bias: 0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #GPT和Llama的经典做法，能让训练早期的梯度更稳定。
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        input_ids: [batch_size, seq_len] 的 token id
        """

        # 1. 先通过嵌入层得到向量表示
        hidden_states = self.embed_tokens(input_ids)  # [B, T, C]
        #实际表示hidden_states = self.embed_tokens.forward(input_ids)

        # 2. 简单加个 dropout
        hidden_states = self.dropout(hidden_states)
        
        # KV cache 相关：如果提供了 past_key_values，就传给每一层；如果 use_cache=True，就让每层返回 present_key_values
        present_key_values = () if use_cache else None
        
        # === 3. 依次通过每一层 Decoder（新增梯度检查点） ===
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None 
            # 判断是否使用梯度检查点
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点（训练模式下）
                # torch.utils.checkpoint.checkpoint：
                # - 前向时：只保存输入和输出，不保存中间结果
                # - 反向时：重新运行一遍前向，计算梯度
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda x: layer(x, use_cache=False)[0],                  # 要执行的模块
                    hidden_states,          # 输入参数
                    use_reentrant=False     # PyTorch 2.0+推荐设为False
                )
                present_key_value = None
            else:
                # 正常模式（推理或未开启梯度检查点）
                hidden_states, present_key_value = layer(
                    hidden_states,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
        # 4. 最后做一次 RMSNorm
        hidden_states = self.norm(hidden_states)

        # 5. 返回 hidden_states（后面会在外面接 LM 头）
        return hidden_states, present_key_values

class LlamaDecoderLayer(nn.Module):
    """
    单层 Decoder：RMSNorm + SelfAttention + 残差
                + RMSNorm + FFN + 残差
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # 注意力相关
        self.self_attn = SelfAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 前馈网络相关
        self.mlp = FeedForward(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        """

        # === 自注意力子层 ===
        # 残差 1
        residual = hidden_states

        # pre-norm
        normed_hidden = self.input_layernorm(hidden_states)

        # Self-Attention
        attn_output, present_key_value = self.self_attn(
            normed_hidden,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # 残差加回去
        hidden_states = residual + attn_output

        # === 前馈网络子层 ===
        # 残差 2
        residual = hidden_states

        # pre-norm
        normed_hidden = self.post_attention_layernorm(hidden_states)

        # FFN
        ffn_output = self.mlp(normed_hidden)

        # 残差加回去
        hidden_states = residual + ffn_output

        return hidden_states, present_key_value    

class FeedForward(nn.Module):
    """
    前馈网络（FFN），使用 SwiGLU 激活，Llama 风格
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        hidden_size = config.hidden_size

        # 如果 intermediate_size 没写，就用 4 * hidden_size
        intermediate_size = config.intermediate_size or (4 * hidden_size)

        # gate_proj / up_proj 是 SwiGLU 的两支
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # down_proj 把维度再投影回 hidden_size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(xW_g) * (xW_u)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return x

class SelfAttention(nn.Module):
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

class LlamaForCausalLM(nn.Module):
    """
    在 LlamaModel 外面加一个线性层，把 hidden 映射到 vocab，
    用于因果语言建模（Causal LM）。
    """
    """LlamaForCausalLM = LlamaModel + 词表线性头 + 损失计算
    这是 HuggingFace Transformers 框架里的命名习惯：
    LlamaModel：只包含 Transformer 主体，输出是 [batch, seq_len, hidden_size] 的隐藏向量，不直接给你词表概率。
    LlamaForCausalLM：在 LlamaModel 外面再套一层，加一个线性层（lm_head）把 hidden -> vocab，并且提供：
        logits：每个位置对每个 token 的打分 [batch, seq_len, vocab_size]
        loss：如果你给了 labels，就自动算交叉熵损失，方便训练。"""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # 1. 底座：我们刚刚实现的 LlamaModel
        self.model = LlamaModel(config)

        # 2. 词表线性层：hidden_size -> vocab_size
        self.lm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False
        )

        # （可选）权重共享：embedding 和 lm_head 共用一套权重
        # 这样可以少一份参数，也是一种常见做法（Llama/MiniMind 等都这么干）
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T] token id
        labels: Optional[torch.Tensor] = None,  # [B, T]，可选，用于算 loss
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        """
        返回：
          - 如果 labels 是 None：只返回 logits
          - 如果 labels 不为 None：返回 (loss, logits)
        """

        # 1. 先通过 LlamaModel 得到 hidden_states: [B, T, C]
        hidden_states, present_key_values = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)  # [B, T, hidden_size]

        # 2. 通过 lm_head 映射到词表维度： [B, T, vocab_size]
        logits = self.lm_head(hidden_states)

        # 3. 如果不给 labels，就只做前向推理
        if labels is None:
            if use_cache:
                return logits, present_key_values
            else:
                return logits
        # 4. 如果给了 labels，就计算语言模型的交叉熵损失
        #    为了预测“下一个 token”，会做一个右移操作（shift）
        #    - shift_logits: 不包含最后一个时间步
        #    - shift_labels: 不包含第一个时间步
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()       # [B, T-1]

        # 把 [B, T-1, V] 拉平成 [B*(T-1), V]，方便用 cross_entropy
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100   # 忽略 label = -100 的位置（常用于 padding）
        )

        return (loss, logits, present_key_values) if use_cache else (loss, logits)
    
    def enable_gradient_checkpointing(self):
        """
        开启梯度检查点（训练长序列时使用）
        
        使用场景：
        - 序列长度 > 1024 tokens
        - 显存不够时
        - batch size需要更大时
        
        代价：训练速度慢20-30%
        """
        self.model.gradient_checkpointing = True
        print("✓ 已开启梯度检查点（显存优化模式）")

    def disable_gradient_checkpointing(self):
        """
        关闭梯度检查点（默认状态）
        
        使用场景：
        - 短序列训练
        - 显存足够时
        - 追求最快训练速度
        """
        self.model.gradient_checkpointing = False
        print("✓ 已关闭梯度检查点（速度优先模式）")

class GraphLlamaForCausalLM(nn.Module):
    """
    Graph-Enhanced Llama
    论文参考：GreaseLM (ICLR 2022), GraphGPT (arXiv 2023)
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # 1) 基座Llama
        self.llama = LlamaForCausalLM(config)

        # 2) 图模块
        self.use_graph = config.use_graph  # ✅ 从config取
        if self.use_graph:
            self.graph_encoder = GraphEncoder(
                node_dim=config.graph_node_dim,  # ✅ 从config取
                hidden_size=config.hidden_size,
                num_layers=config.graph_num_layers,  # ✅
                encoder_type=config.graph_encoder_type,  # ✅
                use_lora=config.graph_use_lora if config.graph_use_lora is not None else config.use_lora,  # ✅ 继承
                lora_r=config.lora_r,  # ✅
                gradient_checkpointing=self.llama.model.gradient_checkpointing  # ✅ 继承
            )
            self.fusion = GraphTextFusion(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                use_flash_attention=True  # ✅ 和Llama保持一致
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        graph_data: Optional[dict] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        # 1) 文本embedding
        text_emb = self.llama.model.embed_tokens(input_ids)  # [B, T, C]

        # 2) 融合图信息（如果提供）
        if self.use_graph and graph_data is not None:
            node_emb, _ = self.graph_encoder(
                graph_data["node_features"],
                graph_data["edge_index"],
                graph_data.get("batch", None)
            )
            # 扩展batch维度
            if node_emb.dim() == 2:
                node_emb = node_emb.unsqueeze(0).expand(text_emb.size(0), -1, -1)
            # 融合
            text_emb = self.fusion(text_emb, node_emb)

        # 3) 替换embedding后继续Llama前向
        hidden_states = self.llama.model.dropout(text_emb)

        present_key_values = () if use_cache else None
        for i, layer in enumerate(self.llama.model.layers):
            past_kv = past_key_values[i] if past_key_values else None

            if self.llama.model.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda x: layer(x, use_cache=False)[0],
                    hidden_states,
                    use_reentrant=False
                )
                present_kv = None
            else:
                hidden_states, present_kv = layer(hidden_states, past_kv, use_cache)

            if use_cache:
                present_key_values = present_key_values + (present_kv,)

        hidden_states = self.llama.model.norm(hidden_states)
        logits = self.llama.lm_head(hidden_states)

        if labels is None:
            return (logits, present_key_values) if use_cache else logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return (loss, logits, present_key_values) if use_cache else (loss, logits)