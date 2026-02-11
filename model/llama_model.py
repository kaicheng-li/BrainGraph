import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .llama_config import LlamaConfig

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
    多头自注意力（不含 RoPE、KV cache，先实现最基本版本）
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads 
        self.rope_theta = config.rope_theta

        # 每个头的维度 = hidden_size / 头数
        assert self.hidden_size % self.num_heads == 0, "hidden_size 必须能整除 num_attention_heads"
        self.head_dim = self.hidden_size // self.num_heads
        
        # === KV Cache关键修改1：计算分组数 ===
        # num_kv_groups = 每组有多少个Q头共享1组KV
        # 例如：32 ÷ 8 = 4（每4个Q头共享1组KV）
        assert self.num_heads % self.num_kv_heads == 0, "num_heads 必须能整除 num_kv_heads"
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # 把输入投影到 Q / K / V 三个空间
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # K/V投影：输出维度 = num_kv_heads * head_dim（比Q小！）
        # 例如：8 * 64 = 512，而Q是 32 * 64 = 2048
        kv_hidden_size = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=False)  # ← 改这里
        self.v_proj = nn.Linear(self.hidden_size, kv_hidden_size, bias=False)  # ← 改这里


        # 注意力输出再映射回 hidden_size
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states: [B, T, C]
        past_key_value: (k_cache, v_cache)
          - k_cache: [B, num_kv_heads, past_len, head_dim]
          - v_cache: [B, num_kv_heads, past_len, head_dim]
        """
        bsz, seq_len, _ = hidden_states.shape

        # 1. 线性投影得到 Q, K, V
        q = self.q_proj(hidden_states)  # [B, T, C]
        k = self.k_proj(hidden_states)  # [B, T, C]
        v = self.v_proj(hidden_states)  # [B, T, C]

        # 1) 准备频率：inv_freq 形状 [head_dim/2]
        dim = self.head_dim
        device = hidden_states.device

        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
            )
        )  # [dim/2]

        # 2) 位置索引：0,1,...,T-1
        t = torch.arange(seq_len, device=device, dtype=torch.float32)  # [T]

        # 3) 外积得到角度矩阵 freqs: [T, dim/2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]

        # 4) 拼成 dim 维，并算出 cos/sin
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        cos = emb.cos()[None, None, :, :]  # [1, 1, T, dim]
        sin = emb.sin()[None, None, :, :]  # [1, 1, T, dim]

        # 5) 定义一个帮助函数，旋转后一半
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        # 6) 先 reshape 成 [B, T, H, D] 再应用 RoPE
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)  # [B, T, H, D]
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)  # [B, T, H, D]
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)  # [B, T, H, D]
        # 为了和 [1,1,T,D] 的 cos/sin 对齐，把 T 维放到后面：[B, H, T, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        # 应用旋转： (x * cos) + (rotate_half(x) * sin)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        # === KV Cache拼接 ===
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # 拼到序列维
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None

        # === 新增：GQA关键 - 扩展K/V头数以匹配Q ===
        # 现状：k和v是 [B, 2, T, D]，q是 [B, 4, T, D]
        # 需要：让k和v也变成 [B, 4, T, D]
        if self.num_kv_groups > 1:  # 如果num_kv_groups=2，表示需要复制
            # Step 1: 在第3维插入一个新维度
            # k: [B, 2, T, D] → [B, 2, 1, T, D]
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            
            # Step 2: 沿着新维度重复 num_kv_groups 次
            # k: [B, 2, 1, T, D] → [B, 2, 2, T, D]（每个KV头复制2次）
            k = k.repeat(1, 1, self.num_kv_groups, 1, 1)
            v = v.repeat(1, 1, self.num_kv_groups, 1, 1)
            
            # Step 3: 合并成和Q一样的头数
            # k: [B, 2, 2, T, D] → reshape → [B, 4, T, D]
            # 结果：k[0,1]用原来的k[0]，k[2,3]用原来的k[1]
            k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v.reshape(bsz, self.num_heads, -1, self.head_dim)
        # 现在 k 和 v 的形状是 [B, H, T, D]，和 q 对齐了

        # 4. 计算注意力分数： Q * K^T / sqrt(D)
        try:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,      # 不需要手动传mask
                dropout_p=0.0,       # 暂时不用dropout
                is_causal=True,      # 自动应用因果mask！
                scale=None           # 自动缩放 1/sqrt(D)
            )  # 输出: [B, H, T, D]
        except Exception as e:
            # 如果 scaled_dot_product_attention 不可用，回退到手动实现
            attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]
            attn_scores = attn_scores / math.sqrt(self.head_dim)

            # === 因果 mask：遮住右上角，让每个位置只能看见自己和左边 ===
            # 构造一个 [T, T] 的矩阵，上三角（不含对角线）为 1，其它为 0
            # 例如 T=4 时：
            # [[0, 1, 1, 1],
            #  [0, 0, 1, 1],
            #  [0, 0, 0, 1],
            #  [0, 0, 0, 0]]
            seq_len = q.size(-2) #q的形状是[B, H, T, D]，所以seq_len是T
            causal_mask = torch.triu(   #triu函数功能是返回一个上三角矩阵，对角线为1，其它为0
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )  # [T, T], True 表示“要遮住”

            # 把要遮住的位置加上一个非常大的负数（-1e9 或 -inf），softmax 后这些位置的概率≈0
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

            # 5. 做 softmax 得到注意力权重
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, T, T]

            # 6. 用注意力权重加权 V，得到每个位置的聚合信息
            attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]

        # 7. 把 heads 维度挪回去，并合并回 hidden_size：
        #    [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
        attn_output = attn_output.transpose(1, 2)  # [B, T, H, D]
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)  # [B, T, C]

        # 8. 过一个线性层，映射回 hidden_size（这一步是“输出投影”）
        output = self.o_proj(attn_output)  # [B, T, C]

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