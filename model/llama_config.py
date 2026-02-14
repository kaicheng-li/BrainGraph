from dataclasses import dataclass
from typing import Optional

@dataclass
class LlamaConfig:
    """
    Llama 风格 Decoder-only Transformer 的配置类
    """
    # 词表大小（多少个 token ID）
    vocab_size: int = 32000
    # 每个 token 嵌入 / 隐藏向量的维度
    hidden_size: int = 512
    # Transformer 块的层数
    num_hidden_layers: int = 8
    # 多头注意力里的头数
    num_attention_heads: int = 8
    # KV 头的数量（GQA：query 头多，key/value 头少）
    num_key_value_heads: int = 8
    # 前馈网络中间层维度（如果为 None，我们后面会用 4 * hidden_size）
    intermediate_size: Optional[int] = None
    # 最大支持的序列长度（位置编码最大长度）
    max_position_embeddings: int = 2048
    # RMSNorm 里的 eps，防止除以 0
    rms_norm_eps: float = 1e-5
    # RoPE 的 base，控制位置编码频率（Llama 用 1e4 或 1e6 一类的值）
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict]=None # {"type": "ntk", "factor": 4.0}
    
    # === Graph配置（新增） ===
    use_graph: bool = False                    # ← 是否启用Graph模块
    graph_node_dim: int = 64                   # ← Graph节点特征维度
    graph_num_layers: int = 3                  # ← Graph编码器层数
    graph_encoder_type: str = "gat"            # ← GCN或GAT
    graph_use_lora: bool = False               # ← Graph是否用LoRA
    
    # === LoRA配置 ===
    use_lora: bool = False
    lora_r: int = 8              # LoRA秩
    lora_alpha: int = 16         # 缩放因子
    lora_dropout: float = 0.05   # Dropout
    lora_target_modules: Optional[list] = None  # 要应用LoRA的模块，如["q_proj", "v_proj"]
    
    # 特殊 token 的 ID（这里先给一组常见默认值）
    bos_token_id: int = 1  # beginning of sentence
    eos_token_id: int = 2  # end of sentence
    pad_token_id: int = 0  # padding
    # === Tokenizer配置（新增）===
    tokenizer_type: str = "tiktoken"  # "tiktoken" | "huggingface"
    tokenizer_name: str = "cl100k_base"  # tiktoken编码名或HF模型名