# === model/__init__.py 修复 ===
from .llama_config import LlamaConfig  # ← 恢复原始导入
from .llama_model import LlamaModel, LlamaForCausalLM, GraphLlamaForCausalLM
from .graph_encoder import GraphEncoder, GCNEncoder, GATEncoder
from .fusion_layer import GraphTextFusion
from .graph_policy import GraphPolicy
from .tokenizer import get_tokenizer
from .lora import LoRALinear, mark_only_lora_as_trainable
from .rope_extended import NTKScaledRoPE

__all__ = [
    "LlamaConfig", "LlamaModel", "LlamaForCausalLM", 
    "GraphLlamaForCausalLM", "GraphEncoder", "GCNEncoder", 
    "GATEncoder", "GraphTextFusion", "GraphPolicy", 
    "get_tokenizer", "LoRALinear", "NTKScaledRoPE",
    "mark_only_lora_as_trainable"
]