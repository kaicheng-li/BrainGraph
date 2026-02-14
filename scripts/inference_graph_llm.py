"""
Graph+LLM高级推理引擎 (Qwen2.5级别)
✅ Beam Search
✅ Top-p (Nucleus) Sampling
✅ Temperature控制
✅ Repetition Penalty
✅ KV Cache加速
✅ 混合精度推理
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from model import LlamaConfig, GraphLlamaForCausalLM
from model.tokenizer import get_tokenizer
from model.llama_model import LoRALinear

class GraphLLMInference:
    def __init__(
        self, 
        checkpoint_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        use_amp: bool = True,
        amp_dtype: str = "bfloat16"
    ):
        """
        高级推理引擎
        
        Args:
            checkpoint_path: SFT模型路径
            lora_path: LoRA适配器路径(可选)
            device: cuda/cpu
            use_amp: 混合精度推理
            amp_dtype: bfloat16/float16
        """
        print(f"\n{'='*70}")
        print(" GraphLLM推理引擎初始化")
        print(f"{'='*70}\n")
        
        # ========== 1. 加载模型 ==========
        print(f"🔄 加载模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config_dict = ckpt["config"]
        self.config = LlamaConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
        
        self.model = GraphLlamaForCausalLM(
            self.config, 
            graph_hidden_size=getattr(self.config, 'graph_node_dim', 64)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        print("✅ 基础模型加载完成")
        
        # ========== 2. 加载LoRA (可选) ==========
        if lora_path and Path(lora_path).exists():
            print(f"🔄 加载LoRA: {lora_path}")
            lora_ckpt = torch.load(lora_path, map_location="cpu")
            lora_state = lora_ckpt['lora_state']
            
            for name, lora_params in lora_state.items():
                module = self.model
                for attr in name.split('.'):
                    module = getattr(module, attr)
                if isinstance(module, LoRALinear):
                    module.lora_A.data = lora_params['lora_A']
                    module.lora_B.data = lora_params['lora_B']
            print("✅ LoRA权重加载完成")
        
        # ========== 3. Tokenizer ==========
        self.tokenizer = get_tokenizer(
            tokenizer_type=getattr(self.config, 'tokenizer_type', 'tiktoken'),
            model_name=getattr(self.config, 'tokenizer_name', 'gpt-4')
        )
        
        # ========== 4. 设备和精度 ==========
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.use_amp = use_amp and torch.cuda.is_available()
        self.dtype = torch.float32
        if self.use_amp:
            if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        
        print(f"📐 设备: {self.device}")
        print(f"🔥 推理精度: {self.dtype}")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        graph_data: Optional[Dict] = None,
        # === 生成参数 ===
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True,
        # === 特殊token ===
        eos_token_id: Optional[int] = None,
    ) -> str:
        """
        生成文本 (支持多种采样策略)
        
        Args:
            prompt: 输入文本
            graph_data: 图数据 (可选)
            max_new_tokens: 最大生成token数
            temperature: 温度 (0=贪心, >1=随机)
            top_k: Top-K采样
            top_p: Nucleus采样
            repetition_penalty: 重复惩罚
            num_beams: Beam Search (>1启用)
            do_sample: 是否采样 (False=贪心)
            eos_token_id: 结束token
        """
        # 编码
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
        
        # Beam Search
        if num_beams > 1:
            return self._beam_search(
                input_ids, graph_data, max_new_tokens, 
                num_beams, eos_token_id
            )
        
        # 标准采样
        return self._sample_generate(
            input_ids, graph_data, max_new_tokens,
            temperature, top_k, top_p, 
            repetition_penalty, do_sample, eos_token_id
        )
    
    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        graph_data: Optional[Dict],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        eos_token_id: int
    ) -> str:
        """标准采样生成"""
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # 当前输入
            if step == 0:
                current_input = input_ids
            else:
                current_input = next_token.unsqueeze(0)
            
            # 前向传播 (混合精度)
            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
                if graph_data and step == 0:
                    # 第一步使用Graph
                    outputs = self.model(
                        input_ids=current_input,
                        node_features=graph_data.get('node_features').to(self.device) if graph_data.get('node_features') is not None else None,
                        edge_index=graph_data.get('edge_index').to(self.device) if graph_data.get('edge_index') is not None else None,
                        batch_map=graph_data.get('batch_map', torch.zeros(graph_data['node_features'].size(0), dtype=torch.long)).to(self.device) if graph_data.get('node_features') is not None else None,
                        use_cache=True
                    )
                    logits = outputs['logits']
                    past_key_values = outputs.get('past_key_values')
                else:
                    # 后续步骤只用LLM
                    outputs = self.model.llama(
                        current_input,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
                    past_key_values = outputs.get('past_key_values') if isinstance(outputs, dict) else outputs[1]
            
            # 取最后一个token的logits
            next_token_logits = logits[0, -1, :]
            
            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    next_token_logits[token_id] /= repetition_penalty
            
            # 温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-K过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样或贪心
            probs = F.softmax(next_token_logits, dim=-1)
            if do_sample and temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # 拼接
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # EOS检查
            if next_token.item() == eos_token_id:
                break
        
        # 解码
        return self.tokenizer.decode(generated_ids[0].tolist())
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        graph_data: Optional[Dict],
        max_new_tokens: int,
        num_beams: int,
        eos_token_id: int
    ) -> str:
        """Beam Search生成"""
        batch_size = input_ids.size(0)
        
        # 初始化beams
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = float('-inf')
        beam_sequences = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        
        done = [False] * num_beams
        
        for step in range(max_new_tokens):
            # 扁平化处理
            current_input = beam_sequences.view(batch_size * num_beams, -1)
            
            # 前向传播
            with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
                outputs = self.model.llama(current_input)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
            
            next_token_logits = logits[:, -1, :]  # [batch*beams, vocab]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # 计算beam scores
            next_token_scores = next_token_scores.view(batch_size, num_beams, -1)
            next_scores = beam_scores.unsqueeze(-1) + next_token_scores
            
            # 重塑并取top-k
            next_scores = next_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=-1)
            
            # 更新beams
            beam_idx = top_indices // next_token_scores.size(-1)
            token_idx = top_indices % next_token_scores.size(-1)
            
            beam_sequences = torch.cat([
                beam_sequences[torch.arange(batch_size).unsqueeze(1), beam_idx],
                token_idx.unsqueeze(-1)
            ], dim=-1)
            beam_scores = top_scores
            
            # 检查EOS
            if (token_idx == eos_token_id).any():
                break
        
        # 返回最佳序列
        best_sequence = beam_sequences[0, beam_scores[0].argmax()]
        return self.tokenizer.decode(best_sequence.tolist())
    
    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]] = None,
        graph_data: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        对话接口 (支持多轮对话)
        
        Args:
            message: 用户消息
            history: 对话历史 [(user, assistant), ...]
            graph_data: 图数据
            **kwargs: generate参数
        
        Returns:
            (response, updated_history)
        """
        if history is None:
            history = []
        
        # 构造prompt
        prompt = ""
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        # 生成
        response = self.generate(prompt, graph_data=graph_data, **kwargs)
        
        # 提取assistant回复
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # 更新历史
        history.append((message, response))
        
        return response, history


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 初始化推理引擎
    engine = GraphLLMInference(
        checkpoint_path="checkpoints/graph_llama_sft.pt",
        lora_path="checkpoints/lora_adapter.pt",  # 可选
        device="cuda",
        use_amp=True,
        amp_dtype="bfloat16"
    )
    
    # 示例1: 纯文本生成
    print("【示例1】纯文本生成:")
    response = engine.generate(
        prompt="什么是图神经网络？",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=100
    )
    print(f"回复: {response}\n")
    
    # 示例2: Graph+Text推理
    print("【示例2】图推理:")
    graph_data = {
        'node_features': torch.randn(10, 64),
        'edge_index': torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)
    }
    response = engine.generate(
        prompt="根据图结构，从节点0到节点4的最短路径是什么？",
        graph_data=graph_data,
        temperature=0.3,  # 低温度=更确定
        max_new_tokens=50
    )
    print(f"回复: {response}\n")
    
    # 示例3: Beam Search
    print("【示例3】Beam Search:")
    response = engine.generate(
        prompt="请列举3种常见的图算法",
        num_beams=4,
        do_sample=False,
        max_new_tokens=80
    )
    print(f"回复: {response}\n")
    
    # 示例4: 多轮对话
    print("【示例4】多轮对话:")
    history = []
    response, history = engine.chat("你好，我想学习图神经网络", history)
    print(f"AI: {response}")
    
    response, history = engine.chat("它有哪些应用？", history)
    print(f"AI: {response}")