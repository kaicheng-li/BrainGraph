import torch
from model import LlamaConfig, LlamaForCausalLM
import time
def generate(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,  # [1, prefix_len]
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 50
):
    """
    带KV Cache的自回归生成
    
    参数：
    - input_ids: 初始prompt的token ids [1, prefix_len]
    - max_new_tokens: 最多生成多少个新token
    - temperature: 温度（越大越随机，越小越确定）
    - top_k: 只从概率最高的top_k个token中采样
    """
    model.eval()
    device = input_ids.device
    
    past_key_values = None  # 初始没有缓存
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            start=time.time()
            # 第一步：处理整个prompt
            # 后续步骤：只处理上一步生成的1个token
            if step == 0:
                current_input = input_ids  # [1, prefix_len]
            else:
                current_input = next_token.unsqueeze(0)  # [1, 1]
            
            # 前向传播（使用KV cache）
            logits, past_key_values = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # 取最后一个位置的logits: [1, vocab_size]
            next_token_logits = logits[0, -1, :]
            
            # 温度缩放
            next_token_logits = next_token_logits / temperature
            
            # Top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Softmax + 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接到生成序列
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            elapsed = (time.time() - start) * 1000
            print(f"Step {step+1}: {elapsed:.2f}ms | token={next_token.item()}")
    
    return generated_ids


# === 使用示例 ===
if __name__ == "__main__":
    # 1. 创建模型
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64
    )
    model = LlamaForCausalLM(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. 准备prompt（随机生成一些token作为示例）
    prompt_ids = torch.randint(0, config.vocab_size, (1, 5)).to(device)
    print(f"Prompt tokens: {prompt_ids.tolist()}")
    
    # 3. 生成文本
    print("\n开始生成...")
    generated = generate(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=50
    )
    
    print(f"\n完整生成序列: {generated.tolist()}")
    print(f"生成长度: {generated.shape[1]} tokens")