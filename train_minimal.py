import torch
import math
from model import LlamaConfig, LlamaForCausalLM
from torch.cuda.amp import autocast, GradScaler 

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """
    warmup + cosine decay：
    - 前 warmup_steps：lr 从 0 线性升到 base_lr
    - 后面：lr 从 base_lr 按余弦慢慢降到接近 0
    """
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

def main():
    # 1. 配置模型（先用一个很小的模型，方便在CPU上跑）
    config = LlamaConfig(
        vocab_size=1000,        # 词表大小（先来个小的）
        hidden_size=128,        # 隐藏维度
        num_hidden_layers=2,    # 层数
        num_attention_heads=4,  # 注意力头数
        num_key_value_heads=2,
        max_position_embeddings=64
    )

    # 2. 创建模型
    model = LlamaForCausalLM(config)
    
    # === 新增：开启梯度检查点（可选） ===
    # 如果显存不够或想训练更长序列，就开启
    USE_GRADIENT_CHECKPOINTING = False  # 改成True开启

    if USE_GRADIENT_CHECKPOINTING:
        model.enable_gradient_checkpointing()
    else:
        print("梯度检查点：关闭（默认模式）")

    # 3. 选择设备：如果有GPU就用GPU，否则用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. 构造一批“假数据”
    batch_size = 2
    seq_len = 10

    base_lr = 1e-3
    warmup_steps = 10
    # 5. 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    # ===AMP 新增：初始化梯度缩放器（只在GPU上用） ===
    use_amp = torch.cuda.is_available()  # 只有GPU才用AMP
    scaler = GradScaler(enabled=use_amp)  # enabled=False时，scaler什么都不做
    print(f"使用混合精度训练: {use_amp}")
    
    num_steps = 50
    for step in range(1, num_steps + 1):
        # 6. 前向 + 反向 + 更新（一个训练 step）
        model.train()  # 切到训练模式
        lr = get_lr(step, warmup_steps, num_steps, base_lr)  #前面的warmup_steps从0线性升到base_lr，后面按余弦慢慢降到接近0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 每一步重新采样一批“数据”
        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, seq_len),
            dtype=torch.long
        ).to(device)
        labels = input_ids.clone()
        
        # === AMP修改：用autocast包裹前向传播 ===
        # autocast会自动把模型计算转成FP16（在GPU上）
        with autocast(enabled=use_amp):
        # 前向：得到 loss 和 logits
            loss, logits = model(input_ids=input_ids, labels=labels)

        print("loss =", loss.item())
        print("logits 形状 =", logits.shape)  # [batch_size, seq_len, vocab_size]

        # 清空梯度
        optimizer.zero_grad()

        """loss.backward() + optimizer.step()
        反向传播，把 loss 对每个参数的梯度算出来
    s   tep() 用梯度更新参数，完成一次训练步骤"""
        # 反向传播+AMP修改：用scaler.scale()包裹loss.backward()
        scaler.scale(loss).backward()
        
        # scaler.unscale_(optimizer)：把梯度缩小回原始值再梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #梯度裁剪，防止梯度变化过大导致训练重坏

        p = model.model.embed_tokens.weight  # 取 embedding 的权重矩阵
        print("更新前 p[0,0] =", p[0, 0].item())
        print("它的梯度 grad[0,0] =", p.grad[0, 0].item())
        
        # 参数更新采用AMP：用scaler.step()和scaler.update()
        scaler.step(optimizer)  # 检查梯度是否有NaN/Inf
        scaler.update() #动态调整缩放倍数

        print("更新后 p[0,0] =", p[0, 0].item())

        print(f"step {step}/{num_steps} | loss = {loss.item():.4f} | lr = {lr:.6f}")



if __name__ == "__main__":
    main()