import torch
import math
from model import LlamaConfig, LlamaForCausalLM,get_tokenizer
from torch.cuda.amp import autocast, GradScaler 
import json
import os

def load_training_data(data_path: str = "./data/training_data.jsonl"):
    """
    从JSONL文件加载训练文本
    
    文件格式示例：
    {"text": "第一句话"}
    {"text": "第二句话"}
    
    返回：["第一句话", "第二句话", ...]
    """
    texts = []
    
    # 尝试打开文件
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # 跳过空行
                    try:
                        item = json.loads(line)  # 解析JSON
                        if 'text' in item:
                            texts.append(item['text'])
                        else:
                            print(f"  第{line_num}行缺少'text'字段")
                    except json.JSONDecodeError:
                        print(f"  第{line_num}行JSON格式错误")
        
        if len(texts) == 0:
            raise ValueError("数据文件为空")
        
        return texts
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f" 找不到数据文件: {data_path}\n"
            f"请先创建：mkdir -p data && touch {data_path}"
        )

def prepare_batch_data(texts, tokenizer, batch_size, max_length=128):
    """
    把文本列表处理成模型能用的batch
    
    输入：["文本1", "文本2", ...]
    输出：[{"input_ids": tensor, "labels": tensor}, ...]
    """
    batches = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_input_ids = []
        batch_labels = []
        
        for text in batch_texts:
            # 1. 把文本转成token IDs
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            
            # 2. 截断（太长了就切掉）
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # 3. 填充（太短了就补齐）
            if len(token_ids) < max_length:
                pad_len = max_length - len(token_ids)
                labels = token_ids + [-100] * pad_len
                token_ids = token_ids + [tokenizer.pad_token_id] * pad_len
            else:
                labels = token_ids.copy()
            
            batch_input_ids.append(token_ids)
            batch_labels.append(labels)
        
        batches.append({
            "input_ids": torch.tensor(batch_input_ids),
            "labels": torch.tensor(batch_labels)
        })
    
    return batches

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
        hidden_size=512,        # 隐藏维度
        num_hidden_layers=8,    # 层数
        num_attention_heads=8,  # 注意力头数
        num_key_value_heads=4,
        max_position_embeddings=512,
        tokenizer_type="tiktoken",    # 新增：指定tokenizer类型
        tokenizer_name="cl100k_base"  # 新增：tokenizer名称
    )

    # 初始化真实tokenizer
    print(f"初始化 {config.tokenizer_type} Tokenizer...")
    tokenizer = get_tokenizer(
        config.tokenizer_type,
        encoding_name=config.tokenizer_name
    )

    # 同步真实的vocab_size
    config.vocab_size = tokenizer.vocab_size
    print(f" Tokenizer加载完成，vocab_size={config.vocab_size}")
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
    
    print("加载训练数据...")
    data_path = "./data/training_data.jsonl"

    try:
        texts = load_training_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        print("请先准备数据文件！")
        return  # 退出程序
    
    batch_size = 4
    seq_len = 128

    base_lr = 1e-3
    warmup_steps = 10
    # 5. 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    batches = prepare_batch_data(texts, tokenizer, batch_size, seq_len)
    print(f"创建了 {len(batches)} 个batch")
    
    # ===AMP 新增：初始化梯度缩放器（只在GPU上用） ===
    use_amp = torch.cuda.is_available()  # 只有GPU才用AMP
    scaler = GradScaler(enabled=use_amp)  # enabled=False时，scaler什么都不做
    print(f"使用混合精度训练: {use_amp}")
    

    num_epochs = 3  # 训练3轮
    total_steps = num_epochs * len(batches)
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        for batch_idx, batch in enumerate(batches, 1):
            global_step += 1
        
            # 6. 前向 + 反向 + 更新（一个训练 step）
            model.train()  # 切到训练模式
            lr = get_lr(global_step, warmup_steps, total_steps, base_lr)  #前面的warmup_steps从0线性升到base_lr，后面按余弦慢慢降到接近0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            #使用真实数据
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
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
            step() 用梯度更新参数，完成一次训练步骤"""
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

            print(f"step {global_step}/{total_steps} | loss = {loss.item():.4f} | lr = {lr:.6f}")

    # 保存模型
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, os.path.join(checkpoint_dir, "final_model.pt"))
    print(f"\n模型已保存到: {checkpoint_dir}/final_model.pt")

if __name__ == "__main__":
    main()