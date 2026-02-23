"""
LLaMA预训练（MiniMind架构 + 原有优化）
✅ argparse + DDP + Checkpoint + wandb（新增）
✅ GQA + NTK-RoPE（保留）
✅ 梯度检查点（保留）
✅ 混合精度训练（保留）
✅ 梯度累积（保留）
"""
import argparse
import os
import sys
__package__ = "trainer"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT) 
import warnings
import time
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.llama_config import LlamaConfig
from model.llama_model import LlamaForCausalLM
from datasets.pretrain_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)
import wandb as wandb_lib
warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个epoch"""
    start_time = time.time()
    
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 动态学习率（Warmup + Cosine）
        total_steps = args.epochs * iters
        warmup_steps = int(args.warmup_ratio * total_steps)
        lr = get_lr(epoch * iters + step, total_steps, args.learning_rate, 
                    args.min_lr, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播（混合精度）
        with autocast_ctx:
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]  # (loss, logits)
            loss = loss / args.accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积 + 梯度裁剪
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, lr: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.1f}min')
            
            if wandb:
                wandb.log({
                    'loss': current_loss,
                    'lr': current_lr,
                    'epoch': epoch + 1,
                    'step': epoch * iters + step
                })
        
        # 保存checkpoint
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints')
            model.train()
            del state_dict
        
        del input_ids, labels, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA Pretraining (MiniMind + Qwen2.5)")
    
    # === 目录和保存 ===
    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument('--save_weight', default='llama_pretrain', type=str, help="保存权重前缀")
    
    # === 训练配置 ===
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="最小学习率（Cosine衰减）")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup比例")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    
    # === 模型配置 ===
    parser.add_argument('--vocab_size', default=32000, type=int, help="词表大小")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer层数")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="GQA: KV头数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="最大序列长度")
    parser.add_argument('--rope_theta', default=10000.0, type=float, help="RoPE基础频率")
    parser.add_argument('--rope_scaling_factor', default=1.0, type=float, help="NTK-RoPE扩展因子")
    
    # === 优化配置 ===
    parser.add_argument('--use_gradient_checkpointing', default=1, type=int, choices=[0, 1], 
                        help="是否启用梯度检查点（节省显存）")
    
    # === 数据 ===
    parser.add_argument("--data_path", type=str, default="./datasets/", help="预训练数据路径")
    
    # === 权重和续训 ===
    parser.add_argument('--from_weight', default='none', type=str, help="基础权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否续训")
    
    # === wandb ===
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MyTransformer-Pretrain")
    
    # === torch.compile ===
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    
    args = parser.parse_args()
    
    # ========== 1. 初始化DDP和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置模型和检查checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ✅ 保留所有优化配置
    lm_config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,  # ✅ GQA
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        rope_scaling={"type": "ntk", "factor": args.rope_scaling_factor} if args.rope_scaling_factor > 1.0 else None,  # ✅ NTK-RoPE
    )
    
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume == 1 else None
    
    # ========== 3. 混合精度 ✅ ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Pretrain-E{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb = wandb_lib.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 模型和数据 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # ✅ 梯度检查点
    if args.use_gradient_checkpointing == 1 and torch.cuda.is_available():
        model.enable_gradient_checkpointing()
        Logger('💾 梯度检查点: 已启用（节省~70%显存）')
    
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 打印优化信息
    Logger(f"\n📐 模型配置:")
    Logger(f"  - 隐藏层: {args.hidden_size}d × {args.num_hidden_layers}层")
    Logger(f"  - 注意力: {args.num_attention_heads}头 (GQA: {args.num_key_value_heads} KV头)")
    Logger(f"  - 上下文: {args.max_seq_len} tokens")
    if args.rope_scaling_factor > 1.0:
        Logger(f"  - NTK-RoPE扩展: {args.rope_scaling_factor}x")
    Logger(f"  - 梯度检查点: {'✅ 已启用' if args.use_gradient_checkpointing else '❌ 未启用'}")
    Logger(f"  - 混合精度: {args.dtype.upper()}")
    Logger(f"  - 梯度累积: {args.accumulation_steps} steps")
    
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # ✅ 混合精度Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # ✅ 优化器（Qwen2.5参数）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.95),  # ✅ Qwen2.5标准
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # ========== 6. 从checkpoint恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler'):
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f"✅ 从Epoch {start_epoch}, Step {start_step} 恢复训练")
    
    # ========== 7. DDP包裹 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    Logger(f"\n{'='*70}\n 开始训练\n{'='*70}\n")
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0:
            Logger(f"⏭️ Epoch {epoch+1} 跳过前 {skip} batches")
        
        train_epoch(epoch, loader, len(loader), start_step=skip, wandb=wandb)
        start_step = 0  # 后续epoch从头开始
    
    # ========== 9. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
    
    Logger("\n✅ 训练完成！\n")