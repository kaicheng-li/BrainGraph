"""
LoRA微调（对标MiniMind）
✅ argparse + DDP + Checkpoint + wandb
✅ 基于Graph+LLM SFT模型
✅ 只训练LoRA参数
"""
import argparse
import os
import sys
import warnings
import time
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.llama_config import LlamaConfig
from model.lora import LoRALinear
from datasets.graph_sft_dataset import GraphSFTDataset
from datasets.data_utils import collate_graph_batch
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler, apply_lora
)

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """训练一个epoch"""
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        graph_data = {
            "node_features": batch['node_features'].to(args.device),
            "edge_index": batch['edge_index'].to(args.device),
            "batch": batch.get('batch_map', torch.zeros(batch['node_features'].size(0), dtype=torch.long)).to(args.device)
        }
        
        # 动态学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播
        with autocast_ctx:
            outputs = model(input_ids, labels=labels, graph_data=graph_data)
            loss = outputs[0]
            loss = loss / args.accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
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
                wandb.log({'loss': current_loss, 'lr': current_lr})
        
        # 保存checkpoint
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # 保存LoRA权重
            lora_state = {}
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    lora_state[name] = {
                        'lora_A': module.lora_A.data.cpu(),
                        'lora_B': module.lora_B.data.cpu()
                    }
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
            torch.save(lora_state, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints')
            model.train()
        
        del input_ids, labels, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning (MiniMind Style)")
    
    # === 目录和保存 ===
    parser.add_argument("--save_dir", type=str, default="out/lora")
    parser.add_argument('--save_weight', default='lora_adapter', type=str)
    
    # === 训练配置 ===
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=50)
    
    # === 模型配置 ===
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    
    # === LoRA配置 ===
    parser.add_argument('--lora_rank', default=8, type=int, help="LoRA秩")
    parser.add_argument('--lora_alpha', default=16, type=int, help="LoRA缩放因子")
    
    # === 数据 ===
    parser.add_argument("--data_path", type=str, default="./data/graph_sft/")
    
    # === 权重和续训 ===
    parser.add_argument('--from_weight', default='graph_llama_sft', type=str, help="基于Graph SFT模型")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    
    # === wandb ===
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MyTransformer-LoRA")
    
    # === torch.compile ===
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    
    args = parser.parse_args()
    
    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 从graph_llama_sft.pt加载config
    sft_ckpt_path = f"out/{args.from_weight}_{args.hidden_size}.pth"
    if os.path.exists(sft_ckpt_path):
        Logger(f"🔄 从 {sft_ckpt_path} 加载配置")
        sft_ckpt = torch.load(sft_ckpt_path, map_location='cpu')
        # 假设保存时包含config
        lm_config = LlamaConfig(**sft_ckpt) if isinstance(sft_ckpt, dict) else LlamaConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_graph=True
        )
    else:
        lm_config = LlamaConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_graph=True
        )
    
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume == 1 else None
    
    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_lib
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb = wandb_lib.init(project=args.wandb_project, id=wandb_id, resume=resume)
    
    # ========== 5. 模型和LoRA ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 应用LoRA
    lora_count = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    
    # 冻结主模型，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in lora_params)
    total = sum(p.numel() for p in model.parameters())
    Logger(f"📊 LoRA参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    if args.use_compile == 1:
        model = torch.compile(model)
    
    train_ds = GraphSFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=0.01)
    
    # ========== 6. 从checkpoint恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        if ckp_data.get('scaler'):
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 训练 ==========
    Logger(f"\n{'='*70}\n 开始LoRA训练\n{'='*70}\n")
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collate_graph_batch,
                          num_workers=args.num_workers, pin_memory=True)
        
        train_epoch(epoch, loader, len(loader), lora_params, start_step=skip, wandb=wandb)
        start_step = 0
    
    # ========== 9. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
    
    Logger("\n✅ LoRA训练完成！\n")