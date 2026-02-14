"""
训练工具函数（完全对标MiniMind）
包含：DDP初始化、Logger、学习率调度、Checkpoint管理、模型初始化等
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import uuid
import random
import numpy as np
from model.lora import LoRALinear
import torch.nn as nn
from model.llama_model import LlamaForCausalLM, GraphLlamaForCausalLM
from model.tokenizer import get_tokenizer

# ========== 日志工具 ==========
def Logger(msg):
    """统一日志输出（只在主进程打印）"""
    if is_main_process():
        print(msg)

def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

# ========== 分布式训练初始化 ==========
def init_distributed_mode():
    """
    初始化DDP分布式训练
    返回: local_rank (int)
    
    使用方法:
        # 单卡: python train.py
        # 多卡: torchrun --nproc_per_node=4 train.py
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
        Logger(f"✅ DDP初始化: rank {rank}/{world_size}, local_rank {local_rank}")
        return local_rank
    else:
        Logger("📱 单GPU模式")
        return 0

# ========== 随机种子 ==========
def setup_seed(seed):
    """设置所有随机种子（确保可复现）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== 学习率调度 ==========
def get_lr(step, total_steps, base_lr, min_lr=0.0, warmup_steps=0):
    """
    Warmup + Cosine Decay 学习率调度
    
    Args:
        step: 当前步数
        total_steps: 总步数
        base_lr: 初始学习率
        min_lr: 最小学习率
        warmup_steps: warmup步数
    """
    if step < warmup_steps:
        # Warmup阶段：线性增长
        return base_lr * step / max(1, warmup_steps)
    else:
        # Cosine衰减阶段
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

# ========== Checkpoint管理 ==========
def lm_checkpoint(config, weight='pretrain', model=None, optimizer=None, scaler=None, 
                  epoch=None, step=None, wandb=None, save_dir='./checkpoints'):
    """
    统一的Checkpoint管理函数（完全对标MiniMind）
    
    两种模式:
    1. 保存模式: 传入model/optimizer等，保存checkpoint
    2. 加载模式: 只传入config和weight，返回checkpoint数据
    
    Args:
        config: 模型配置
        weight: 权重名称（如'pretrain', 'graph_llama_sft'等）
        model: 模型（保存时需要）
        optimizer: 优化器（保存时需要）
        scaler: GradScaler（保存时需要）
        epoch: 当前epoch
        step: 当前step
        wandb: wandb实例
        save_dir: checkpoint保存目录
    
    Returns:
        None (保存模式) 或 checkpoint字典 (加载模式)
    """
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f"{save_dir}/{weight}_resume.pt"
    
    if model is None:
        # ========== 加载模式 ==========
        if os.path.exists(ckp_path):
            Logger(f"✅ 检测到checkpoint: {ckp_path}")
            ckp = torch.load(ckp_path, map_location='cpu')
            return ckp
        else:
            Logger(f"⚠️ 未找到checkpoint: {ckp_path}")
            return None
    else:
        # ========== 保存模式 ==========
        from torch.nn.parallel import DistributedDataParallel
        
        # 获取原始模型（去除DDP/compile包装）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 去除torch.compile包装
        
        ckp_data = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'step': step,
            'config': config.__dict__,
            'wandb_id': wandb.id if wandb else None,
        }
        
        torch.save(ckp_data, ckp_path)
        Logger(f"💾 Checkpoint已保存: {ckp_path} (Epoch {epoch}, Step {step})")
        return None

# ========== 跳过已训练batch的Sampler ==========
class SkipBatchSampler(Sampler):
    """
    续训时跳过已训练的batch
    
    使用场景:
        从checkpoint恢复训练时，跳过当前epoch已训练的步数
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        Args:
            sampler: 原始sampler（可以是DistributedSampler或indices列表）
            batch_size: batch大小
            skip_batches: 跳过的batch数量
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skip_count = 0
        
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skip_count < self.skip_batches:
                    skip_count += 1
                    batch = []
                    continue
                yield batch
                batch = []
        
        # 最后一个不完整batch
        if len(batch) > 0 and skip_count >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

# ========== 模型初始化 ==========
def init_model(config, from_weight='none', device='cuda'):
    """
    统一的模型初始化函数（完全对标MiniMind）
    
    功能:
    1. 根据config决定使用LlamaForCausalLM还是GraphLlamaForCausalLM
    2. 加载预训练权重（如果from_weight != 'none'）
    3. 初始化tokenizer
    
    Args:
        config: LlamaConfig实例
        from_weight: 权重名称，如'llama_pretrain', 'graph_llama_sft'等
        device: 设备
    
    Returns:
        (model, tokenizer)
    """
    
    # ========== 1. 根据config.use_graph选择模型 ==========
    if getattr(config, 'use_graph', False):
        Logger("🔧 创建 GraphLlamaForCausalLM (含Graph模块)")
        model = GraphLlamaForCausalLM(config)
    else:
        Logger("🔧 创建 LlamaForCausalLM (纯LLM)")
        model = LlamaForCausalLM(config)
    
    # ========== 2. 初始化tokenizer ==========
    tokenizer_type = getattr(config, 'tokenizer_type', 'tiktoken')
    tokenizer_name = getattr(config, 'tokenizer_name', 'gpt-4')
    tokenizer = get_tokenizer(tokenizer_type, tokenizer_name)
    
    # 更新config的vocab_size（以tokenizer实际大小为准）
    config.vocab_size = tokenizer.vocab_size
    
    # ========== 3. 加载预训练权重 ==========
    if from_weight != 'none':
        # 尝试多种路径格式（对标MiniMind的灵活性）
        possible_paths = [
            f"out/{from_weight}_{config.hidden_size}.pth",
            f"out/{from_weight}_{config.hidden_size}_moe.pth" if getattr(config, 'use_moe', False) else None,
            f"checkpoints/{from_weight}.pt",
            f"checkpoints/{from_weight}_{config.hidden_size}.pt",
            f"{from_weight}",  # 支持直接传入完整路径
        ]
        
        weight_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                weight_path = path
                break
        
        if weight_path:
            Logger(f"🔄 加载权重: {weight_path}")
            ckpt = torch.load(weight_path, map_location='cpu')
            
            # 兼容多种checkpoint格式
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            
            # 加载权重（strict=False允许部分加载）
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                Logger(f"⚠️ 缺失的key: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                Logger(f"⚠️ 多余的key: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
            Logger("✅ 权重加载成功")
        else:
            Logger(f"⚠️ 未找到权重 '{from_weight}'，从头开始训练")
    
    # ========== 4. 移动到设备 ==========
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    Logger(f"📐 模型参数: {total_params/1e6:.2f}M")
    
    return model, tokenizer

# ========== 保存LoRA权重（专用函数） ==========
def save_lora(model, save_path):
    """
    保存LoRA权重（对标MiniMind的save_lora函数）
    
    Args:
        model: 应用了LoRA的模型
        save_path: 保存路径
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                'lora_A': module.lora_A.data.cpu(),
                'lora_B': module.lora_B.data.cpu(),
            }
    
    if lora_state:
        torch.save(lora_state, save_path)
        Logger(f"💾 LoRA权重已保存: {save_path} ({len(lora_state)} 层)")
    else:
        Logger("⚠️ 未找到LoRA层，无法保存")

# ========== 应用LoRA（专用函数） ==========
def apply_lora(model, rank=8, alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']):
    """
    给模型应用LoRA（对标MiniMind的apply_lora函数）
    
    Args:
        model: 要应用LoRA的模型
        rank: LoRA秩
        alpha: LoRA缩放因子
        target_modules: 要应用LoRA的模块名称列表
    """
    lora_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            is_target = any(target in name for target in target_modules)
            
            if is_target:
                # 获取父模块和属性名
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                # 创建LoRA层
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    bias=module.bias is not None
                )
                
                # 复制原始权重
                lora_layer.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data.clone()
                
                # 替换模块
                parent = model
                for attr in parent_name.split('.'):
                    if attr:
                        parent = getattr(parent, attr)
                setattr(parent, attr_name, lora_layer)
                
                lora_count += 1
    
    Logger(f"✅ LoRA应用成功: {lora_count} 层 (rank={rank}, alpha={alpha})")
    return lora_count

# ========== 统计LoRA参数 ==========
def count_lora_parameters(model):
    """
    统计LoRA参数数量
    
    Returns:
        (lora_params, total_params, lora_ratio)
    """
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name.lower())
    lora_ratio = lora_params / total_params * 100 if total_params > 0 else 0
    
    return lora_params, total_params, lora_ratio

# ========== 获取可训练参数列表 ==========
def get_trainable_params(model, mode='all'):
    """
    获取可训练参数列表
    
    Args:
        model: 模型
        mode: 'all' | 'lora' | 'non_lora'
    
    Returns:
        参数列表
    """
    if mode == 'all':
        return [p for p in model.parameters() if p.requires_grad]
    elif mode == 'lora':
        return [p for name, p in model.named_parameters() if p.requires_grad and 'lora' in name.lower()]
    elif mode == 'non_lora':
        return [p for name, p in model.named_parameters() if p.requires_grad and 'lora' not in name.lower()]
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ========== 冻结/解冻参数 ==========
def freeze_params(model, exclude_patterns=None):
    """
    冻结模型参数
    
    Args:
        model: 模型
        exclude_patterns: 排除的模式列表（包含这些字符串的参数不冻结）
    """
    exclude_patterns = exclude_patterns or []
    
    frozen_count = 0
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        should_freeze = not any(pattern in name for pattern in exclude_patterns)
        
        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            unfrozen_count += 1
    
    Logger(f"🔒 参数冻结: {frozen_count} 个冻结, {unfrozen_count} 个可训练")