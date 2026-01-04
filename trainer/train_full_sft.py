"""
MiniMind 全参数监督微调（Full SFT）脚本

本脚本实现了语言模型的监督微调流程，是 MiniMind 训练的第二阶段。

监督微调目标:
- 让模型学会遵循指令
- 学习对话格式和回复风格
- 提升模型的任务完成能力

与预训练的区别:
1. 数据格式: 使用对话格式（user/assistant）而非纯文本
2. 损失计算: 只计算 assistant 回复部分的损失
3. 学习率: 通常比预训练小 10-100 倍
4. 训练轮数: 通常 1-3 轮

主要特性:
1. 支持单 GPU 和多 GPU 分布式训练（DDP）
2. 支持混合精度训练（bfloat16/float16）
3. 支持梯度累积
4. 支持断点续训
5. 支持 wandb/swanlab 日志记录
6. 支持 MoE（混合专家）架构

使用方法:
    # 单 GPU 训练
    python train_full_sft.py --epochs 2 --batch_size 16
    
    # 多 GPU 分布式训练
    torchrun --nproc_per_node=4 train_full_sft.py --epochs 2 --batch_size 16
    
    # 基于预训练权重微调
    python train_full_sft.py --from_weight pretrain
"""

import os
import sys

# 设置包名以支持相对导入
__package__ = "trainer"
# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, total_tokens=0):
    """
    训练一个 epoch
    
    执行完整的 SFT 训练循环，与预训练的主要区别是:
    - 使用 SFTDataset 提供的 loss_mask，只计算 assistant 回复的损失
    - 学习率通常更小
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        total_tokens (int): 已处理的 token 总数
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用交叉熵损失，通过 loss_mask 只计算 assistant 回复
    - 使用余弦退火学习率调度
    - 支持梯度累积以模拟更大的批次
    - 使用梯度裁剪防止梯度爆炸
    """
    # 定义损失函数（不进行 reduction，以便应用 loss_mask）
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    # 遍历数据批次
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将数据移动到 GPU
        X = X.to(args.device)           # 输入 token ID
        Y = Y.to(args.device)           # 目标 token ID
        loss_mask = loss_mask.to(args.device)  # 损失掩码（只在 assistant 回复处为 1）
        
        # 计算当前学习率（余弦退火）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（使用混合精度）
        with autocast_ctx:
            res = model(X)
            
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # (batch*seq, vocab)
                Y.view(-1)                                   # (batch*seq,)
            ).view(Y.size())  # 恢复形状 (batch, seq)

            # 应用 loss_mask（只计算 assistant 回复的损失）
            # 这是 SFT 与预训练的关键区别
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 添加 MoE 辅助损失（如果使用 MoE）
            loss += res.aux_loss
            
            # 梯度累积：将损失除以累积步数
            loss = loss / args.accumulation_steps

        # 反向传播（使用梯度缩放器处理混合精度）
        scaler.scale(loss).backward()

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放
            scaler.unscale_(optimizer)
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 清理 GPU 缓存
            torch.cuda.empty_cache()

        # 统计本 step 的 token 数（按 loss_mask 计算有效 token）
        step_tokens = loss_mask.sum().item()
        if dist.is_initialized():
            # 分布式训练时乘以 GPU 数量
            step_tokens *= dist.get_world_size()
        total_tokens += step_tokens

        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            now_time = time.time()
            spend_time = now_time - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 计算速度: it/s 和 token/s（以两次日志之间的时间窗口为准）
            interval_tokens = total_tokens - last_log_tokens
            interval_time = max(now_time - last_log_time, 1e-6)
            interval_steps = max(step - last_log_step, 1)
            iter_per_sec = interval_steps / interval_time
            tokens_per_sec = interval_tokens / interval_time
            
            # 更新日志时间点
            last_log_time = now_time
            last_log_tokens = total_tokens
            last_log_step = step
            
            # 打印日志
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} lr:{current_lr:.12f} '
                f'epoch_Time:{eta_min}min it/s:{iter_per_sec:.2f} '
                f'token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}'
            )
            
            # 记录到 wandb
            if wandb: 
                wandb.log({
                    "loss": current_loss,
                    "lr": current_lr,
                    "epoch_Time": eta_min,
                    "it_per_sec": iter_per_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "total_tokens": total_tokens
                })

        # 保存检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式
            
            # 构建检查点路径
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 半精度保存以节省空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # 保存完整检查点（包含优化器状态等）
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints', 
                scaler=scaler,
                total_tokens=total_tokens
            )
            
            model.train()  # 切换回训练模式
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=2, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, 
                        help="初始学习率")
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, 
                        help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, 
                        help="模型保存间隔")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, 
                        help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                        help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, 
                        help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重参数
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", 
                        help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, 
                        help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否自动检测&续训（0=否，1=是）")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", 
                        help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 使用绝对路径避免 HuggingFace 路径验证问题
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # 初始化模型和 tokenizer（基于预训练权重）
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    
    # 创建 SFT 数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    total_tokens = 0
    
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        total_tokens = ckp_data.get('total_tokens', 0)
    
    # ========== 7. DDP 包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点：跳过已训练的步数
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_tokens = train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb, total_tokens)
        else:
            # 默认从头开始
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            total_tokens = train_epoch(epoch, loader, len(loader), 0, wandb, total_tokens)
