"""
MiniMind DPO（Direct Preference Optimization）训练脚本

本脚本实现了直接偏好优化算法，用于对齐模型输出与人类偏好。

DPO 原理:
DPO 是 RLHF 的简化替代方案，直接从偏好数据学习，无需训练奖励模型。

核心思想:
- 给定一个问题，有两个回答：chosen（好的）和 rejected（差的）
- 让模型学习偏好 chosen 回答，而不是 rejected 回答
- 通过对比学习的方式优化模型

DPO 损失函数:
L_DPO = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))

其中:
- y_w: 选中的回答 (chosen/winner)
- y_l: 拒绝的回答 (rejected/loser)
- π: 策略模型（正在训练的模型）
- π_ref: 参考模型（冻结的原始模型）
- β: 温度参数，控制偏好强度

与 RLHF 的区别:
1. 不需要训练奖励模型
2. 不需要 PPO 等复杂的 RL 算法
3. 训练更稳定，超参数更少
4. 计算效率更高

使用方法:
    python train_dpo.py --epochs 1 --batch_size 4 --learning_rate 4e-8
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
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将 logits 转换为 log 概率
    
    对于每个位置，计算目标 token 的 log 概率。
    
    Args:
        logits: 模型输出的 logits，形状为 (batch_size, seq_len, vocab_size)
        labels: 目标 token ID，形状为 (batch_size, seq_len)
        
    Returns:
        log_probs: 每个位置的 log 概率，形状为 (batch_size, seq_len)
        
    计算过程:
    1. 对 logits 应用 log_softmax 得到 log 概率分布
    2. 使用 gather 提取目标 token 的 log 概率
    """
    # 计算 log 概率分布
    log_probs = F.log_softmax(logits, dim=2)
    
    # 提取目标 token 的 log 概率
    # gather 操作: 从 log_probs 中按 labels 索引提取值
    log_probs_per_token = torch.gather(
        log_probs, 
        dim=2, 
        index=labels.unsqueeze(2)
    ).squeeze(-1)
    
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO 损失
    
    DPO 损失函数的核心实现，通过对比 chosen 和 rejected 回答的
    log 概率差异来优化模型。
    
    Args:
        ref_log_probs: 参考模型的 log 概率，形状为 (batch_size, seq_len)
        policy_log_probs: 策略模型的 log 概率，形状为 (batch_size, seq_len)
        mask: 损失掩码，形状为 (batch_size, seq_len)
        beta: 温度参数，控制偏好强度
        
    Returns:
        loss: DPO 损失值（标量）
        
    计算过程:
    1. 计算序列级别的平均 log 概率
    2. 分离 chosen 和 rejected 样本
    3. 计算 log ratio 差异
    4. 应用 sigmoid 损失
    
    注意:
    - batch 的前半部分是 chosen，后半部分是 rejected
    - 使用序列长度归一化防止长序列主导
    """
    # 计算序列长度（用于归一化）
    # 参考: https://github.com/jingyaogong/minimind/issues/298
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止除零
    
    # 计算序列级别的平均 log 概率
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    # 假设 batch 的前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算 log ratio
    # π_logratios = log π(y_w|x) - log π(y_l|x)
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # ref_logratios = log π_ref(y_w|x) - log π_ref(y_l|x)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # DPO 的核心: logits = π_logratios - ref_logratios
    logits = pi_logratios - ref_logratios
    
    # 应用 sigmoid 损失: -log(σ(β * logits))
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1, total_tokens=0):
    """
    训练一个 epoch（DPO 版本）
    
    DPO 训练的特点:
    - 需要同时运行策略模型和参考模型
    - 参考模型保持冻结，只用于计算基准 log 概率
    - 使用 DPO 损失而非交叉熵损失
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        ref_model: 参考模型（冻结）
        lm_config: 模型配置
        start_step (int): 起始步数
        wandb: wandb/swanlab 日志对象
        beta (float): DPO 温度参数
        total_tokens (int): 已处理的 token 总数
        
    Returns:
        int: 更新后的 token 总数
    """
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # 获取 chosen 和 rejected 数据
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        # 合并 chosen 和 rejected 数据
        # 这样可以一次前向传播处理两者
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 参考模型前向传播（不计算梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 策略模型前向传播
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 计算 DPO 损失
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # 统计本 step 的 token 数（按 mask 计算有效 token）
        step_tokens = mask.sum().item()
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
            
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} lr:{current_lr:.12f} '
                f'epoch_Time:{eta_min}min it/s:{iter_per_sec:.2f} '
                f'token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}'
            )
            
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
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints',
                total_tokens=total_tokens
            )
            
            model.train()
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, 
                        help="初始学习率（建议<=5e-8避免遗忘）")
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
    parser.add_argument('--max_seq_len', default=1024, type=int, 
                        help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重参数
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", 
                        help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, 
                        help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否自动检测&续训（0=否，1=是）")
    
    # DPO 特定参数
    parser.add_argument('--beta', default=0.1, type=float, 
                        help="DPO中的beta参数")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", 
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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # 策略模型（正在训练的模型）
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 参考模型（冻结，用于计算基准 log 概率）
    ref_model, _ = init_model(lm_config, args.from_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 创建数据集和优化器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
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
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_tokens = train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, lm_config, start_step, wandb, args.beta, total_tokens)
        else:
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            total_tokens = train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta, total_tokens)
