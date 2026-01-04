"""
MiniMind LoRA 微调脚本

本脚本实现了使用 LoRA（Low-Rank Adaptation）进行参数高效微调。

LoRA 原理:
- 冻结预训练模型的所有参数
- 在注意力层的 Q、K、V 投影旁添加低秩分解矩阵
- 只训练这些低秩矩阵（通常 < 1% 的参数）
- 训练完成后可以将 LoRA 权重合并到原始模型

LoRA 优势:
1. 参数高效: 只需训练很少的参数
2. 内存高效: 不需要存储完整的梯度
3. 快速训练: 训练速度更快
4. 可插拔: 可以轻松切换不同的 LoRA 适配器
5. 无推理延迟: 可以将权重合并到原始模型

适用场景:
- 个性化定制（如身份认同）
- 领域适配（如医疗、法律）
- 风格迁移
- 资源受限的微调

使用方法:
    # 训练身份认同 LoRA
    python train_lora.py --lora_name lora_identity --data_path ../dataset/lora_identity.jsonl
    
    # 训练医疗领域 LoRA
    python train_lora.py --lora_name lora_medical --data_path ../dataset/lora_medical.jsonl
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
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None, total_tokens=0):
    """
    训练一个 epoch（LoRA 版本）
    
    与全参数微调的区别:
    - 只更新 LoRA 参数，原始模型参数保持冻结
    - 梯度裁剪只应用于 LoRA 参数
    - 保存时只保存 LoRA 权重
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        lora_params (list): LoRA 参数列表
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        total_tokens (int): 已处理的 token 总数
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用交叉熵损失
    - 只对 LoRA 参数进行梯度裁剪
    - 使用余弦退火学习率调度
    """
    # 定义损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    # 遍历数据批次
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将数据移动到 GPU
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with autocast_ctx:
            res = model(X)
            
            # 计算损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 应用 loss_mask
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            
            # 只对 LoRA 参数进行梯度裁剪
            # 这是 LoRA 训练的关键区别
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
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
            
            # 保存 LoRA 权重（只保存 LoRA 参数，文件很小）
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            save_lora(model, lora_save_path)
            
            # 保存完整检查点（用于断点续训）
            lm_checkpoint(
                lm_config, 
                weight=args.lora_name, 
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
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out/lora", 
                        help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", 
                        help="LoRA权重名称(如lora_identity/lora_medical等)")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
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
    parser.add_argument("--log_interval", type=int, default=10, 
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, 
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
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", 
                        help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, 
                        help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否自动检测&续训（0=否，1=是）")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", 
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
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
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
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用 LoRA、冻结非 LoRA 参数 ==========
    # 使用绝对路径避免 HuggingFace 路径验证问题
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # 初始化模型（基于 full_sft 权重）
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    
    # 应用 LoRA 到模型
    # 这会在 Q、K、V 投影层添加 LoRA 适配器
    apply_lora(model)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非 LoRA 参数，收集 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # LoRA 参数需要梯度
            param.requires_grad = True
            lora_params.append(param)
        else:
            # 原始模型参数冻结
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 优化器只优化 LoRA 参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    total_tokens = 0
    if ckp_data:
        # 使用 strict=False 因为 LoRA 参数可能不在原始检查点中
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        total_tokens = ckp_data.get('total_tokens', 0)
    
    # ========== 8. DDP 包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 断点续训
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_tokens = train_epoch(epoch, loader, len(loader) + start_step + 1, lora_params, start_step, wandb, total_tokens)
        else:
            # 从头开始
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            total_tokens = train_epoch(epoch, loader, len(loader), lora_params, 0, wandb, total_tokens)
