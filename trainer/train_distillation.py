"""
MiniMind 知识蒸馏训练脚本

本脚本实现了知识蒸馏（Knowledge Distillation）训练，让小模型学习大模型的知识。

知识蒸馏原理:
知识蒸馏是一种模型压缩技术，通过让较小的学生模型学习较大的教师模型的输出分布，
在保持性能的同时减少模型参数量和计算成本。

核心思想:
- 使用大模型（教师）生成的软标签（soft labels）指导小模型（学生）训练
- 软标签包含更多信息（概率分布）而非硬标签（one-hot）
- 通过温度缩放（temperature scaling）软化概率分布

蒸馏损失函数:
L = α * L_CE + (1 - α) * L_KL

其中:
- L_CE: 交叉熵损失（学生模型预测与真实标签）
- L_KL: KL 散度损失（学生模型与教师模型的输出分布）
- α: 平衡系数（通常 0.5）

温度缩放:
- 使用温度参数 T > 1 软化概率分布
- 温度越高，分布越平滑，包含更多信息
- 最终损失需要乘以 T² 进行缩放

优势:
1. 模型压缩: 小模型可以接近大模型的性能
2. 知识传递: 软标签包含更多信息
3. 训练稳定: 软标签提供更平滑的梯度

使用方法:
    python train_distillation.py --epochs 6 --batch_size 32 --alpha 0.5 --temperature 1.5
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
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息
warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏损失（KL 散度）
    
    使用温度缩放软化概率分布，然后计算学生模型与教师模型之间的 KL 散度。
    
    Args:
        student_logits: 学生模型的 logits，形状为 [batch*seq, vocab_size]
        teacher_logits: 教师模型的 logits，形状为 [batch*seq, vocab_size]
        temperature: 温度参数（> 1），用于软化概率分布
        reduction: 损失归约方式（'batchmean' 表示对 batch 求平均）
        
    Returns:
        loss: 蒸馏损失（已乘以 T² 进行缩放）
        
    计算过程:
    1. 对教师模型 logits 应用 softmax（带温度缩放）得到软标签
    2. 对学生模型 logits 应用 log_softmax（带温度缩放）
    3. 计算 KL 散度
    4. 乘以 T² 进行缩放（因为温度缩放改变了损失尺度）
    """
    # 计算教师模型的软标签（不计算梯度）
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 计算学生模型的 log 概率（带温度缩放）
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度: KL(student || teacher)
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    
    # 乘以 T² 进行缩放（温度缩放改变了损失尺度）
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0, total_tokens=0):
    """
    训练一个 epoch（知识蒸馏版本）
    
    知识蒸馏训练的特点:
    - 同时使用真实标签（hard labels）和教师模型软标签（soft labels）
    - 通过平衡系数 alpha 控制两种损失的权重
    - 教师模型保持冻结，只用于生成软标签
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        teacher_model: 教师模型（冻结，用于生成软标签）
        lm_config_student: 学生模型配置
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        alpha (float): 交叉熵损失权重（总损失 = α * CE + (1-α) * KL）
        temperature (float): 温度参数（用于软化概率分布）
        total_tokens (int): 已处理的 token 总数
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用混合损失：交叉熵损失 + 蒸馏损失
    - 教师模型和学生模型的词汇表大小可能不同，需要对齐
    - 使用梯度累积以模拟更大的批次
    - 使用梯度裁剪防止梯度爆炸
    """
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    # 确保教师模型处于评估模式且不计算梯度
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将数据移动到 GPU
        X = X.to(args.device)           # 输入 token ID
        Y = Y.to(args.device)           # 目标 token ID
        loss_mask = loss_mask.to(args.device)  # 损失掩码
        
        # 计算当前学习率（余弦退火）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with autocast_ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（只在 eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # 如果教师模型和学生模型的词汇表大小不同，需要截断对齐
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth 交叉熵损失（学生模型预测与真实标签）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        
        # 添加 MoE 辅助损失（如果使用 MoE）
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) 蒸馏损失（学生模型与教师模型的 KL 散度）
        if teacher_model is not None:
            # 只对有效位置（loss_mask == 1）计算蒸馏损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = α * CE + (1-α) * Distill
        # α 控制两种损失的权重：α=1 时只有 CE，α=0 时只有蒸馏
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

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
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
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
                f'loss:{current_loss:.6f} ce:{ce_loss.item():.4f} distill:{distill_loss.item():.4f} '
                f'lr:{current_lr:.12f} epoch_Time:{eta_min}min it/s:{iter_per_sec:.2f} '
                f'token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}'
            )
            
            # 记录到 wandb
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
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
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典（处理 DDP 情况）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 半精度保存以节省空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # 保存完整检查点（包含优化器、scaler 等）
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, 
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', total_tokens=total_tokens)
            
            model.train()  # 切换回训练模式
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=6, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
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
    parser.add_argument("--max_seq_len", type=int, default=512, 
                        help="训练的最大截断长度")
    parser.add_argument('--student_hidden_size', default=512, type=int, 
                        help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, 
                        help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, 
                        help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, 
                        help="教师模型隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重参数
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", 
                        help="训练数据路径")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, 
                        help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, 
                        help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否自动检测&续训（0=否，1=是）")
    
    # 蒸馏特定参数
    parser.add_argument('--alpha', default=0.5, type=float, 
                        help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL（通常0.3-0.7）")
    parser.add_argument('--temperature', default=1.5, type=float, 
                        help="蒸馏温度（推荐范围1.0-2.0，温度越高分布越平滑）")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", 
                        help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子（分布式训练时每个进程使用不同的种子）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建学生和教师模型配置
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size, 
        num_hidden_layers=args.student_num_layers, 
        use_moe=bool(args.use_moe)
    )
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size, 
        num_hidden_layers=args.teacher_num_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有可恢复的检查点
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 创建自动混合精度上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    # 使用绝对路径避免 HuggingFace 路径验证问题
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # 初始化学生模型（正在训练的模型）
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化教师模型（冻结，用于生成软标签）
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    
    # 创建数据集和优化器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        total_tokens = ckp_data.get('total_tokens', 0)
    
    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        # 忽略 RoPE 的 buffer（不需要同步）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DDP 包装模型
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch（确保每个 epoch 的数据顺序不同）
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点：跳过已训练的步数
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_tokens = train_epoch(epoch, loader, len(loader) + start_step + 1, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature, total_tokens)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            total_tokens = train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature, total_tokens)
