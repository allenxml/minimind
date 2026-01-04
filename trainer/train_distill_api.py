"""
MiniMind 在线 API 蒸馏训练脚本

本脚本实现了在线 API 蒸馏训练,在训练时实时调用教师模型 API。

警告: 
- 在线 API 蒸馏会显著降低训练速度(需要等待 API 响应)
- 建议使用离线模式:先用 generate_distill_data_from_api.py 生成数据,再用标准蒸馏脚本训练
- 在线模式适合小规模实验或对实时性有特殊要求的场景

训练流程:
1. 从数据集加载输入
2. 实时调用 OpenRouter API 获取教师模型的响应
3. 使用教师模型的 logits 或输出文本进行蒸馏

支持两种蒸馏模式:
- text_distill: 基于文本的蒸馏(使用生成的文本作为标签)
- logits_distill: 基于 logits 的蒸馏(需要 API 支持返回 logprobs)

使用方法:
    python train_distill_api.py \
        --api_key YOUR_OPENROUTER_KEY \
        --teacher_model openai/gpt-4 \
        --data_path ../dataset/sft_mini_512.jsonl \
        --distill_mode text_distill \
        --epochs 3 \
        --batch_size 4
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
import requests
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
from typing import List, Dict, Optional

# 忽略警告信息
warnings.filterwarnings('ignore')


class OpenRouterClient:
    """OpenRouter API 客户端(简化版)"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jingyaogong/minimind",
            "X-Title": "MiniMind Online Distillation"
        }
    
    def get_teacher_response(
        self, 
        model: str, 
        input_ids: torch.Tensor,
        tokenizer,
        temperature: float = 0.7,
        max_tokens: int = 512,
        return_logprobs: bool = False
    ) -> tuple:
        """
        获取教师模型的响应
        
        Args:
            model: 模型名称
            input_ids: 输入 token IDs
            tokenizer: tokenizer
            temperature: 温度参数
            max_tokens: 最大生成长度
            return_logprobs: 是否返回 log 概率
            
        Returns:
            (生成的文本, log概率或None)
        """
        # 解码输入
        input_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        
        # 构建消息
        messages = [{"role": "user", "content": input_text}]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 如果需要 logprobs(仅部分模型支持)
        if return_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            choice = result['choices'][0]
            content = choice['message']['content']
            
            # 提取 logprobs(如果有)
            logprobs = None
            if return_logprobs and 'logprobs' in choice:
                logprobs = choice['logprobs']
            
            return content, logprobs
            
        except Exception as e:
            Logger(f"API 调用失败: {e}")
            return "", None


def text_based_distill_loss(student_logits, teacher_text_ids, mask):
    """
    基于文本的蒸馏损失
    
    使用教师模型生成的文本作为标签,计算学生模型的交叉熵损失。
    
    Args:
        student_logits: 学生模型的 logits [batch, seq, vocab]
        teacher_text_ids: 教师模型生成的文本 token IDs [batch, seq]
        mask: 损失掩码 [batch, seq]
        
    Returns:
        损失值
    """
    loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_text_ids.view(-1),
        ignore_index=0,
        reduction='none'
    )
    mask_flat = mask.view(-1)
    loss = torch.sum(loss * mask_flat) / (mask_flat.sum() + 1e-8)
    return loss


def train_epoch(epoch, loader, iters, api_client, teacher_model_name, tokenizer, lm_config, 
                start_step=0, wandb=None, distill_mode='text_distill', api_cache=None, total_tokens=0):
    """
    训练一个 epoch(在线 API 蒸馏版本)
    
    警告: 每个 batch 都需要调用 API,训练速度会很慢!
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        api_client: OpenRouter API 客户端
        teacher_model_name (str): 教师模型名称
        tokenizer: tokenizer
        lm_config: 模型配置
        start_step (int): 起始步数
        wandb: wandb/swanlab 日志对象
        distill_mode (str): 蒸馏模式('text_distill')
        api_cache (dict): API 响应缓存(可选,用于加速)
        total_tokens (int): 已处理的 token 总数
        
    Returns:
        int: 更新后的 token 总数
    """
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    # API 调用统计
    api_call_count = 0
    api_cache_hit = 0
    
    if api_cache is None:
        api_cache = {}
    
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将数据移动到 GPU
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ========== 调用教师模型 API ==========
        batch_size = X.size(0)
        teacher_texts = []
        
        for i in range(batch_size):
            # 生成缓存键(基于输入序列)
            cache_key = tuple(X[i].tolist())
            
            if cache_key in api_cache:
                # 缓存命中
                teacher_text = api_cache[cache_key]
                api_cache_hit += 1
            else:
                # 调用 API
                teacher_text, _ = api_client.get_teacher_response(
                    model=teacher_model_name,
                    input_ids=X[i:i+1],
                    tokenizer=tokenizer,
                    temperature=args.api_temperature,
                    max_tokens=args.max_seq_len,
                    return_logprobs=False
                )
                api_cache[cache_key] = teacher_text
                api_call_count += 1
                
                # API 速率限制
                if api_call_count % 5 == 0:
                    time.sleep(0.5)
            
            teacher_texts.append(teacher_text)
        
        # 将教师文本编码为 token IDs
        teacher_text_ids = []
        for text in teacher_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            # 截断或填充到目标长度
            target_len = Y.size(1)
            if len(ids) > target_len:
                ids = ids[:target_len]
            else:
                ids = ids + [0] * (target_len - len(ids))
            teacher_text_ids.append(ids)
        
        teacher_text_ids = torch.tensor(teacher_text_ids, dtype=torch.long).to(args.device)
        
        # ========== 前向传播和损失计算 ==========
        with autocast_ctx:
            res = model(X)
            student_logits = res.logits
            
            # 基于文本的蒸馏损失
            if distill_mode == 'text_distill':
                loss = text_based_distill_loss(student_logits, teacher_text_ids, loss_mask)
            else:
                raise ValueError(f"不支持的蒸馏模式: {distill_mode}")
            
            # 添加 MoE 辅助损失
            if lm_config.use_moe:
                loss += res.aux_loss
            
            # 梯度累积
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
        
        # 统计本 step 的 token 数
        step_tokens = loss_mask.sum().item()
        if dist.is_initialized():
            step_tokens *= dist.get_world_size()
        total_tokens += step_tokens
        
        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            now_time = time.time()
            spend_time = now_time - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 计算速度
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
                f'API_calls:{api_call_count} cache_hits:{api_cache_hit} '
                f'epoch_Time:{eta_min}min it/s:{iter_per_sec:.2f} '
                f'token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}'
            )
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "lr": current_lr,
                    "api_calls": api_call_count,
                    "api_cache_hits": api_cache_hit,
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
            
            # 同时保存 API 缓存
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, 
                         save_dir='../checkpoints', total_tokens=total_tokens,
                         extra_data={'api_cache': api_cache})
            
            model.train()
    
    return total_tokens, api_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Online API Distillation")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out",
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='api_distill', type=str,
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size(建议小一点,API调用很慢)")
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
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="模型保存间隔")
    
    # 模型架构参数
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="训练的最大截断长度")
    parser.add_argument('--hidden_size', default=512, type=int,
                        help="学生模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int,
                        help="学生模型隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="是否使用MoE架构")
    
    # API 相关参数
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenRouter API 密钥")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="API 基础 URL")
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="教师模型名称(如 'openai/gpt-4')")
    parser.add_argument("--api_temperature", type=float, default=0.7,
                        help="API 调用温度参数")
    parser.add_argument("--distill_mode", type=str, default='text_distill',
                        choices=['text_distill'],
                        help="蒸馏模式")
    
    # 数据和权重参数
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl",
                        help="训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str,
                        help="学生模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1],
                        help="是否自动检测&续训")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-API-Distillation",
                        help="wandb项目名")
    
    args = parser.parse_args()
    
    # ========== 警告提示 ==========
    print("=" * 70)
    print("警告: 在线 API 蒸馏训练速度非常慢!")
    print("建议使用离线模式:")
    print("1. 使用 generate_distill_data_from_api.py 预先生成数据")
    print("2. 使用标准蒸馏训练脚本进行训练")
    print("=" * 70)
    
    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录和模型 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None
    
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
        wandb_run_name = f"MiniMind-API-Distill-{args.teacher_model.replace('/', '-')}-Epoch-{args.epochs}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=tokenizer_path, 
                                   save_dir=args.save_dir, device=args.device)
    
    Logger(f'学生模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 创建 API 客户端
    api_client = OpenRouterClient(api_key=args.api_key, base_url=args.base_url)
    Logger(f'教师模型: {args.teacher_model} (通过 OpenRouter API)')
    
    # 创建数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    total_tokens = 0
    api_cache = {}
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        total_tokens = ckp_data.get('total_tokens', 0)
        api_cache = ckp_data.get('extra_data', {}).get('api_cache', {})
        Logger(f'恢复 API 缓存: {len(api_cache)} 条')
    
    # ========== 7. DDP 包装 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), 
                                            args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, 
                              num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step')
            total_tokens, api_cache = train_epoch(epoch, loader, len(loader) + start_step + 1, 
                                                  api_client, args.teacher_model, tokenizer, 
                                                  lm_config, start_step, wandb, args.distill_mode, 
                                                  api_cache, total_tokens)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
            total_tokens, api_cache = train_epoch(epoch, loader, len(loader), api_client, 
                                                  args.teacher_model, tokenizer, lm_config, 
                                                  0, wandb, args.distill_mode, api_cache, total_tokens)

