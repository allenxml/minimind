"""
MiniMind SPO（Self-Play Optimization）训练脚本

本脚本实现了自对弈优化算法，用于强化学习对齐。

SPO 原理:
SPO 是一种基于自适应价值追踪的强化学习算法，通过维护一个动态基线来稳定训练。

核心思想:
- 使用自适应价值追踪器（AutoAdaptiveValueTracker）维护基线
- 基线通过 Beta 分布参数（alpha, beta）动态更新
- 使用 KL 散度自适应调整更新速率（rho）
- 优势 = 奖励 - 基线（而非组内相对优势）

SPO 损失函数:
L = -E[log π(y|x) * advantages + β * KL(π||π_ref)]

其中:
- advantages: 奖励 - 自适应基线
- β: KL 惩罚系数
- π_ref: 参考模型（冻结）

自适应价值追踪:
- 使用 Beta 分布参数 alpha 和 beta 维护基线
- 基线 = alpha / (alpha + beta)
- 根据 KL 散度自适应调整更新速率 rho
- rho 控制基线的更新速度（防止基线变化过快）

与 GRPO 的区别:
1. 使用自适应基线而非组内相对优势
2. 基线跨 batch 更新，提供更稳定的估计
3. 使用 KL 散度自适应调整更新速率

使用方法:
    python train_spo.py --epochs 1 --batch_size 2 --learning_rate 1e-7
"""

import os
import sys

# 设置包名以支持相对导入
__package__ = "trainer"
# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import time
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息
warnings.filterwarnings('ignore')


class AutoAdaptiveValueTracker:
    """
    SPO 自适应价值追踪器
    
    使用 Beta 分布参数（alpha, beta）维护动态基线，用于估计响应价值。
    基线会根据奖励和策略变化自适应更新。
    
    原理:
    - 基线 = alpha / (alpha + beta)
    - 根据奖励更新 alpha 和 beta
    - 使用 rho 控制更新速率（防止基线变化过快）
    - rho 可以根据 KL 散度自适应调整
    
    参数:
    - rho_mode: 更新模式（'kl' 或 'constant'）
    - rho_const: 常数更新速率（当 rho_mode='constant' 时使用）
    - D_half: KL 散度半衰期（用于计算自适应 rho）
    - clip_lower/clip_upper: rho 的裁剪范围
    """
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode  # 更新模式
        self.rho_const = rho_const  # 常数更新速率
        self.D_half = D_half  # KL 散度半衰期
        self.clip_lower = clip_lower  # rho 下界
        self.clip_upper = clip_upper  # rho 上界
        
        # 初始化 Beta 分布参数
        # N_init 确保初始基线在合理范围内
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init  # Beta 分布参数 alpha
        self.beta = 0.5 * N_init   # Beta 分布参数 beta
        self.old_mean_logprob = None  # 上一次的平均 log 概率（用于计算 KL）

    def get_baselines(self, batch_size):
        """
        获取当前基线值
        
        Args:
            batch_size: 批次大小
            
        Returns:
            baselines: 基线张量，形状为 [batch_size]
        """
        # 计算基线: E[Beta(alpha, beta)] = alpha / (alpha + beta)
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        """
        计算自适应更新速率 rho
        
        rho 根据策略变化（KL 散度）自适应调整:
        - 策略变化大（KL 大）-> rho 小 -> 基线更新慢
        - 策略变化小（KL 小）-> rho 大 -> 基线更新快
        
        Args:
            cur_mean_logprob: 当前平均 log 概率
            
        Returns:
            rho: 更新速率（在 [clip_lower, clip_upper] 范围内）
        """
        if self.rho_mode == 'constant':
            return self.rho_const
        
        if self.old_mean_logprob is None:
            return self.rho_const
        
        # 计算 KL 散度（策略变化程度）
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        
        # 根据 KL 散度计算 rho: rho = 2^(-kl / D_half)
        # KL 越大，rho 越小（基线更新越慢）
        rho = 2 ** (-kl / self.D_half)
        
        # 裁剪到合理范围
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        """
        更新基线参数
        
        根据当前奖励和策略变化更新 alpha 和 beta。
        
        Args:
            rewards: 奖励张量
            cur_logprobs: 当前 log 概率（可选，用于计算 KL）
            response_masks: 响应掩码（可选，用于计算平均 log 概率）
            
        Returns:
            rho: 使用的更新速率
        """
        # 计算自适应 rho（如果提供了 log 概率）
        if cur_logprobs is not None and response_masks is not None:
            # 计算平均 log 概率
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        # 归一化奖励到 [0, 1] 范围
        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        
        # 更新 Beta 分布参数
        # alpha 增加（奖励高时），beta 增加（奖励低时）
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        
        return rho


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励
    
    奖励由两部分组成:
    1. 格式奖励（仅推理模型）: 检查响应是否符合指定格式
    2. 奖励模型分数: 使用奖励模型评估响应质量
    
    Args:
        prompts: 提示列表，长度为 B（batch size）
        responses: 响应列表，长度为 B
        reward_model: 奖励模型（用于评估响应质量）
        reward_tokenizer: 奖励模型的 tokenizer
        
    Returns:
        rewards: 奖励张量，形状为 [B]
        
    奖励计算细节:
    - 对于推理模型（reasoning=1）:
      * 格式奖励: 检查是否包含正确的 <think> 和 <answer> 标签
      * 标记奖励: 检查标签数量是否正确（防止重复或缺失）
      * 奖励模型分数: 评估完整响应和答案部分（加权组合）
    - 对于普通模型（reasoning=0）:
      * 仅使用奖励模型分数
    """
    def reasoning_model_reward(rewards):
        """
        计算推理模型的格式和标记奖励
        
        这部分奖励鼓励模型生成符合指定格式的响应。
        """
        # 定义两种可能的格式模式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        # 检查每个响应是否匹配格式
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 格式奖励: 如果匹配格式，给予 0.5 分
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 标记奖励: 检查关键标签的数量是否正确
        def mark_num(text):
            """
            计算标记奖励
            
            检查关键标签的数量是否正确（每个标签应该只出现一次）。
            这防止了格式错误（如重复标签或缺失标签）。
            """
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励为零
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 如果是推理模型，添加格式和标记奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用奖励模型计算响应质量分数
    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0  # 奖励分数裁剪范围 [-3, 3]

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 解析 prompt 中的消息格式
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整的对话（包含生成的响应）
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            
            # 使用奖励模型评估完整响应
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            score = max(min(score, scale), -scale)  # 裁剪到 [-scale, scale]

            # 对于推理模型，额外评估答案部分
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 单独评估答案部分
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 加权组合: 完整响应 40%，答案部分 60%
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        # 将奖励模型分数转换为张量并添加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def spo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, value_tracker, start_step=0, wandb=None, total_tokens=0):
    """
    训练一个 epoch（SPO 版本）
    
    SPO 训练流程:
    1. 使用策略模型生成响应
    2. 使用奖励模型评估响应质量
    3. 使用自适应价值追踪器获取基线
    4. 计算优势（reward - baseline）
    5. 计算策略损失（包含优势项和 KL 惩罚项）
    6. 更新价值追踪器
    7. 反向传播更新策略模型
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        ref_model: 参考模型（冻结，用于计算 KL 散度）
        reward_model: 奖励模型（用于评估响应质量）
        reward_tokenizer: 奖励模型的 tokenizer
        value_tracker: 自适应价值追踪器（用于维护基线）
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        total_tokens (int): 已处理的 token 总数（生成的 tokens）
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用自适应基线而非组内相对优势
    - 基线跨 batch 更新，提供更稳定的估计
    - 通过 KL 散度自适应调整基线更新速率
    - 使用 KL 惩罚防止策略偏离参考模型太远
    """
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # 获取提示文本
        prompts = batch['prompt']  # list[str], length B
        
        # Tokenize 提示（左侧填充，用于生成）
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # 如果设置了最大序列长度，截断提示部分
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 使用策略模型生成响应
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)  # [B, P+R]

        # 提取生成的响应部分（去掉提示部分）
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B, R]

        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            计算每个 token 的 log 概率
            
            Args:
                mdl: 模型
                input_ids: 输入 token ID，形状为 [batch, seq_len]
                n_keep: 需要计算 log 概率的 token 数量（通常是响应长度）
                
            Returns:
                per_token_logps: 每个 token 的 log 概率，形状为 [batch, n_keep]
            """
            # 处理推理模式下的张量
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # 获取 logits（保留最后 n_keep+1 个位置，然后去掉最后一个）
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            # 对每个样本计算 log 概率
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 计算 log_softmax，然后提取目标 token 的 log 概率
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        # 计算策略模型的每个 token log 概率
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B, R]
        
        # 计算参考模型的每个 token log 概率（不计算梯度）
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B, R]

        # 解码生成的响应文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # list[str], length B
        
        # 计算奖励（包含格式奖励和奖励模型分数）
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B]

        # 从价值追踪器获取基线（归一化到 [0, 1]）
        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]

        # 将基线反归一化到与原始奖励相同的范围 [-3, 3]
        scale = 3.0
        unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
        
        # 计算优势（奖励 - 基线）
        advantages = rewards - unnormalized_baselines  # [B]

        # 裁剪优势到合理范围（防止梯度爆炸）
        # 注意：不再做 batch 内归一化，因为基线已经提供了跨 batch 的稳定基线
        advantages = advantages.clamp(-5.0, 5.0)

        # 创建完成掩码（标记有效响应部分，忽略 padding）
        is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B, R]
        
        # 统计本 step 生成的 token 数（按 completion_mask 计算有效 token）
        step_tokens = completion_mask.sum().item()
        if dist.is_initialized():
            # 分布式训练时乘以 GPU 数量
            step_tokens *= dist.get_world_size()
        total_tokens += step_tokens

        # 计算 KL 散度（用于惩罚策略偏离参考模型）
        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
        
        # 计算策略损失
        # 损失 = -log π * advantages + β * KL
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
        
        # 对每个响应计算平均损失，然后对所有响应求平均
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # scalar
        
        # 反向传播
        loss.backward()

        # 更新价值追踪器（使用当前奖励和 log 概率）
        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 清理 GPU 缓存
            torch.cuda.empty_cache()

        # 日志记录
        if step % args.log_interval == 0 or step == iters:
            now_time = time.time()
            policy_loss_val = loss.item() * args.accumulation_steps  # 恢复真实损失值
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']
            
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

            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}, '
                   f'it/s:{iter_per_sec:.2f} token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}')

            # 记录到 wandb
            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr,
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
            
            # 获取模型状态字典（处理 DDP 情况）
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            
            # 半精度保存以节省空间
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            
            # 保存完整检查点（包含优化器、调度器等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler, total_tokens=total_tokens)
            
            model.train()  # 切换回训练模式

        # 清理内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind SPO (Self-Play Optimization)")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='spo', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, 
                        help="初始学习率")
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, 
                        help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, 
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="模型保存间隔")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, 
                        help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                        help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, 
                        help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, 
                        help="生成的最大长度")
    
    # 数据和权重参数
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", 
                        help="RLAIF数据路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否自动检测&续训（0=否，1=是）")
    
    # SPO 特定参数
    parser.add_argument("--beta", type=float, default=0.02, 
                        help="KL惩罚系数（防止策略偏离参考模型太远）")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], 
                        help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", 
                        help="Reward模型路径")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SPO", 
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
    
    # 创建模型配置（包含生成长度）
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有可恢复的检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
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
        wandb_run_name = f"MiniMind-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========
    # 根据推理类型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # 使用绝对路径避免 HuggingFace 路径验证问题
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # Policy模型（正在训练的模型）
    model, tokenizer = init_model(lm_config, base_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    
    # Reference模型（冻结，用于计算 KL 散度）
    ref_model, _ = init_model(lm_config, base_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward模型（冻结，用于评估响应质量）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # Value Tracker（自适应价值追踪器，用于维护基线）
    value_tracker = AutoAdaptiveValueTracker(rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96)
    
    # 创建数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算总迭代次数和调度器
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    total_tokens = 0
    if ckp_data:
        # 恢复模型权重
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复调度器状态
        scheduler.load_state_dict(ckp_data['scheduler'])
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
            total_tokens = spo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, value_tracker, start_step, wandb, total_tokens)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            total_tokens = spo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, value_tracker, 0, wandb, total_tokens)
