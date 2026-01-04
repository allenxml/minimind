"""
MiniMind PPO（Proximal Policy Optimization）训练脚本

本脚本实现了近端策略优化算法，用于强化学习对齐。

PPO 原理:
PPO 是一种策略梯度算法，通过限制策略更新的幅度来稳定训练。

核心思想:
- 使用 Actor-Critic 架构：Actor 生成响应，Critic 评估价值
- 通过裁剪机制限制策略更新幅度，防止策略变化过大
- 使用 KL 散度惩罚防止策略偏离参考模型太远

PPO 损失函数:
L = L_policy + c_vf * L_value + c_kl * KL(π||π_ref)

其中:
- L_policy: 策略损失（带裁剪的 PPO 目标）
- L_value: 价值损失（MSE）
- KL: 与参考模型的 KL 散度
- c_vf, c_kl: 系数

PPO 裁剪机制:
- 计算重要性采样比率: r = π_new / π_old
- 裁剪比率到 [1-ε, 1+ε] 范围内
- 取裁剪前后的最小值，防止策略更新过大

与 GRPO 的区别:
1. 需要训练 Critic 模型（价值函数）
2. 使用绝对优势（reward - value）而非相对优势
3. 使用裁剪机制而非组内归一化

使用方法:
    python train_ppo.py --epochs 1 --batch_size 2 --learning_rate 8e-8
"""

import os
import sys

# 设置包名以支持相对导入
__package__ = "trainer"
# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息
warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    """
    自定义的 Critic 模型，用于估计状态价值
    
    Critic 模型继承自 MiniMindForCausalLM，但将语言模型头替换为价值头，
    输出每个位置的价值估计（标量）。
    
    架构:
    - 基础 Transformer 编码器（与 Actor 共享）
    - 价值头（value_head）：将隐藏状态映射到标量价值
    
    用途:
    - 估计生成序列的价值，用于计算优势（advantage）
    - 优势 = 奖励 - 价值估计
    """
    def __init__(self, params):
        super().__init__(params)
        # 替换语言模型头为价值头（输出单一标量）
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，计算每个位置的价值估计
        
        Args:
            input_ids: 输入 token ID
            attention_mask: 注意力掩码
            **kwargs: 其他参数
            
        Returns:
            values: 每个位置的价值估计，形状为 [batch_size, seq_len]
        """
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 应用层归一化
        hidden_states = self.model.norm(outputs[0])
        # 使用价值头获取价值估计（每个位置一个标量）
        values = self.value_head(hidden_states).squeeze(-1)
        return values


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
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
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
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
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
        for prompt, response in zip(prompts, responses):
            # 解析 prompt 中的消息格式
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整的对话（包含生成的响应）
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            
            # 使用奖励模型评估完整响应
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 裁剪奖励分数到合理范围
            scale = 3.0
            score = max(min(score, scale), -scale)

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


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None, total_tokens=0):
    """
    训练一个 epoch（PPO 版本）
    
    PPO 训练流程:
    1. 使用 Actor 模型生成响应
    2. 使用奖励模型评估响应质量
    3. 使用 Critic 模型估计价值
    4. 计算优势（reward - value）
    5. 计算策略损失（带裁剪的 PPO 目标）
    6. 计算价值损失和 KL 散度损失
    7. 反向传播更新 Actor 和 Critic
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        old_actor_model: 旧 Actor 模型（用于计算重要性采样比率）
        ref_model: 参考模型（冻结，用于计算 KL 散度）
        actor_scheduler: Actor 学习率调度器
        critic_scheduler: Critic 学习率调度器
        reward_model: 奖励模型（用于评估响应质量）
        reward_tokenizer: 奖励模型的 tokenizer
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        total_tokens (int): 已处理的 token 总数（生成的 tokens）
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用裁剪机制限制策略更新幅度
    - 定期更新 old_actor_model 以计算重要性采样比率
    - 使用 KL 散度惩罚防止策略偏离参考模型太远
    """
    actor_model.train()
    critic_model.train()
    
    # 时间和速度统计
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = total_tokens
    last_log_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        # 获取提示文本
        prompts = batch["prompt"]  # list[str], length B
        
        # Tokenize 提示
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_lengths = torch.full((enc.input_ids.size(0),), enc.input_ids.shape[1], dtype=torch.long, device=enc.input_ids.device)  # [B]

        # 使用 Actor 模型生成响应
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R]

        # 解码生成的响应文本
        responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
        
        # 计算奖励（包含格式奖励和奖励模型分数）
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # 使用 Critic 模型估计价值
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        
        # 获取序列最后一个有效位置的价值（作为整体价值估计）
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        
        # 计算优势（奖励 - 价值估计）
        advantages = rewards - values.detach()  # [B]

        # 计算当前 Actor 的 log 概率
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]（shifted for next token prediction）
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        # 创建响应掩码（只计算响应部分的损失，不包括提示部分）
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        
        # 统计本 step 生成的 token 数（按 final_mask 计算有效 token）
        step_tokens = final_mask.sum().item()
        if dist.is_initialized():
            # 分布式训练时乘以 GPU 数量
            step_tokens *= dist.get_world_size()
        total_tokens += step_tokens
        
        # 计算序列级别的 log 概率（只对响应部分求和）
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # 计算旧 Actor 和参考模型的 log 概率（用于重要性采样和 KL 散度）
        with torch.no_grad():
            # 旧 Actor 的 log 概率（用于计算重要性采样比率）
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # 参考模型的 log 概率（用于计算 KL 散度）
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # 计算 KL 散度（用于监控和惩罚）
        kl = (actor_logp - old_logp).mean()  # scalar（当前策略与旧策略的 KL）
        kl_ref = (actor_logp - ref_logp).mean()  # scalar（当前策略与参考策略的 KL）
        
        # 计算重要性采样比率
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO 裁剪目标: min(r * A, clip(r, 1-ε, 1+ε) * A)
        surr1 = ratio * advantages  # [B]（未裁剪）
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]（裁剪后）
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar（取最小值，防止策略更新过大）
        
        # 价值损失（MSE）
        value_loss = F.mse_loss(values, rewards)  # scalar
        
        # 总损失 = 策略损失 + 价值损失系数 * 价值损失 + KL 散度系数 * KL 散度
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        
        # 反向传播
        loss.backward()

        # 梯度累积完成后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            
            # 更新参数
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            
            # 清空梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            # 清理 GPU 缓存
            torch.cuda.empty_cache()

        # 日志记录
        if is_main_process():
            now_time = time.time()
            
            # 计算平均响应长度
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            # 获取训练指标
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']
            
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

            # 记录到 wandb
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "it_per_sec": iter_per_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "total_tokens": total_tokens
                })

            # 打印日志
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}, "
                   f"it/s:{iter_per_sec:.2f} token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}")

        # 定期更新 old_actor_model（用于计算重要性采样比率）
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # 保存检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()  # 切换到评估模式
            
            # 构建检查点路径
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取 Actor 模型状态字典（处理 DDP 情况）
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            
            # 半精度保存以节省空间
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # 保存完整检查点（包括 Actor、Critic、优化器等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler, total_tokens=total_tokens)
            
            actor_model.train()  # 切换回训练模式
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, 
                        help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, 
                        help="Critic学习率")
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
    
    # PPO 特定参数
    parser.add_argument("--clip_epsilon", type=float, default=0.1, 
                        help="PPO裁剪参数（限制策略更新幅度）")
    parser.add_argument("--vf_coef", type=float, default=0.5, 
                        help="Value function系数（价值损失的权重）")
    parser.add_argument("--kl_coef", type=float, default=0.02, 
                        help="KL散度惩罚系数（防止策略偏离参考模型太远）")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], 
                        help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, 
                        help="更新old_actor_model的频率（用于计算重要性采样比率）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", 
                        help="Reward模型路径")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", 
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
    
    # 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # 根据推理类型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # 使用绝对路径避免 HuggingFace 路径验证问题
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    
    # Actor模型（正在训练的模型）
    actor_model, tokenizer = init_model(lm_config, base_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    tokenizer.padding_side = 'left'  # PPO需要左侧padding（用于生成）
    
    # Old Actor模型（用于计算重要性采样比率）
    old_actor_model, _ = init_model(lm_config, base_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # Reference模型（冻结，用于计算 KL 散度）
    ref_model, _ = init_model(lm_config, base_weight, tokenizer_path=tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Critic模型（用于估计价值）
    # 从基础权重加载，然后替换语言模型头为价值头
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)  # strict=False 因为价值头是新的
    critic_model = critic_model.to(args.device)
    
    # Reward模型（冻结，用于评估响应质量）
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # 创建数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总迭代次数和调度器
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        # 恢复优化器状态
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        # 恢复调度器状态
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        total_tokens = ckp_data.get('total_tokens', 0)
    
    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        # 忽略 RoPE 的 buffer（不需要同步）
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DDP 包装模型
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch（确保每个 epoch 的数据顺序不同）
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点：跳过已训练的步数
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_tokens = ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb, total_tokens)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            total_tokens = ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb, total_tokens)
