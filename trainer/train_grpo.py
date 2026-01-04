"""
MiniMind GRPO（Group Relative Policy Optimization）训练脚本

本脚本实现了分组相对策略优化算法，用于强化学习对齐。

GRPO 原理:
GRPO 是一种基于相对策略优化的强化学习算法，通过对比同一 prompt 的多个生成样本
来学习更好的策略。

核心思想:
- 对每个 prompt 生成多个候选回答（num_generations 个）
- 使用奖励模型评估每个回答的质量
- 计算组内相对优势（相对于组内平均奖励）
- 通过策略梯度优化，提升高奖励回答的概率

GRPO 损失函数:
L = -E[exp(log π(y|x) - log π(y|x).detach()) * advantages - β * KL(π||π_ref)]

其中:
- advantages: 组内归一化的相对优势
- β: KL 惩罚系数，防止策略偏离参考模型太远
- π_ref: 参考模型（冻结）

与 PPO 的区别:
1. 不需要 Critic 模型（价值函数）
2. 使用组内相对优势而非绝对优势
3. 计算更简单，训练更稳定

使用方法:
    python train_grpo.py --epochs 1 --batch_size 2 --num_generations 8
"""

import os
import sys
import time

# 设置包名以支持相对导入
__package__ = "trainer"
# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
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


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励
    
    奖励由两部分组成:
    1. 格式奖励（仅推理模型）: 检查响应是否符合指定格式
    2. 奖励模型分数: 使用奖励模型评估响应质量
    
    Args:
        prompts: 提示列表，长度为 B（batch size）
        responses: 响应列表，长度为 B * num_generations
        reward_model: 奖励模型（用于评估响应质量）
        reward_tokenizer: 奖励模型的 tokenizer
        
    Returns:
        rewards: 奖励张量，形状为 [B * num_generations]
        
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

        # 标记奖励: 检查标签数量
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
        batch_size = len(prompts)
        scale = 3.0  # 奖励分数裁剪范围 [-3, 3]

        # 遍历每个 prompt 和对应的多个生成样本
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

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


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None, total_tokens=0):
    """
    训练一个 epoch（GRPO 版本）
    
    GRPO 训练流程:
    1. 对每个 prompt 生成多个候选响应（num_generations 个）
    2. 使用奖励模型评估每个响应的质量
    3. 计算组内相对优势（相对于组内平均奖励）
    4. 计算策略损失（包含优势项和 KL 惩罚项）
    5. 反向传播更新策略模型
    
    Args:
        epoch (int): 当前 epoch 索引
        loader (DataLoader): 数据加载器
        iters (int): 总迭代次数
        ref_model: 参考模型（冻结，用于计算 KL 散度）
        reward_model: 奖励模型（用于评估响应质量）
        reward_tokenizer: 奖励模型的 tokenizer
        start_step (int): 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志对象
        total_tokens (int): 已处理的 token 总数（生成的 tokens）
        
    Returns:
        int: 更新后的 token 总数
        
    训练细节:
    - 使用组内相对优势，而非绝对优势
    - 通过 KL 惩罚防止策略偏离参考模型太远
    - 使用梯度累积以模拟更大的批次
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

        # 生成多个候选响应（每个 prompt 生成 num_generations 个）
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

        # 提取生成的响应部分（去掉提示部分）
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
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
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        
        # 计算参考模型的每个 token log 概率（不计算梯度）
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # 解码生成的响应文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # 计算奖励（包含格式奖励和奖励模型分数）
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # 计算组内相对优势
        # 将奖励按 prompt 分组: [B*num_gen] -> [B, num_gen]
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        
        # 计算每个组的均值和标准差
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        
        # 计算组内相对优势: (reward - group_mean) / group_std
        # 裁剪到 [-10, 10] 防止极端值
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        
        # 全局归一化优势（可选，进一步稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # 创建完成掩码（标记有效响应部分，忽略 padding）
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]
        
        # 统计本 step 生成的 token 数（按 completion_mask 计算有效 token）
        step_tokens = completion_mask.sum().item()
        if dist.is_initialized():
            # 分布式训练时乘以 GPU 数量
            step_tokens *= dist.get_world_size()
        total_tokens += step_tokens

        # 计算 KL 散度（用于惩罚策略偏离参考模型）
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        
        # 计算策略损失
        # 损失 = -[exp(log π - log π.detach()) * advantages - β * KL]
        # 这里使用 exp(log π - log π.detach()) 来避免数值不稳定
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # [B*num_gen, R]
        
        # 对每个响应计算平均损失，然后对所有响应求平均
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # scalar
        
        # 反向传播
        loss.backward()

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
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}, '
                   f'it/s:{iter_per_sec:.2f} token/s:{tokens_per_sec:.0f} total_tokens:{int(total_tokens)}')

            # 记录到 wandb
            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
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
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()
    
    return total_tokens


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, 
                        help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, 
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
    
    # GRPO 特定参数
    parser.add_argument("--num_generations", type=int, default=8, 
                        help="每个prompt生成的样本数（用于计算组内相对优势）")
    parser.add_argument("--beta", type=float, default=0.02, 
                        help="KL惩罚系数（防止策略偏离参考模型太远）")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], 
                        help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", 
                        help="Reward模型路径")
    
    # 日志参数
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", 
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
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
        total_tokens = ckp_data.get('total_tokens', 0)
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
            total_tokens = grpo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, start_step, wandb, total_tokens)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            total_tokens = grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb, total_tokens)
