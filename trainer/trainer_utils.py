"""
训练工具函数集合

本模块提供 MiniMind 训练过程中使用的各种工具函数和类:

1. 分布式训练支持
   - is_main_process: 检查是否为主进程
   - init_distributed_mode: 初始化分布式训练环境
   
2. 日志和检查点
   - Logger: 日志打印（只在主进程打印）
   - lm_checkpoint: 保存和加载检查点
   
3. 学习率调度
   - get_lr: 余弦退火学习率调度
   
4. 模型初始化
   - init_model: 初始化模型和 tokenizer
   - setup_seed: 设置随机种子
   
5. 数据加载
   - SkipBatchSampler: 支持跳过批次的采样器（用于断点续训）
   
6. SafeTensors 支持
   - save_model_safetensors: 保存为 SafeTensors 格式
   - load_model_safetensors: 从 SafeTensors 加载

使用示例:
    from trainer.trainer_utils import init_model, lm_checkpoint, Logger
    
    # 初始化模型
    model, tokenizer = init_model(config, 'pretrain')
    
    # 保存检查点
    lm_checkpoint(config, model=model, optimizer=optimizer, epoch=0, step=100)
    
    # 加载检查点
    ckp_data = lm_checkpoint(config)
"""

import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM
from model.gpu_utils import ensure_gpu_compatibility

# SafeTensors 支持（可选依赖）
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def is_main_process():
    """
    检查当前进程是否为主进程
    
    在分布式训练中，只有主进程（rank=0）应该执行某些操作，
    如打印日志、保存检查点等。
    
    Returns:
        bool: 如果是主进程或非分布式模式，返回 True
        
    Example:
        >>> if is_main_process():
        ...     print("这条消息只在主进程打印")
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    日志打印函数
    
    只在主进程打印日志，避免分布式训练时重复打印。
    
    Args:
        content: 要打印的内容
        
    Example:
        >>> Logger(f"Epoch 1, Loss: 0.5")
        Epoch 1, Loss: 0.5  # 只在主进程显示
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率调度
    
    学习率从 lr 开始，按余弦曲线衰减到 lr/10。
    这种调度方式在训练后期提供更小的学习率，有助于模型收敛。
    
    公式:
    lr_t = lr/10 + 0.5 * lr * (1 + cos(π * t / T))
    
    其中:
    - t: 当前步数
    - T: 总步数
    - lr: 初始学习率
    
    Args:
        current_step (int): 当前训练步数
        total_steps (int): 总训练步数
        lr (float): 初始学习率
        
    Returns:
        float: 当前步的学习率
        
    Example:
        >>> lr = get_lr(500, 1000, 1e-4)
        >>> print(f"当前学习率: {lr}")
    """
    # 余弦退火: 从 lr 衰减到 lr/10
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    """
    初始化分布式训练环境
    
    检测环境变量判断是否为分布式训练，如果是则初始化 NCCL 后端。
    
    分布式训练环境变量:
    - RANK: 全局进程排名
    - LOCAL_RANK: 本地 GPU 排名
    - WORLD_SIZE: 总进程数
    
    Returns:
        int: 本地 GPU 排名（非分布式模式返回 0）
        
    Example:
        >>> local_rank = init_distributed_mode()
        >>> device = f"cuda:{local_rank}"
    """
    # 检查是否为分布式模式
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    # 初始化进程组
    dist.init_process_group(backend="nccl")
    
    # 获取本地 GPU 排名并设置设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return local_rank


def setup_seed(seed: int):
    """
    设置随机种子以确保可重复性
    
    设置 Python、NumPy、PyTorch 的随机种子，
    并配置 cuDNN 为确定性模式。
    
    Args:
        seed (int): 随机种子值
        
    Note:
        确定性模式可能会降低性能，但能保证结果可重复。
        
    Example:
        >>> setup_seed(42)
        >>> # 现在所有随机操作都是可重复的
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
    
    # 确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clean_old_checkpoints(save_dir, weight, hidden_size, moe_path, keep_last_n):
    """
    清理旧的 checkpoint 文件，只保留最近 N 个
    
    在训练过程中会产生大量检查点文件，这个函数帮助管理磁盘空间。
    
    Args:
        save_dir (str): checkpoint 目录
        weight (str): 权重名称前缀
        hidden_size (int): 隐藏层大小
        moe_path (str): MoE 后缀（'_moe' 或 ''）
        keep_last_n (int): 保留最近 N 个 checkpoint
        
    Example:
        >>> _clean_old_checkpoints('checkpoints', 'pretrain', 512, '', 3)
        🗑️  已删除旧checkpoint: pretrain_512_step100.pth
    """
    import glob
    
    # 查找所有相关的 checkpoint 文件
    pattern_pth = f'{save_dir}/{weight}_{hidden_size}{moe_path}_step*.pth'
    pattern_safetensors = f'{save_dir}/{weight}_{hidden_size}{moe_path}_step*.safetensors'
    pattern_resume = f'{save_dir}/{weight}_{hidden_size}{moe_path}_step*_resume.pth'
    
    # 获取所有 checkpoint 文件（不包括 resume 文件）
    ckpt_files_pth = glob.glob(pattern_pth)
    ckpt_files_safetensors = glob.glob(pattern_safetensors)
    resume_files = glob.glob(pattern_resume)
    
    # 过滤掉 resume 文件
    ckpt_files_pth = [f for f in ckpt_files_pth if '_resume.pth' not in f]
    
    # 按修改时间排序（最新的在前）
    ckpt_files_pth.sort(key=os.path.getmtime, reverse=True)
    ckpt_files_safetensors.sort(key=os.path.getmtime, reverse=True)
    resume_files.sort(key=os.path.getmtime, reverse=True)
    
    # 删除旧的 checkpoint
    deleted_count = 0
    for files_list in [ckpt_files_pth, ckpt_files_safetensors, resume_files]:
        if len(files_list) > keep_last_n:
            for old_file in files_list[keep_last_n:]:
                try:
                    os.remove(old_file)
                    deleted_count += 1
                    Logger(f"🗑️  已删除旧checkpoint: {os.path.basename(old_file)}")
                except Exception as e:
                    Logger(f"⚠️  删除失败 {old_file}: {e}")
    
    if deleted_count > 0:
        Logger(f"✅ 清理完成，保留最近 {keep_last_n} 个checkpoint")


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, 
                  save_dir='../checkpoints', save_safetensors=True, keep_last_n=3, save_with_step=True, **kwargs):
    """
    保存或加载模型检查点
    
    这是一个双向函数:
    - 当 model 不为 None 时: 保存检查点
    - 当 model 为 None 时: 加载检查点
    
    保存的内容:
    1. 模型权重（.pth 和可选的 .safetensors）
    2. 恢复点（包含优化器状态、epoch、step 等）
    
    Args:
        lm_config: 模型配置对象
        weight (str): 权重名称前缀（如 'pretrain', 'full_sft'）
        model: 模型对象（None 表示加载模式）
        optimizer: 优化器对象
        epoch (int): 当前 epoch
        step (int): 当前 step
        wandb: wandb/swanlab 对象（用于保存 run ID）
        save_dir (str): 保存目录
        save_safetensors (bool): 是否同时保存 SafeTensors 格式
        keep_last_n (int): 保留最近 N 个 checkpoint（0 表示全部保留）
        save_with_step (bool): 是否在文件名中包含步数
        **kwargs: 其他需要保存的对象（如 scaler, scheduler）
        
    Returns:
        dict or None: 加载模式返回检查点数据，保存模式返回 None
        
    Example:
        # 保存检查点
        >>> lm_checkpoint(config, model=model, optimizer=optimizer, epoch=0, step=100)
        💾 已保存模型: checkpoints/pretrain_512_step100.pth
        
        # 加载检查点
        >>> ckp_data = lm_checkpoint(config)
        >>> model.load_state_dict(ckp_data['model'])
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    
    # 根据配置决定文件名
    if save_with_step and step > 0:
        # 带步数的文件名：pretrain_512_step3600.pth
        ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_step{step}.pth'
        resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_step{step}_resume.pth'
    else:
        # 传统文件名（会覆盖）
        ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
        resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ========== 保存模式 ==========
        from torch.nn.parallel import DistributedDataParallel
        
        # 获取模型状态字典
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        
        # 保存为 .pth 格式（半精度以节省空间）
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)  # 原子操作，防止写入中断
        Logger(f"💾 已保存模型: {ckp_path}")
        
        # 同时保存为 .safetensors 格式（如果启用且可用）
        if save_safetensors and SAFETENSORS_AVAILABLE:
            if save_with_step and step > 0:
                safetensors_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_step{step}.safetensors'
            else:
                safetensors_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.safetensors'
            save_model_safetensors(model, safetensors_path, half_precision=True)
        
        # 获取 wandb run ID（用于断点续训）
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建恢复数据
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # 保存额外的对象（如 scaler, scheduler）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        # 保存恢复点
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        Logger(f"💾 已保存恢复点: {resume_path}")
        
        # 清理旧的 checkpoint（如果启用）
        if keep_last_n > 0 and save_with_step:
            _clean_old_checkpoints(save_dir, weight, lm_config.hidden_size, moe_path, keep_last_n)
            
    else:
        # ========== 加载模式 ==========
        import glob
        
        # 先尝试传统文件名
        if os.path.exists(resume_path):
            latest_resume = resume_path
        else:
            # 查找所有带步数的 resume 文件
            pattern = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_step*_resume.pth'
            resume_files = glob.glob(pattern)
            
            if resume_files:
                # 按修改时间排序，取最新的
                latest_resume = max(resume_files, key=os.path.getmtime)
                Logger(f"📂 找到最新checkpoint: {os.path.basename(latest_resume)}")
            else:
                Logger(f"⚠️  未找到checkpoint: {resume_path}")
                return None
        
        # 加载检查点数据
        ckp_data = torch.load(latest_resume, map_location='cpu')
        
        # 处理 GPU 数量变化的情况
        saved_ws = ckp_data.get('world_size', 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            # 调整 step 以适应新的 GPU 数量
            ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
            Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
        
        return ckp_data


def save_model_safetensors(model, save_path, half_precision=True):
    """
    保存模型为 SafeTensors 格式
    
    SafeTensors 是一种安全、快速的模型权重格式:
    - 安全: 纯数据格式，不会执行恶意代码
    - 快速: 加载速度比 pickle 快 2-3 倍
    - 跨框架: 支持 PyTorch/TensorFlow/JAX
    
    Args:
        model: 模型对象
        save_path (str): 保存路径（.safetensors）
        half_precision (bool): 是否使用半精度保存
        
    Returns:
        bool: 是否保存成功
        
    Example:
        >>> save_model_safetensors(model, 'model.safetensors')
        ✅ 模型已保存为 SafeTensors: model.safetensors
    """
    if not SAFETENSORS_AVAILABLE:
        Logger("⚠️  safetensors 未安装，跳过 .safetensors 保存")
        Logger("   安装方法: pip install safetensors")
        return False
    
    try:
        from torch.nn.parallel import DistributedDataParallel
        
        # 获取 state_dict
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        # 转换为半精度并确保连续存储（safetensors 要求）
        if half_precision:
            state_dict = {k: v.half().contiguous() for k, v in state_dict.items()}
        else:
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        
        # 保存
        save_file(state_dict, save_path)
        Logger(f"✅ 模型已保存为 SafeTensors: {save_path}")
        return True
        
    except Exception as e:
        Logger(f"❌ SafeTensors 保存失败: {e}")
        return False


def load_model_safetensors(save_path, device='cpu'):
    """
    从 SafeTensors 文件加载模型权重
    
    Args:
        save_path (str): .safetensors 文件路径
        device (str): 加载到的设备
        
    Returns:
        dict: state_dict
        
    Raises:
        ImportError: 如果 safetensors 未安装
        
    Example:
        >>> state_dict = load_model_safetensors('model.safetensors')
        >>> model.load_state_dict(state_dict)
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors 未安装，请运行: pip install safetensors")
    
    try:
        state_dict = load_file(save_path, device=str(device))
        Logger(f"✅ 从 SafeTensors 加载模型: {save_path}")
        return state_dict
    except Exception as e:
        Logger(f"❌ SafeTensors 加载失败: {e}")
        raise


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', 
               device='cuda', auto_gpu_config=True, prefer_safetensors=True):
    """
    初始化模型和 tokenizer
    
    这是模型初始化的主要入口函数，处理:
    1. GPU 兼容性检测和配置
    2. Tokenizer 加载
    3. 模型创建
    4. 权重加载（支持 .pth 和 .safetensors）
    
    Args:
        lm_config: 模型配置对象
        from_weight (str): 要加载的权重名称（'none' 表示不加载）
        tokenizer_path (str): tokenizer 路径
        save_dir (str): 权重保存目录
        device (str): 目标设备
        auto_gpu_config (bool): 是否自动检测并配置 GPU 兼容性
        prefer_safetensors (bool): 是否优先加载 .safetensors 格式
        
    Returns:
        Tuple[model, tokenizer]: 初始化后的模型和 tokenizer
        
    Example:
        >>> config = MiniMindConfig(hidden_size=512)
        >>> model, tokenizer = init_model(config, 'pretrain')
        所加载Model可训练参数：26.000 百万
    """
    # GPU 兼容性检测和自动配置（支持 sm_120 Blackwell 架构）
    if auto_gpu_config and 'cuda' in device:
        lm_config = ensure_gpu_compatibility(lm_config)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 创建模型
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        
        # 尝试加载权重
        weights = None
        weight_path = None
        
        # 优先尝试加载 .safetensors 格式
        if prefer_safetensors and SAFETENSORS_AVAILABLE:
            safetensors_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.safetensors'
            if os.path.exists(safetensors_path):
                try:
                    weights = load_model_safetensors(safetensors_path, device=device)
                    weight_path = safetensors_path
                except Exception as e:
                    Logger(f"⚠️  SafeTensors 加载失败，尝试 .pth 格式: {e}")
        
        # 如果没有加载成功，回退到 .pth 格式
        if weights is None:
            pth_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(pth_path):
                weights = torch.load(pth_path, map_location=device)
                weight_path = pth_path
                Logger(f"从 PyTorch 格式加载: {pth_path}")
            else:
                Logger(f"⚠️  未找到权重文件: {pth_path}")
        
        # 加载权重
        if weights is not None:
            model.load_state_dict(weights, strict=False)
        else:
            Logger(f"⚠️  未找到可用的权重文件，使用随机初始化")

    # 打印模型参数量
    Logger(f'所加载Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    支持跳过批次的采样器
    
    用于断点续训时跳过已经训练过的批次。
    
    工作原理:
    1. 包装一个基础采样器
    2. 按批次大小分组
    3. 跳过前 skip_batches 个批次
    4. 返回剩余批次
    
    Attributes:
        sampler: 基础采样器
        batch_size (int): 批次大小
        skip_batches (int): 要跳过的批次数
        
    Example:
        >>> sampler = DistributedSampler(dataset)
        >>> batch_sampler = SkipBatchSampler(sampler, batch_size=32, skip_batches=100)
        >>> loader = DataLoader(dataset, batch_sampler=batch_sampler)
        >>> # loader 会跳过前 100 个批次
    """
    
    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        初始化跳过批次采样器
        
        Args:
            sampler: 基础采样器（如 DistributedSampler）
            batch_size (int): 批次大小
            skip_batches (int): 要跳过的批次数
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        迭代批次
        
        Yields:
            list: 每个批次的样本索引列表
        """
        batch = []
        skipped = 0
        
        for idx in self.sampler:
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                # 检查是否需要跳过
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                    
                yield batch
                batch = []
        
        # 处理最后一个不完整的批次
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        返回批次总数（减去跳过的批次）
        
        Returns:
            int: 实际返回的批次数
        """
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
