"""
语言模型数据集实现

本模块实现了 MiniMind 训练所需的各种数据集类，支持不同的训练阶段:

1. PretrainDataset - 预训练数据集
   - 用于无监督预训练
   - 输入: 纯文本
   - 目标: 预测下一个 token

2. SFTDataset - 监督微调数据集
   - 用于指令微调
   - 输入: 对话格式（system/user/assistant）
   - 目标: 只计算 assistant 回复的损失

3. DPODataset - 直接偏好优化数据集
   - 用于 DPO 训练
   - 输入: 问题 + 选中回答 + 拒绝回答
   - 目标: 让模型偏好选中的回答

4. RLAIFDataset - 强化学习数据集
   - 用于 PPO/GRPO/SPO 训练
   - 输入: 提示词
   - 目标: 通过奖励模型优化生成

数据格式:
- 预训练: {"text": "..."}
- SFT: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- DPO: {"prompt": "...", "chosen": "...", "rejected": "..."}
- RLAIF: {"prompt": "..."}
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset

# 禁用 tokenizers 的并行化，避免在多进程 DataLoader 中出现警告
# 这是因为 tokenizers 在 fork 后可能会出现死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集
    
    用于语言模型的无监督预训练阶段。模型学习预测下一个 token，
    从而学习语言的统计规律和知识。
    
    数据格式:
    每行一个 JSON 对象，包含 "text" 字段:
    {"text": "这是一段预训练文本..."}
    
    处理流程:
    1. 读取 JSONL 文件中的所有文本
    2. 使用 tokenizer 将文本转换为 token ID
    3. 截断或填充到固定长度
    4. 创建输入-输出对（X 是输入，Y 是右移一位的目标）
    5. 创建损失掩码（忽略填充位置）
    
    Attributes:
        data (list): 所有样本的列表
        tokenizer: 分词器
        max_length (int): 最大序列长度
        bos_id (int): 序列开始标记 ID
        eos_id (int): 序列结束标记 ID
        pad_id (int): 填充标记 ID
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        
        Args:
            data_path (str): JSONL 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_length (int): 最大序列长度，超过会截断
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 获取特殊 token ID
        self.bos_id = tokenizer('<|im_start|>').input_ids[0]  # 序列开始
        self.eos_id = tokenizer('<|im_end|>').input_ids[0]    # 序列结束
        self.pad_id = tokenizer('<|endoftext|>').input_ids[0] # 填充
        
        # 加载数据
        self.data = []
        total_chars = 0  # 统计总字符数用于估算 token
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
                # 统计字符数（快速）
                total_chars += len(sample.get('text', ''))
        
        # 使用采样估算 token 数量（避免全量 tokenize 太慢）
        sample_size = min(1000, len(self.data))
        if sample_size > 0:
            sample_indices = random.sample(range(len(self.data)), sample_size)
            sample_tokens = 0
            sample_chars = 0
            for idx in sample_indices:
                text = self.data[idx].get('text', '')
                sample_chars += len(text)
                sample_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            
            # 根据采样估算总 token 数
            if sample_chars > 0:
                tokens_per_char = sample_tokens / sample_chars
                estimated_tokens = int(total_chars * tokens_per_char)
            else:
                estimated_tokens = 0
        else:
            estimated_tokens = 0
        
        self.total_tokens = estimated_tokens
        
        # 格式化 token 数量显示
        if estimated_tokens >= 1e9:
            token_str = f"~{estimated_tokens / 1e9:.2f}B"
        elif estimated_tokens >= 1e6:
            token_str = f"~{estimated_tokens / 1e6:.2f}M"
        elif estimated_tokens >= 1e3:
            token_str = f"~{estimated_tokens / 1e3:.2f}K"
        else:
            token_str = f"~{estimated_tokens}"
        
        print(f"预训练数据集加载完成: {len(self.data)} 条样本, {token_str} tokens")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        处理流程:
        1. 获取原始文本
        2. 添加 BOS 和 EOS 标记
        3. Tokenize 并截断
        4. 创建输入 X 和目标 Y（Y = X 右移一位）
        5. 创建损失掩码
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - X: 输入 token ID，形状为 (max_length,)
                - Y: 目标 token ID，形状为 (max_length,)
                - loss_mask: 损失掩码，形状为 (max_length,)
        """
        sample = self.data[idx]
        text = sample['text']
        
        # 添加特殊标记并 tokenize
        # 格式: <|im_start|>文本<|im_end|>
        text = f"<|im_start|>{text}<|im_end|>"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze(0)  # (max_length,)
        
        # 创建输入和目标
        # X: 输入序列（去掉最后一个 token）
        # Y: 目标序列（去掉第一个 token，即右移一位）
        X = input_ids[:-1]
        Y = input_ids[1:]
        
        # 创建损失掩码
        # 只计算非填充位置的损失
        loss_mask = (Y != self.pad_id).float()
        
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    监督微调（SFT）数据集
    
    用于指令微调阶段。模型学习根据用户指令生成合适的回复。
    只计算 assistant 回复部分的损失，不计算用户输入部分。
    
    数据格式:
    每行一个 JSON 对象，包含 "conversations" 字段:
    {
        "conversations": [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
    }
    
    处理流程:
    1. 使用 tokenizer 的 chat_template 格式化对话
    2. Tokenize 整个对话
    3. 创建损失掩码，只在 assistant 回复位置为 1
    
    Attributes:
        data (list): 所有样本的列表
        tokenizer: 分词器
        max_length (int): 最大序列长度
        bos_id (int): 序列开始标记 ID
        eos_id (int): 序列结束标记 ID
        pad_id (int): 填充标记 ID
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化 SFT 数据集
        
        Args:
            data_path (str): JSONL 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_length (int): 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 获取特殊 token ID
        self.bos_id = tokenizer('<|im_start|>').input_ids[0]
        self.eos_id = tokenizer('<|im_end|>').input_ids[0]
        self.pad_id = tokenizer('<|endoftext|>').input_ids[0]
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
        
        print(f"SFT 数据集加载完成: {len(self.data)} 条样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def _create_chat_prompt(self, conversations):
        """
        使用 chat_template 创建对话提示
        
        Args:
            conversations (list): 对话列表
            
        Returns:
            str: 格式化后的对话文本
        """
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

    def _get_assistant_mask(self, input_ids, conversations):
        """
        创建 assistant 回复的掩码
        
        只有 assistant 的回复部分需要计算损失，
        用户输入和系统提示不计算损失。
        
        Args:
            input_ids (Tensor): token ID 序列
            conversations (list): 原始对话列表
            
        Returns:
            Tensor: 损失掩码，assistant 回复位置为 1，其他为 0
        """
        # 初始化掩码为全 0
        mask = torch.zeros_like(input_ids, dtype=torch.float)
        
        # 找到每个 assistant 回复的位置
        # 策略: 找到 "assistant\n" 后的内容直到 "<|im_end|>"
        input_ids_list = input_ids.tolist()
        
        # 获取 assistant 标记的 token ID
        assistant_start = self.tokenizer('assistant\n').input_ids
        
        i = 0
        while i < len(input_ids_list):
            # 检查是否是 assistant 回复的开始
            if i + len(assistant_start) <= len(input_ids_list):
                if input_ids_list[i:i+len(assistant_start)] == assistant_start:
                    # 找到 assistant 回复，标记直到 eos
                    j = i + len(assistant_start)
                    while j < len(input_ids_list) and input_ids_list[j] != self.eos_id:
                        mask[j] = 1.0
                        j += 1
                    # 也标记 eos token
                    if j < len(input_ids_list):
                        mask[j] = 1.0
                    i = j + 1
                    continue
            i += 1
        
        return mask

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - X: 输入 token ID
                - Y: 目标 token ID
                - loss_mask: 损失掩码（只在 assistant 回复处为 1）
        """
        sample = self.data[idx]
        conversations = sample['conversations']
        
        # 格式化对话
        text = self._create_chat_prompt(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        
        # 创建输入和目标
        X = input_ids[:-1]
        Y = input_ids[1:]
        
        # 创建损失掩码
        # 首先排除填充位置
        loss_mask = (Y != self.pad_id).float()
        
        # 然后只保留 assistant 回复部分
        assistant_mask = self._get_assistant_mask(input_ids, conversations)
        loss_mask = loss_mask * assistant_mask[1:]  # 对齐到 Y
        
        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    直接偏好优化（DPO）数据集
    
    用于 DPO 训练阶段。模型学习偏好人类选择的回答，
    而不是被拒绝的回答。
    
    数据格式:
    每行一个 JSON 对象:
    {
        "prompt": "用户问题...",
        "chosen": "好的回答...",
        "rejected": "差的回答..."
    }
    
    或者对话格式:
    {
        "conversations": [...],
        "chosen": "好的回答...",
        "rejected": "差的回答..."
    }
    
    DPO 损失函数:
    L = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    其中:
    - y_w: 选中的回答 (chosen)
    - y_l: 拒绝的回答 (rejected)
    - π: 策略模型
    - π_ref: 参考模型
    - β: 温度参数
    
    Attributes:
        data (list): 所有样本的列表
        tokenizer: 分词器
        max_length (int): 最大序列长度
    """
    
    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        初始化 DPO 数据集
        
        Args:
            data_path (str): JSONL 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_length (int): 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 获取特殊 token ID
        self.bos_id = tokenizer('<|im_start|>').input_ids[0]
        self.eos_id = tokenizer('<|im_end|>').input_ids[0]
        self.pad_id = tokenizer('<|endoftext|>').input_ids[0]
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
        
        print(f"DPO 数据集加载完成: {len(self.data)} 条样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def _process_sample(self, prompt, response):
        """
        处理单个提示-回复对
        
        Args:
            prompt (str): 提示文本
            response (str): 回复文本
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (X, Y, loss_mask)
        """
        # 构建完整对话
        if isinstance(prompt, list):
            # 对话格式
            conversations = prompt + [{"role": "assistant", "content": response}]
            text = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # 简单格式
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        
        # 创建输入和目标
        X = input_ids[:-1]
        Y = input_ids[1:]
        
        # 创建损失掩码（只计算回复部分）
        loss_mask = (Y != self.pad_id).float()
        
        return X, Y, loss_mask

    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回选中和拒绝两个版本的数据。
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含以下键:
                - x_chosen: 选中回答的输入
                - y_chosen: 选中回答的目标
                - mask_chosen: 选中回答的掩码
                - x_rejected: 拒绝回答的输入
                - y_rejected: 拒绝回答的目标
                - mask_rejected: 拒绝回答的掩码
        """
        sample = self.data[idx]
        
        # 获取提示
        if 'conversations' in sample:
            prompt = sample['conversations']
        else:
            prompt = sample['prompt']
        
        # 处理选中的回答
        x_chosen, y_chosen, mask_chosen = self._process_sample(prompt, sample['chosen'])
        
        # 处理拒绝的回答
        x_rejected, y_rejected, mask_rejected = self._process_sample(prompt, sample['rejected'])
        
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }


class RLAIFDataset(Dataset):
    """
    强化学习（RLAIF）数据集
    
    用于 PPO、GRPO、SPO 等强化学习训练。
    只提供提示词，模型生成回复后由奖励模型评分。
    
    数据格式:
    每行一个 JSON 对象:
    {"prompt": "用户问题..."}
    
    或者对话格式:
    {
        "conversations": [
            {"role": "user", "content": "问题..."}
        ]
    }
    
    工作流程:
    1. 数据集提供提示词
    2. 模型生成回复
    3. 奖励模型对回复评分
    4. 使用 RL 算法更新模型
    
    Attributes:
        data (list): 所有样本的列表
        tokenizer: 分词器
        max_length (int): 最大序列长度
    """
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化 RLAIF 数据集
        
        Args:
            data_path (str): JSONL 数据文件路径
            tokenizer: HuggingFace tokenizer
            max_length (int): 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
        
        print(f"RLAIF 数据集加载完成: {len(self.data)} 条样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含 'prompt' 键的字典
        """
        sample = self.data[idx]
        
        # 获取或构建提示
        if 'conversations' in sample:
            # 使用 chat_template 格式化
            prompt = self.tokenizer.apply_chat_template(
                sample['conversations'],
                tokenize=False,
                add_generation_prompt=True  # 添加 assistant 开始标记
            )
        else:
            prompt = sample['prompt']
        
        return {'prompt': prompt}


def collate_fn_dpo(batch):
    """
    DPO 数据集的批处理函数
    
    将多个样本组合成一个批次。
    
    Args:
        batch (list): 样本列表
        
    Returns:
        dict: 批处理后的数据
    """
    return {
        'x_chosen': torch.stack([item['x_chosen'] for item in batch]),
        'y_chosen': torch.stack([item['y_chosen'] for item in batch]),
        'mask_chosen': torch.stack([item['mask_chosen'] for item in batch]),
        'x_rejected': torch.stack([item['x_rejected'] for item in batch]),
        'y_rejected': torch.stack([item['y_rejected'] for item in batch]),
        'mask_rejected': torch.stack([item['mask_rejected'] for item in batch])
    }


def collate_fn_rlaif(batch):
    """
    RLAIF 数据集的批处理函数
    
    Args:
        batch (list): 样本列表
        
    Returns:
        dict: 批处理后的数据
    """
    return {
        'prompt': [item['prompt'] for item in batch]
    }
